import os
import torch
import torch.nn as nn
from typing import Tuple, Optional, Any
from models.base_nets import DifferenceExpansion, ResInvertibleBlock, HistogramShifting, AffineInvertibleBlock
from models.utils import cleanup_old_checkpoints


class LatentDifferenceExpansion(nn.Module):
    """
    Latent Difference Expansion Network for Reversible Watermarking.

    This model combines invertible neural networks (INN) with the classic difference
    expansion (DE) technique to achieve high-capacity, perfectly reversible image
    watermarking. Unlike traditional pixel-domain DE, this method operates in the
    learned latent space, where features are more decorrelated and better suited
    for difference expansion embedding. The architecture leverages the invertibility
    property of coupling layers to ensure lossless image recovery after watermark
    extraction.

    Architecture Overview:
        ┌─────────────────────────────────────────────────────────────────┐
        │                   Latent Difference Expansion                   │
        ├─────────────────────────────────────────────────────────────────┤
        │  Cover Image → Latent Encoding (k×) → DE Embed → Latent Decode │
        │                                                         (k×)   │
        │                           ↓                        ↓           │
        │                    Watermarked Image      Secret Embedded      │
        └─────────────────────────────────────────────────────────────────┘

    Core Concept:
        Traditional difference expansion operates directly on pixel differences
        in the spatial domain. This network learns an invertible transformation
        that maps the image to a latent space (z) where:
            - Pixel dependencies are decorrelated
            - Feature distributions are more uniform
            - Difference expansion achieves higher capacity with less distortion

        The watermark is embedded in this latent space via difference expansion,
        then transformed back to the image domain. The entire pipeline is
        mathematically invertible, guaranteeing perfect reconstruction.

    The network consists of three main components:
        1. Latent Encoding Path: k invertible blocks that transform the cover
           image into a latent representation z optimized for DE embedding.
        2. Latent Difference Expansion Core: Embeds binary secret bits into
           differences between adjacent latent features using classic DE.
        3. Latent Decoding Path: k invertible blocks (weight-shared with encoders)
           that reconstruct the watermarked image from the latent representation.

    Properties:
        - Latent-Space Operation: DE operates in learned latent space for better
          embedding efficiency and visual quality.
        - Perfect Reversibility: Mathematically invertible architecture guarantees
          exact reconstruction of the original cover image.
        - Trainable Latent Transformations: Encoding/decoding blocks are learned
          end-to-end, optimizing the latent representation for DE watermarking.
        - Differentiable Rounding: Supports hard rounding (inference) and stochastic
          differentiable rounding (training) for gradient flow.
        - Configurable Depth: Parameter k controls model capacity and complexity.
        - Flexible Block Modes: Supports both ResBlock (additive coupling) and
          AffineBlock (affine coupling) for different expressiveness requirements.

    References:
        - Tian, J. (2003). Reversible data embedding using a difference expansion.
          IEEE Transactions on Circuits and Systems for Video Technology.
        - Dinh, L., et al. (2014). Nice: Non-linear independent components estimation.

    Example:
        >>> # Using ResBlock mode (additive coupling)
        >>> model = LatentDifferenceExpansion(n_channels=3, base_filters=16, k=3, block_mode="ResBlock")
        >>>
        >>> # Using AffineBlock mode (affine coupling with scaling)
        >>> model = LatentDifferenceExpansion(n_channels=3, base_filters=16, k=3, block_mode="AffineBlock")
        >>>
        >>> # Embedding: Hide secret into cover image via latent space
        >>> cover = torch.randn(4, 3, 256, 256)
        >>> secret = torch.randint(0, 2, (4, 3, 256, 256))
        >>> watermarked, _ = model(cover, secret, hard_round=False, reverse=False)
        >>>
        >>> # Extraction: Recover original and extract secret from latent space
        >>> recovered, extracted = model(watermarked, None, hard_round=True, reverse=True)
        >>> assert torch.allclose(recovered, cover, atol=1e-5)
        >>> assert torch.all(extracted == secret)
    """

    def __init__(
            self,
            n_channels: int = 3,
            base_filters: int = 4,
            use_cbam: bool = True,
            clamp: float = 4.0,
            scale: float = 1.0,
            bpp: int = 3,
            k: int = 3,
            block_mode: str = "ResBlock"
    ):
        """
        Initialize the Latent Difference Expansion network.

        The architecture uses symmetric encoding/decoding paths with weight sharing
        to ensure perfect invertibility. Decoding blocks are initialized with the
        same weights as their corresponding encoding blocks, providing an identity
        mapping at the start of training.

        Args:
            n_channels (int): Number of input/output image channels.
                Typically 3 for RGB images, 1 for grayscale.

            base_filters (int): Base number of convolutional filters in each block.
                Larger values increase model capacity and computational cost.

            use_cbam (bool): Enable Convolutional Block Attention Module in each
                invertible block for channel and spatial attention.

            clamp (float): Clamping value for sigmoid activation in affine coupling
                layers. Controls the range of scaling transformations. Only used
                when block_mode is "AffineBlock".

            scale (float): Quantization scale factor for stochastic rounding.
                Higher values reduce quantization noise.

            bpp (int): Bits per pixel - controls embedding capacity in latent space.
                Higher values allow more secret data but may increase distortion.

            k (int): Number of invertible blocks in encoding and decoding paths.
                More blocks provide stronger transformations but increase complexity.

            block_mode (str): Type of invertible block to use.
                - "ResBlock": Additive coupling (y = x + f(x')), simpler and stable.
                - "AffineBlock": Affine coupling (y = x * exp(s) + t), more expressive
                  but requires careful clamping for stability.
        """
        super().__init__()
        self.n_channels = n_channels
        self.base_filters = base_filters
        self.use_cbam = use_cbam
        self.clamp = clamp
        self.scale = scale
        self.bpp = bpp
        self.k = k
        self.block_mode = block_mode

        # Validate block mode
        assert self.block_mode in ["ResBlock", "AffineBlock"], \
            f"block_mode must be 'ResBlock' or 'AffineBlock', got {self.block_mode}"

        # ========== Latent Encoding Path ==========
        self.latent_encoders = nn.ModuleList()
        for _ in range(k):
            if self.block_mode == "ResBlock":
                encoder = ResInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, scale
                )
            else:  # AffineBlock
                encoder = AffineInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, clamp, scale
                )
            self.latent_encoders.append(encoder)

        # ========== Latent Difference Expansion Core ==========
        self.latent_difference_expansion = DifferenceExpansion(bpp)

        # ========== Latent Decoding Path ==========
        self.latent_decoders = nn.ModuleList()
        for i in range(k):
            if self.block_mode == "ResBlock":
                decoder = ResInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, scale
                )
            else:  # AffineBlock
                decoder = AffineInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, clamp, scale
                )
            # Initialize decoder with encoder weights for identity mapping
            if self.block_mode == "ResBlock":
                decoder.load_state_dict(self.latent_encoders[i].state_dict())
            self.latent_decoders.append(decoder)

    def forward(
            self,
            x: torch.Tensor,
            secret: Optional[torch.Tensor],
            hard_round: bool,
            reverse: bool = False
    ) -> tuple[Any, Any, Any]:
        """
        Execute forward pass for watermark embedding or extraction.

        The network operates in two modes controlled by the `reverse` flag.

        Embedding Mode (reverse=False):
            Image → Latent Encoding (k blocks) → DE Embed → Latent Decoding (k blocks) → Watermarked Image

        Extraction Mode (reverse=True):
            Watermarked Image → Latent Decoding (k blocks) → DE Extract → Latent Encoding (k blocks) → Recovered Image + Secret

        Important Note:
            The forward/reverse direction for AffineBlock is opposite to ResBlock
            due to their mathematical formulations. ResBlock uses additive coupling
            with symmetric forward/reverse, while AffineBlock uses affine coupling
            requiring careful direction handling.

        Args:
            x (torch.Tensor): Input image tensor of shape [B, C, H, W].
                Values should be in range [0, 255] for integer pixel values,
                or normalized [0, 1] depending on preprocessing.

            secret (Optional[torch.Tensor]): Watermark bits to embed [B, C, H, W].
                Values should be 0 or 1. Required when reverse=False.

            hard_round (bool): Rounding mode selector.
                - True: Deterministic rounding (inference)
                - False: Stochastic differentiable rounding (training)

            reverse (bool): Operation mode selector.
                - False: Watermark embedding mode
                - True: Watermark extraction mode

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - For embedding: (watermarked_image, None)
                - For extraction: (recovered_image, extracted_secret)

        Raises:
            AssertionError: If secret is None when reverse=False.
        """
        if not reverse:
            # ==================== WATERMARK EMBEDDING MODE ====================
            # Stage 1: Transform image to latent space
            x = x - 127
            for encoder in self.latent_encoders:
                x = encoder(x, hard_round, reverse=False)

            z = x
            # Stage 2: Embed secret bits in latent space using difference expansion
            x = self.latent_difference_expansion(z, secret, reverse=False)

            # Stage 3: Transform back from latent space to image domain
            if self.block_mode == "ResBlock":
                for decoder in reversed(self.latent_decoders):
                    x = decoder(x, hard_round, reverse=True)
            else:  # AffineBlock
                for decoder in reversed(self.latent_decoders):
                    x = decoder(x, hard_round, reverse=False)
            s = x + 127
            return s, secret, z
        else:
            # ==================== WATERMARK EXTRACTION MODE ====================
            # Stage 1: Transform watermarked image to latent space
            x = x - 127
            if self.block_mode == "ResBlock":
                for decoder in self.latent_decoders:
                    x = decoder(x, hard_round, reverse=False)
            else:
                for decoder in self.latent_decoders:
                    x = decoder(x, hard_round, reverse=True)
            # Stage 2: Extract secret bits and recover pre-embedding latent representation
            z, rec_secret = self.latent_difference_expansion(x, secret=None, reverse=True)
            x = z
            # Stage 3: Transform back to original image domain
            for encoder in reversed(self.latent_encoders):
                x = encoder(x, hard_round, reverse=True)
            x = x + 127
            return x, rec_secret, z

    def save(self, args, save_path: str, epoch: int, global_epoch: int) -> None:
        """
        Save model checkpoint with complete training state.

        Args:
            args: Training arguments containing hyperparameters.
            save_path (str): Directory path where checkpoint will be saved.
            epoch (int): Current epoch number (used in filename).
            global_epoch (int): Global training epoch (for multi-stage training).
        """
        os.makedirs(save_path, exist_ok=True)

        save_dict = {
            # Model architecture parameters (required for model reconstruction)
            'n_channels': self.n_channels,
            'base_filters': self.base_filters,
            'use_cbam': self.use_cbam,
            'clamp': self.clamp,
            'scale': self.scale,
            'bpp': self.bpp,
            'k': self.k,
            'block_mode': self.block_mode,

            # Training state
            'epoch': epoch,
            'global_epoch': global_epoch,
            'lambda_penalty': args.lambda_penalty,
            'penalty_start_epoch': args.penalty_start_epoch,
            'interval_epoch': args.interval_epoch,
            'increase_penalty': args.increase_penalty,
            'model_state_dict': self.state_dict(),
        }

        filepath = os.path.join(save_path, f"model_{epoch}.pth")
        torch.save(save_dict, filepath)
        print(f"✅ Model saved: {filepath}")
        cleanup_old_checkpoints(save_path)

    def load(self, filepath: str):
        """
        Load model checkpoint from file.

        Args:
            filepath (str): Path to the checkpoint file (.pth).

        Returns:
            tuple: (self, save_dict) where save_dict contains all checkpoint data
                   including model parameters and training state.

        Raises:
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        save_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict=save_dict['model_state_dict'], strict=False)
        return self, save_dict


class LatentHistogramShifting(nn.Module):
    """
    Deep Histogram Shifting Network for Reversible Data Hiding (RDH).

    This model extends the classic histogram shifting (HS) technique by embedding
    it within a learned invertible neural network. Unlike traditional pixel-domain HS,
    this method operates in a learned latent space where the histogram distribution
    can be optimized for higher embedding capacity and better visual quality.

    Architecture Overview:
        ┌──────────────────────────────────────────────────────────────────────┐
        │                    Latent Histogram Shifting                         │
        ├──────────────────────────────────────────────────────────────────────┤
        │  Cover Image → Latent Encoding (k×) → HS Embed → Latent Decode (k×) │
        │                               ↓                        ↓            │
        │                          Secret Bits            Peak Positions      │
        └──────────────────────────────────────────────────────────────────────┘

    Key Innovations:
        1. Learned Latent Space: The invertible blocks learn an optimal representation
           where the histogram has concentrated peaks, maximizing embedding capacity.
        2. End-to-End Training: All components are jointly optimized to balance
           capacity, visual quality, and reversibility.
        3. Flexible Block Architecture: Supports both ResBlock (additive coupling)
           and AffineBlock (affine coupling) for different expressiveness needs.

    Theoretical Background:
        Histogram shifting embeds data by modifying pixel values at specific
        "peak" positions in the image histogram. The classic algorithm:
            1. Identify the most frequent pixel value(s) in the histogram (peaks)
            2. Shift all pixel values above the peak by +1 to create empty space
            3. Embed bits by incrementing peak pixels (for 1) or leaving unchanged (for 0)
        This network learns to create a latent representation where histogram shifting
        achieves higher capacity with less visual distortion.

    References:
        [1] Z. Ni, Y.-Q. Shi, N. Ansari, and W. Su, "Reversible data hiding,"
            in IEEE Transactions on Circuits and Systems for Video Technology,
            vol. 16, no. 3, pp. 354-362, Mar. 2006.
        [2] L. Dinh, D. Krueger, and Y. Bengio, "NICE: Non-linear Independent
            Components Estimation," in International Conference on Learning
            Representations (ICLR), 2015.

    Example:
        >>> # Using ResBlock mode (additive coupling)
        >>> model = LatentHistogramShifting(n_channels=3, base_filters=16, k=3, block_mode="ResBlock")
        >>>
        >>> # Using AffineBlock mode (affine coupling with scaling)
        >>> model = LatentHistogramShifting(n_channels=3, base_filters=16, k=3, block_mode="AffineBlock")
        >>>
        >>> # Embedding: Hide secret into cover image via latent space
        >>> cover = torch.randn(4, 3, 256, 256)
        >>> secret = torch.randint(0, 2, (4, 3, 256, 256))
        >>> stego, embedded, z, peaks, counts, _ = model(cover, secret=secret, hard_round=False, reverse=False)
        >>>
        >>> # Extraction: Recover original and extract secret using saved peaks
        >>> recovered, extracted, _, _, _, _ = model(stego, hard_round=True, reverse=True, peaks=peaks)
        >>> assert torch.allclose(recovered, cover, atol=1e-5)
        >>> assert torch.all(extracted == embedded)
    """

    def __init__(
            self,
            n_channels: int = 3,
            base_filters: int = 4,
            use_cbam: bool = True,
            clamp: float = 4.0,
            scale: float = 1.0,
            k: int = 3,
            block_mode: str = "ResBlock",
            mode: str = "train"
    ):
        """
        Initialize the Latent Histogram Shifting network.

        The architecture uses symmetric latent encoding and decoding paths with
        weight sharing to ensure perfect invertibility. Decoding blocks are
        initialized with the same weights as their corresponding encoding blocks,
        providing an identity mapping at the start of training.

        Args:
            n_channels: Number of input/output image channels (3 for RGB, 1 for grayscale).
            base_filters: Base number of convolutional filters in each block.
            use_cbam: Enable Convolutional Block Attention Module for channel and spatial attention.
            clamp: Clamping value for sigmoid activation in affine coupling layers.
            scale: Quantization scale factor for stochastic rounding.
            k: Number of invertible blocks in latent encoding and decoding paths.
            block_mode: Type of invertible block - "ResBlock" (additive) or "AffineBlock" (affine).
            mode: Operation mode - "train" for differentiable training, "inference" for deterministic.
        """
        super().__init__()
        self.n_channels = n_channels
        self.base_filters = base_filters
        self.use_cbam = use_cbam
        self.clamp = clamp
        self.scale = scale
        self.k = k
        self.block_mode = block_mode
        assert mode in ["train", "inference"]
        self.mode = mode

        assert self.block_mode in ["ResBlock", "AffineBlock"], \
            f"block_mode must be 'ResBlock' or 'AffineBlock', got {self.block_mode}"

        # ========== Latent Encoding Path ==========
        self.latent_encoders = nn.ModuleList()
        for _ in range(k):
            if self.block_mode == "ResBlock":
                encoder = ResInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, scale
                )
            else:
                encoder = AffineInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, clamp, scale
                )
            self.latent_encoders.append(encoder)

        # ========== Core Histogram Shifting Module ==========
        self.histogram_shifting = HistogramShifting(mode=self.mode)

        # ========== Latent Decoding Path ==========
        self.latent_decoders = nn.ModuleList()
        for i in range(k):
            if self.block_mode == "ResBlock":
                decoder = ResInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, scale
                )
            else:
                decoder = AffineInvertibleBlock(
                    n_channels, n_channels, base_filters, use_cbam, clamp, scale
                )
            if self.block_mode == "ResBlock":
                decoder.load_state_dict(self.latent_encoders[i].state_dict())
            self.latent_decoders.append(decoder)

    def forward(
            self,
            x: torch.Tensor,
            secret: Optional[torch.Tensor] = None,
            hard_round: bool = True,
            reverse: bool = False,
            peaks: Optional[torch.Tensor] = None,
            last_pos: Tuple[int, int, int] = None
    ):
        """
        Execute forward pass for histogram-based watermark embedding or extraction.

        The network operates in two modes:
            - Embedding (reverse=False): Computes peaks from latent representation,
              embeds secret bits, and returns stego image with peak information.
            - Extraction (reverse=True): Uses provided peaks to extract secret bits
              and recover the original cover image.

        Args:
            x: Input image tensor of shape [B, C, H, W].
                - Embedding: Cover image
                - Extraction: Stego image
            secret: Binary secret bits for embedding mode. Shape [B, C, H, W].
            hard_round: True for deterministic rounding, False for stochastic rounding.
            reverse: False for embedding, True for extraction.
            peaks: Peak values [B, 2] required for extraction mode.
            last_pos: Last embedding position (c, i, j) for inference mode extraction.

        Returns:
            Embedding mode (reverse=False):
                - stego_image: Watermarked image
                - embedded_secret: 1D tensor of actually embedded bits
                - z: Latent representation after histogram shifting
                - peaks: Computed peak values [B, 2]
                - topk_counts: Frequency counts of peaks [B, 2]
                - last_pos: Last embedding position (for inference mode)

            Extraction mode (reverse=True):
                - recovered_image: Original cover image
                - extracted_secret: 1D tensor of extracted bits
                - z: Intermediate latent representation
                - None, None, None: Placeholders for interface consistency
        """
        if not reverse:
            # ==================== WATERMARK EMBEDDING MODE ====================
            x = x - 127
            for encoder in self.latent_encoders:
                x = encoder(x, hard_round, reverse=False)

            z = x

            if self.mode == "inference":
                hat_z, embed_secret, peaks, topk_counts, last_pos = self.histogram_shifting(
                    z, secret=secret, reverse=False
                )
            else:
                hat_z, embed_secret, peaks, topk_counts = self.histogram_shifting(
                    z, secret=secret, reverse=False
                )
                last_pos = None

            if self.block_mode == "ResBlock":
                for decoder in reversed(self.latent_decoders):
                    hat_z = decoder(hat_z, hard_round, reverse=True)
            else:
                for decoder in reversed(self.latent_decoders):
                    hat_z = decoder(hat_z, hard_round, reverse=False)

            s = hat_z + 127
            return s, embed_secret, z, peaks, topk_counts, last_pos

        else:
            # ==================== WATERMARK EXTRACTION MODE ====================
            x = x - 127

            if self.block_mode == "ResBlock":
                for decoder in self.latent_decoders:
                    x = decoder(x, hard_round, reverse=False)
            else:
                for decoder in self.latent_decoders:
                    x = decoder(x, hard_round, reverse=True)

            hat_z = x

            z, rec_secret = self.histogram_shifting(
                hat_z, reverse=True, peaks=peaks, last_pos=last_pos
            )

            for encoder in reversed(self.latent_encoders):
                z = encoder(z, hard_round, reverse=True)

            x = z + 127
            return x, rec_secret, z, None, None, None

    def save(self, args, save_path: str, epoch: int, global_epoch: int) -> None:
        """
        Save model checkpoint with complete training state.

        Args:
            args: Training arguments containing hyperparameters.
            save_path: Directory path where checkpoint will be saved.
            epoch: Current epoch number (used in filename).
            global_epoch: Global training epoch for multi-stage training.
        """
        os.makedirs(save_path, exist_ok=True)

        save_dict = {
            'n_channels': self.n_channels,
            'base_filters': self.base_filters,
            'use_cbam': self.use_cbam,
            'clamp': self.clamp,
            'scale': self.scale,
            'k': self.k,
            'block_mode': self.block_mode,
            'epoch': epoch,
            'global_epoch': global_epoch,
            'lambda_penalty': args.lambda_penalty,
            'penalty_start_epoch': args.penalty_start_epoch,
            'interval_epoch': args.interval_epoch,
            'increase_penalty': args.increase_penalty,
            'model_state_dict': self.state_dict(),
        }

        filepath = os.path.join(save_path, f"model_{epoch}.pth")
        torch.save(save_dict, filepath)
        print(f"✅ Model saved: {filepath}")
        cleanup_old_checkpoints(save_path)

    def load(self, filepath: str):
        """
        Load model checkpoint from file.

        Args:
            filepath: Path to the checkpoint file (.pth).

        Returns:
            tuple: (self, save_dict) where save_dict contains checkpoint data.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        save_dict = torch.load(filepath, map_location="cpu", weights_only=False)
        self.load_state_dict(state_dict=save_dict['model_state_dict'], strict=False)
        return self, save_dict


if __name__ == "__main__":
    # from watermarklab.steganography import iSteganoGAN
    # model = iSteganoGAN()
    # total_params = sum(p.numel() for p in model.parameters())
    torch.set_default_tensor_type(torch.DoubleTensor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = LatentDifferenceExpansion(n_channels=3, base_filters=4, use_cbam=True, scale=1, k=4,
                                      block_mode="AffineBlock")
    model.to(device)

    input_tensor = torch.randint(0, 256, (1, 3, 256, 256)).to(device) / 1.
    secret_tensor = torch.randint(0, 2, (1, 3, 256, 256)).to(device) / 1.
    output, _, _ = model(input_tensor, secret_tensor, True, False)
    output_255 = torch.round(output)
    rec_input_tensor, rec_secret_tensor, z = model(output_255, None, True, True)
    print(torch.sum(torch.abs(rec_input_tensor - input_tensor)))
    print(torch.sum(torch.abs(rec_secret_tensor - secret_tensor)))
    print(torch.sum(rec_input_tensor > 255) + torch.sum(rec_input_tensor < 0))
    print(f"Input Shape：{input_tensor.shape}")
    print(f"Output Shape：{output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params：{total_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentHistogramShifting(n_channels=3, base_filters=4, use_cbam=True, scale=1, k=3, block_mode="AffineBlock")
    model.to(device)

    input_tensor = torch.randint(0, 256, (1, 3, 256, 256)).to(device) / 1.
    secret_tensor = torch.randint(0, 2, (1, 3, 256, 256)).to(device) / 1.
    stego, embed_secret, z, peaks, topk_counts = model(input_tensor, secret_tensor, hard_round=True, reverse=False)
    print(f"Peaks: {peaks}")
    recovered, extracted, z, _, _ = model(stego, hard_round=True, reverse=True, peaks=peaks)
    # Debug info
    print(f"Secret-Extracted diff: {torch.sum(torch.abs(extracted - embed_secret)).item()}")
    print(f"Image recovery: {torch.sum(torch.abs(recovered - input_tensor)).item()}")

    # Full verification
    print(f"Perfect secret recovery: {torch.all(extracted == embed_secret)}")
    print(f"Perfect image recovery: {torch.all(recovered == input_tensor)}")
