import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import StochasticRound


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=bias)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bias=False):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7, bias=False):
        """
        Combined Channel and Spatial Attention Module.
        """
        super().__init__()
        self.channel_att = ChannelAttention(channel, reduction=reduction, bias=bias)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel, bias=bias)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNetWithCBAM(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, base_filters=4, use_cbam=True):
        super().__init__()
        self.use_cbam = use_cbam

        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = self._down_block(base_filters, base_filters * 2)
        self.down2 = self._down_block(base_filters * 2, base_filters * 4)
        self.down3 = self._down_block(base_filters * 4, base_filters * 8)
        self.down4 = DoubleConv(base_filters * 8, base_filters * 16)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.up1 = self._up_block(base_filters * 16, base_filters * 8)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.up2 = self._up_block(base_filters * 8, base_filters * 4)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.up3 = self._up_block(base_filters * 4, base_filters * 2)

        self.upconv4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.up4 = self._up_block(base_filters * 2, base_filters)

        # --- Output ---
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def _down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def _up_block(self, in_ch, out_ch):
        layers = [DoubleConv(in_ch, out_ch)]
        if self.use_cbam:
            layers.append(CBAMBlock(out_ch, reduction=2, spatial_kernel=7))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.upconv1(x5)
        x = self._crop_and_concat(x, x4)
        x = self.up1(x)

        x = self.upconv2(x)
        x = self._crop_and_concat(x, x3)
        x = self.up2(x)

        x = self.upconv3(x)
        x = self._crop_and_concat(x, x2)
        x = self.up3(x)

        x = self.upconv4(x)
        x = self._crop_and_concat(x, x1)
        x = self.up4(x)

        logits = self.outc(x)
        logits = self.relu(logits)
        return logits

    def _crop_and_concat(self, up_sampled, bypass):
        if up_sampled.size()[2:] != bypass.size()[2:]:
            up_sampled = F.interpolate(up_sampled, size=bypass.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up_sampled, bypass], dim=1)


class ResInvertibleBlock(nn.Module):
    """
    Residual Invertible Block for reversible image transformations.

    This block implements an invertible transformation based on the coupling layer
    design, where the input is split into three channels and processed through
    residual connections with stochastic rounding.

    The transformation follows this pattern:
        Forward (reverse=False):
            y1 = x1 + round(U(x2))
            y2 = x2 + round(Q(y1))
            y3 = x3 + round(H(y2))
        Reverse (reverse=True):
            y3 = x3 - round(H(x2))
            y2 = x2 - round(Q(x1))
            y1 = x1 - round(U(y2))

    Args:
        n_channels: Number of input channels (must be divisible by 3)
        n_classes: Number of output channels (must be divisible by 3)
        base_filters: Base filter count for internal U-Net modules
        use_cbam: Whether to use CBAM attention in U-Nets
        scale: Scaling factor for stochastic rounding
    """

    def __init__(self, n_channels=3, n_classes=3, base_filters=4, use_cbam=True, scale=1.):
        super().__init__()
        # Stochastic rounding module for differentiable quantization
        self.round = StochasticRound(scale)

        # Three U-Net modules for the three coupling layers
        self.ui = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.qi = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.hi = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)

    def __call__(self, x, hard_round, reverse):
        """
        Apply forward or reverse transformation.

        Args:
            x: Input tensor of shape (N, C, H, W) where C = n_channels
            hard_round: If True, use deterministic rounding; if False, use stochastic rounding
            reverse: If False, apply forward transform; if True, apply reverse transform

        Returns:
            Transformed tensor of shape (N, n_classes, H, W)
        """
        # Split input into three channel groups
        x1 = x[:, 0:1, :, :]  # First channel group (e.g., R)
        x2 = x[:, 1:2, :, :]  # Second channel group (e.g., G)
        x3 = x[:, 2:3, :, :]  # Third channel group (e.g., B)

        if not reverse:
            # Forward transformation: encode
            y1 = x1 + self.round(self.ui(x2), hard_round)
            y2 = x2 + self.round(self.qi(y1), hard_round)
            y3 = x3 + self.round(self.hi(y2), hard_round)
        else:
            # Reverse transformation: decode/reconstruct
            y3 = x3 - self.round(self.hi(x2), hard_round)
            y2 = x2 - self.round(self.qi(x1), hard_round)
            y1 = x1 - self.round(self.ui(y2), hard_round)

        # Concatenate channel groups
        y = torch.cat([y1, y2, y3], dim=1)
        return y


class AffineInvertibleBlock(nn.Module):
    """
    Invertible Block with affine coupling layers for reversible transformations.

    This block implements an advanced invertible transformation using affine coupling
    layers, which provide more expressive power than simple additive coupling.
    The input is split into three channel groups and processed through a series of
    transformations involving both scale and shift operations.

    The transformation follows this pattern:
        Forward (reverse=False):
            y1 = x1 + round(U(x2))
            y2 = x2 * round(exp(sigmoid(S(y1)))) + round(Q(y1))
            y3 = x3 * round(exp(sigmoid(K(y2)))) + round(H(y1))

        Reverse (reverse=True):
            y3 = (x3 - round(H(x1))) / round(exp(sigmoid(K(x2))))
            y2 = (x2 - round(Q(x1))) / round(exp(sigmoid(S(x1))))
            y1 = x1 - round(U(y2))

    The affine coupling (multiplication + addition) provides richer transformations
    while maintaining perfect invertibility. The sigmoid and clamp ensure stable
    scale factors.

    Args:
        n_channels: Number of input channels (must be divisible by 3)
        n_classes: Number of output channels (must be divisible by 3)
        base_filters: Base filter count for internal U-Net modules
        use_cbam: Whether to use CBAM attention in U-Nets
        clamp: Clamping value for sigmoid output to control scale factor range
        scale: Scaling factor for stochastic rounding
    """

    def __init__(self, n_channels=3, n_classes=3, base_filters=4, use_cbam=True, clamp=4., scale=1.):
        super().__init__()
        self.clamp = clamp
        self.round = StochasticRound(scale)
        self.ui = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.si = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.qi = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.ki = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)
        self.hi = UNetWithCBAM(n_channels // 3, n_classes // 3, base_filters, use_cbam)

    def sigmoid(self, s):
        """
        Apply sigmoid and scale to get bounded scale factors.

        The sigmoid ensures the scale factor is positive and bounded,
        while clamp controls the maximum value for stability.
        """
        return self.clamp * (torch.sigmoid(s))

    def exp(self, s):
        """
        Apply exponential to ensure positive scale factors.
        """
        return torch.exp(s)

    def __call__(self, x, hard_round, reverse):
        """
        Apply forward or reverse affine transformation.

        Args:
            x: Input tensor of shape (N, C, H, W) where C = n_channels
            hard_round: If True, use deterministic rounding; if False, use stochastic rounding
            reverse: If False, apply forward transform; if True, apply reverse transform

        Returns:
            Transformed tensor of shape (N, n_classes, H, W)
        """
        # Split input into three channel groups along the channel dimension
        x1 = x[:, 0:1, :, :]  # First channel group (e.g., R)
        x2 = x[:, 1:2, :, :]  # Second channel group (e.g., G)
        x3 = x[:, 2:3, :, :]  # Third channel group (e.g., B)

        if not reverse:
            y1 = x1 + self.round(self.ui(x2), hard_round)
            y2 = x2 * (self.round(self.exp(self.sigmoid(self.si(y1))), hard_round)) + self.round(self.qi(y1),
                                                                                                 hard_round)
            y3 = x3 * (self.round(self.exp(self.sigmoid(self.ki(y2))), hard_round)) + self.round(self.hi(y1),
                                                                                                 hard_round)
        else:
            y3 = (x3 - self.round(self.hi(x1), hard_round)) / (
                self.round(self.exp(self.sigmoid(self.ki(x2))), hard_round))
            y2 = (x2 - self.round(self.qi(x1), hard_round)) / (
                self.round(self.exp(self.sigmoid(self.si(x1))), hard_round))
            y1 = x1 - self.round(self.ui(y2), hard_round)
        y = torch.cat([y1, y2, y3], dim=1)
        return y


class DifferenceExpansion(nn.Module):
    def __init__(self, bpp: int = 3):
        """
        Initializes the Difference Expansion module for Reversible Data Hiding (RDH).

        Args:
            bpp (int): Bits per pixel capacity. Determines how many channel pairs
                       are used for embedding.
                       1: Embeds into (R, G)
                       2: Embeds into (R, G) and (G, B)
                       3: Embeds into (R, G), (G, B), and (R, B)
        """
        super().__init__()
        self.bpp = bpp
        self.floor = StochasticRound(round_mode="floor")

    def _de_pair(self, x1, x2, secret=None, reverse=False):
        """
        Performs Difference Expansion on a pair of tensors (e.g., two color channels).

        Args:
            x1 (Tensor): First channel tensor (e.g., Red).
            x2 (Tensor): Second channel tensor (e.g., Green).
            secret (Tensor): Secret bits to embed. Should be same shape as x1/x2.
            reverse (bool):
                False: Embedding mode (Expand difference and add secret).
                True: Extraction mode (Extract secret and recover original values).

        Returns:
            If reverse=False: Returns modified (y1, y2).
            If reverse=True: Returns recovered (x1_rec, x2_rec) and extracted_secret.
        """
        if not reverse:
            l = self.floor((x1 + x2) / 2)
            h = x1 - x2
            h_new = 2 * h + secret
            y1 = l + self.floor((h_new + 1) / 2)
            y2 = l - self.floor(h_new / 2)
            return y1, y2
        else:
            h = x1 - x2
            extracted_secret = torch.fmod(h, 2).abs()
            h_orig = self.floor(h / 2)
            l = self.floor((x1 + x2) / 2)
            x1_rec = l + self.floor((h_orig + 1) / 2)
            x2_rec = l - self.floor(h_orig / 2)
            return x1_rec, x2_rec, extracted_secret

    def forward(self, x, secret=None, reverse=False):
        """
        Forward pass for embedding or extracting secret data across RGB channels.

        Args:
            x (Tensor): Input image tensor of shape (N, 3, H, W).
            secret (Tensor): Secret data tensor of shape (N, bpp, H, W).
                             Required if reverse=False.
            reverse (bool):
                False: Embed secret into image.
                True: Extract secret from image and recover original image.

        Returns:
            If reverse=False: Returns modified image tensor (N, 3, H, W).
            If reverse=True: Returns tuple (recovered_image, extracted_secret).
        """
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]

        if not reverse:
            if secret is None:
                raise ValueError("Secret tensor must be provided for embedding mode.")
            s_rg = secret[:, 0:1, :, :] if self.bpp >= 1 else None
            s_gb = secret[:, 1:2, :, :] if self.bpp >= 2 else None
            s_rb = secret[:, 2:3, :, :] if self.bpp >= 3 else None
            if self.bpp >= 1:
                R, G = self._de_pair(R, G, s_rg, reverse=False)
            if self.bpp >= 2:
                G, B = self._de_pair(G, B, s_gb, reverse=False)
            if self.bpp >= 3:
                R, B = self._de_pair(R, B, s_rb, reverse=False)
            return torch.cat([R, G, B], dim=1)

        else:
            extracted_secrets = []
            if self.bpp >= 3:
                R, B, s_rb = self._de_pair(R, B, secret=None, reverse=True)
                extracted_secrets.append(s_rb)
            if self.bpp >= 2:
                G, B, s_gb = self._de_pair(G, B, secret=None, reverse=True)
                extracted_secrets.append(s_gb)
            if self.bpp >= 1:
                R, G, s_rg = self._de_pair(R, G, secret=None, reverse=True)
                extracted_secrets.append(s_rg)

            extracted_secrets.reverse()
            if len(extracted_secrets) > 0:
                extracted_secret = torch.cat(extracted_secrets, dim=1)
            else:
                extracted_secret = None
            x_rec = torch.cat([R, G, B], dim=1)
            return x_rec, extracted_secret


class HistogramShifting(nn.Module):
    """
    Histogram Shifting module for Reversible Data Hiding (RDH).

    This module implements the classic histogram shifting algorithm in a differentiable
    manner, enabling end-to-end training within a deep learning framework. The algorithm
    operates by identifying the two most frequent values (peaks) in the latent histogram,
    shifting surrounding values to create vacant bins, and embedding secret bits by
    modifying the peak values.

    Two operation modes are provided:
        - train: Differentiable forward pass with stochastic rounding for gradient flow
        - inference: Deterministic forward pass with for-loop implementation and position tracking

    References:
        Ni et al., "Reversible Data Hiding," IEEE TCSVT, 2006.
    """

    def __init__(self, mode: str = "training"):
        """
        Initialize the Histogram Shifting module.

        Args:
            mode: Operation mode - "train" for differentiable training,
                  "inference" for deterministic inference with position tracking.
        """
        super().__init__()
        assert mode in ["train", "inference"]
        self.mode = mode
        self.round = StochasticRound(round_mode="round")

    def _get_peaks(self, z):
        """
        Identify the two most frequent values in the histogram for each sample in the batch.

        Returns:
            peaks: Peak values (e1, e2) sorted in ascending order, shape (N, 2)
            topk_counts: Frequencies of the two peaks, shape (N, 2)
        """
        N = z.shape[0]
        peaks = torch.zeros((N, 2), dtype=z.dtype, device=z.device)
        topk_counts_final = torch.zeros((N, 2), dtype=z.dtype, device=z.device)
        for i in range(N):
            z_flat = z[i].flatten()
            unique_vals, counts = torch.unique(z_flat, return_counts=True)

            if len(unique_vals) < 2:
                val = unique_vals[0] if len(unique_vals) == 1 else 0
                peaks[i, 0] = val
                peaks[i, 1] = val + 1
                continue

            topk_counts, topk_indices = torch.topk(counts, 2)
            p1 = unique_vals[topk_indices[0]]
            p2 = unique_vals[topk_indices[1]]
            e1 = torch.min(p1, p2)
            e2 = torch.max(p1, p2)
            peaks[i, 0] = e1
            peaks[i, 1] = e2
            topk_counts_final[i] = topk_counts
        return peaks, topk_counts_final

    def forward(self, z, secret=None, reverse=False, peaks=None, last_pos=None):
        """
        Route to the appropriate mode-specific forward function.

        Returns:
            train mode: (z_int, embed_secret, peaks, topk_counts) or (recovered_image, extracted_values_only)
            inference mode: (z_int, embed_secret_list, peaks, topk_counts, last_pos) or (recovered_image, extracted_secret_list)
        """
        if self.mode == "train":
            return self.train_func(z, secret, reverse, peaks)
        else:
            return self.inference_func(z, secret, reverse, peaks, last_pos)

    def inference_func(self, z, secret: list = None, reverse=False, peaks=None, last_pos=None):
        """
        Inference mode histogram shifting using deterministic for-loop implementation.

        In embedding mode, records the last embedding position to avoid unnecessary
        distortion on subsequent pixels. In extraction mode, uses the recorded position
        to terminate processing early.

        Returns:
            Embedding: (z_int, embed_secret_list, peaks, topk_counts, last_pos)
            Extraction: (recovered_image, extracted_secret_list)
        """
        z_int = self.round(z)
        N, C, H, W = z_int.shape
        assert N == 1, "Batch size must be 1 for inference mode"

        if not reverse:
            # ==================== EMBEDDING MODE ====================
            assert secret is not None, "Secret must be provided for embedding mode."

            peaks, topk_counts = self._get_peaks(z_int)
            e1 = peaks[0, 0].item()
            e2 = peaks[0, 1].item()
            total_capacity = int(topk_counts.sum().item())

            if len(secret) >= total_capacity:
                secret_full = secret[:total_capacity]
            else:
                secret_full = secret

            z_out = z_int[0].clone()
            embed_secret_list = []
            idx = 0
            last_pos = (0, 0, 0)

            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        if idx >= len(secret_full):
                            break
                        last_pos = (c, i, j)
                        val = z_out[c, i, j].item()

                        if val < e1:
                            z_out[c, i, j] = val - 1
                        elif val > e2:
                            z_out[c, i, j] = val + 1
                        elif val == e1:
                            bit = secret_full[idx]
                            embed_secret_list.append(bit)
                            z_out[c, i, j] = val - bit
                            idx += 1
                        elif val == e2:
                            bit = secret_full[idx]
                            embed_secret_list.append(bit)
                            z_out[c, i, j] = val + bit
                            idx += 1
                    if idx >= len(secret_full):
                        break
                if idx >= len(secret_full):
                    break

            return z_out.unsqueeze(0), embed_secret_list, peaks, topk_counts, last_pos

        else:
            # ==================== EXTRACTION MODE ====================
            assert peaks is not None, "Peaks must be provided for extraction mode."
            assert last_pos is not None, "last_pos must be provided for extraction mode."

            e1 = peaks[0, 0].item()
            e2 = peaks[0, 1].item()

            z_out = z_int[0].clone()
            extracted_secret_list = []
            now_pos = (0, 0, 0)

            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        now_pos = (c, i, j)
                        val = z_out[c, i, j].item()

                        if val < e1 - 1:
                            z_out[c, i, j] = val + 1
                        elif val > e2 + 1:
                            z_out[c, i, j] = val - 1
                        elif val == e1 - 1:
                            extracted_secret_list.append(1)
                            z_out[c, i, j] = e1
                        elif val == e1:
                            extracted_secret_list.append(0)
                        elif val == e2 + 1:
                            extracted_secret_list.append(1)
                            z_out[c, i, j] = e2
                        elif val == e2:
                            extracted_secret_list.append(0)

                        if now_pos == last_pos:
                            break
                    if now_pos == last_pos:
                        break
                if now_pos == last_pos:
                    break

            return z_out.unsqueeze(0), extracted_secret_list

    def train_func(self, z, secret=None, reverse=False, peaks=None):
        """
        Training mode histogram shifting with differentiable operations.

        Uses vectorized tensor operations for efficient gradient computation.
        Supports batch processing and stochastic rounding for end-to-end training.

        Returns:
            Embedding: (z_int, embed_secret, peaks, topk_counts)
            Extraction: (recovered_image, extracted_values_only)
        """
        z_int = self.round(z)
        N, C, H, W = z_int.shape

        if not reverse:
            # ==================== EMBEDDING MODE ====================
            assert secret is not None, "Secret must be provided for embedding mode."

            peaks, topk_counts = self._get_peaks(z_int)

            e1 = peaks[:, 0].view(N, 1, 1, 1)
            e2 = peaks[:, 1].view(N, 1, 1, 1)

            mask_left = (z_int < e1).float()
            mask_right = (z_int > e2).float()
            mask_peak1 = (z_int == e1).float()
            mask_peak2 = (z_int == e2).float()

            embed_secret = secret[(mask_peak1 + mask_peak2).bool()]

            z_int = mask_left.detach() * (z_int - 1) + (1 - mask_left.detach()) * z_int
            z_int = mask_right.detach() * (z_int + 1) + (1 - mask_right.detach()) * z_int

            z_int = mask_peak1.detach() * (z_int - secret) + (1 - mask_peak1.detach()) * z_int
            z_int = mask_peak2.detach() * (z_int + secret) + (1 - mask_peak2.detach()) * z_int

            return z_int, embed_secret, peaks, topk_counts

        else:
            # ==================== EXTRACTION MODE ====================
            assert peaks is not None, "Peaks must be provided for extraction mode."

            e1 = peaks[:, 0].view(N, 1, 1, 1)
            e2 = peaks[:, 1].view(N, 1, 1, 1)

            extracted_secret = torch.zeros_like(z_int, dtype=torch.long)
            recovered_image = z_int.clone()

            mask_stego_e1_equal = z_int == e1
            mask_stego_e2_equal = z_int == e2
            mask_stego_e1_minus = z_int == (e1 - 1)
            mask_stego_e2_plus = z_int == (e2 + 1)

            mask_rec_left = recovered_image < (e1 - 1)
            mask_rec_right = recovered_image > (e2 + 1)

            extracted_secret = torch.where(mask_stego_e1_minus, torch.ones_like(extracted_secret), extracted_secret)
            extracted_secret = torch.where(mask_stego_e2_plus, torch.ones_like(extracted_secret), extracted_secret)

            recovered_image = torch.where(mask_stego_e1_minus, e1, recovered_image)
            recovered_image = torch.where(mask_stego_e2_plus, e2, recovered_image)

            recovered_image = torch.where(mask_rec_left, recovered_image + 1, recovered_image)
            recovered_image = torch.where(mask_rec_right, recovered_image - 1, recovered_image)

            embed_positions_mask = (mask_stego_e1_minus | mask_stego_e2_plus |
                                    mask_stego_e1_equal | mask_stego_e2_equal)
            extracted_values_only = extracted_secret[embed_positions_mask]

            return recovered_image, extracted_values_only


if __name__ == "__main__":
    # hs = HistogramShifting(mode="train")
    #
    # # Test with smaller image for easier debugging
    # img = torch.zeros(size=(1, 3, 64, 64))
    # img[:, :, 8:54, 8:54] = 1
    # img[:, :, 8:54, 36:54] = 2
    # secret = torch.randint(0, 2, img.shape).float()
    # # secret = [random.randint(0, 1) for i in range(10000)]
    # # Embedding
    # stego, embed_secret, peaks, topk_counts = hs(img, secret=secret, reverse=False)
    # print(f"Peaks: {peaks}")
    #
    # # Extraction
    # recovered, extracted = hs(stego, reverse=True, peaks=peaks)
    #
    # # Debug info
    # print(f"Secret-Extracted diff: {torch.sum(torch.abs(extracted - embed_secret)).item()}")
    # print(f"Image recovery: {torch.sum(torch.abs(recovered - img.long())).item()}")
    #
    # # Full verification
    # print(f"Perfect secret recovery: {torch.all(extracted == embed_secret.long())}")
    # print(f"Perfect image recovery: {torch.all(recovered == img.long())}")

    import random

    hs = HistogramShifting(mode="inference")

    img = torch.zeros(size=(1, 3, 64, 64))
    img[:, :, 8:54, 8:54] = 1
    img[:, :, 8:54, 36:54] = 2

    secret_len = 10000
    secret_list = [random.randint(0, 1) for _ in range(secret_len)]

    print(f"Image shape: {img.shape}")
    print(f"Secret length: {len(secret_list)}")

    # Embedding
    stego, embed_secret_list, peaks, topk_counts, last_pos = hs.inference_func(
        img, secret=secret_list, reverse=False
    )
    print(f"Peaks: {peaks}")
    print(f"Embedded secret length: {len(embed_secret_list)}")
    print(f"Topk counts: {topk_counts}")
    print(f"Last position: {last_pos}")

    # Extraction
    recovered, extracted_list = hs.inference_func(
        stego, reverse=True, peaks=peaks, last_pos=last_pos
    )
    print(f"Extracted secret length: {len(extracted_list)}")

    print(f"\n--- Verification ---")
    print(f"Secret match: {extracted_list == embed_secret_list}")
    print(f"Image recovery: {torch.all(recovered == img.long()).item()}")
