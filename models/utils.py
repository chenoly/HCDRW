# Copyright (c) 2025 chenoly@outlook.com. Licensed under MIT.
import glob
import os
import re
import math
import torch
import kornia
import hashlib
import numpy as np
import torch.nn as nn
from torch import Tensor


class GaussianPrior(nn.Module):
    def __init__(self, target_mean=0., target_std=1., reduction='mean'):
        super().__init__()
        self.target_mean = torch.tensor(target_mean, dtype=torch.float32)
        self.target_std = torch.tensor(target_std, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, z):
        target_mean = self.target_mean.to(z.device)
        target_std = self.target_std.to(z.device)
        # log p(z) = -0.5 * log(2π) - log(std) - 0.5 * ((z - μ)/std)^2
        log_prob = -0.5 * math.log(2 * math.pi) - torch.log(target_std) - 0.5 * ((z - target_mean) / target_std) ** 2

        log_prob = log_prob.sum(dim=-1)
        nll = -log_prob

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class LaplacePrior(nn.Module):
    def __init__(self, target_loc=0, target_scale=1, reduction='mean'):
        super().__init__()
        self.target_loc = torch.tensor(target_loc, dtype=torch.float32)
        self.target_scale = torch.tensor(target_scale, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, z):
        target_loc = self.target_loc.to(z.device)
        target_scale = self.target_scale.to(z.device)
        # log p(z) = -log(2 * scale) - |z - loc| / scale
        log_prob = -torch.log(2 * target_scale) - torch.abs(z - target_loc) / target_scale
        log_prob = log_prob.sum(dim=-1)
        nll = -log_prob
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class StochasticRound(nn.Module):
    def __init__(self, scale=1.0, round_mode: str = "round"):
        super().__init__()
        self.scale = scale
        self.round_mode = round_mode
        assert self.round_mode in ["round", "floor", "ceil"]

    def forward(self, x, hard_round: bool = True):
        scale_x = x / self.scale

        if self.round_mode == "floor":
            round_out = scale_x + (torch.floor(scale_x) - scale_x).detach()
            if not hard_round:
                round_out = round_out + torch.rand_like(x)
        elif self.round_mode == "round":
            round_out = scale_x + (torch.round(scale_x) - scale_x).detach()
            if not hard_round:
                round_out = round_out + torch.rand_like(x) - 0.5
        else:  # ceil
            round_out = scale_x + (torch.ceil(scale_x) - scale_x).detach()
            if not hard_round:
                round_out = round_out - torch.rand_like(x)
        return round_out * self.scale


class PenaltyLoss(nn.Module):
    def __init__(self, max_value=1., min_value=0., penalty_loss_type='mae'):
        """
        Initializes the Penalty Loss for overflow pixels.

        Parameters:
            max_value (float): Maximum allowable pixel value (default is 1).
            min_value (float): Minimum allowable pixel value (default is 0).
            penalty_loss_type (str): Type of loss to use ('mae' or 'mse'). Default is 'mae'.
        """
        super().__init__()
        self.max_value = max_value
        self.min_value = min_value
        self.loss_type = penalty_loss_type.lower()

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif self.loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {penalty_loss_type}. Choose 'mae' or 'mse'")

    def __call__(self, input_tensor):
        """
        Computes the penalty loss for pixels that overflow the allowable range.

        Parameters:
            input_tensor (Tensor): Input tensor to compute penalty loss on.

        Returns:
            Tensor: The penalty loss value.
        """
        below_min = torch.relu(self.min_value - input_tensor)
        loss_min = self.loss_fn(below_min, torch.zeros_like(below_min))
        above_max = torch.relu(input_tensor - self.max_value)
        loss_max = self.loss_fn(above_max, torch.zeros_like(above_max))
        loss = loss_min + loss_max
        return loss


# Normalize function to scale image tensor to [0, 1] range
def normalize(input_image):
    """
    Normalize the input image tensor to the range [0, 1].

    Parameters:
        input_image (Tensor): Input image tensor to normalize.

    Returns:
        Tensor: The normalized image tensor.
    """
    min_vals = input_image.amin(dim=(1, 2, 3), keepdim=True)
    max_vals = input_image.amax(dim=(1, 2, 3), keepdim=True)
    normalized_img = (input_image - min_vals) / (max_vals - min_vals + 1e-5)  # Prevent division by zero
    return normalized_img


# Function to extract the accuracy of a secret image
def extract_accuracy(ext_secret, secret, max_value=1.):
    """
    Extracts the accuracy of a secret image by comparing it to the expected secret.

    Parameters:
        ext_secret (Tensor): The extracted secret image.
        secret (Tensor): The ground truth secret image.

    Returns:
        float: The accuracy value.

    Parameters
    ----------
    secret
    ext_secret
    max_value
    """
    acc = 1.0 - (torch.abs(torch.round(ext_secret.clamp(0., max_value)) - secret).mean())
    return acc.item()


# Function to calculate the number of overflow pixels in a stego image
def overflow_num(stego, mode, min_value=0., max_value=1.):
    """
    Calculate the number of overflow pixels in the stego image.

    Parameters:
        stego (Tensor): The stego image tensor.
        mode (int): The overflow mode (0 for below min_value, 255 for above max_value).
        min_value (float): The minimum allowed pixel value (default is 0).
        max_value (float): The maximum allowed pixel value (default is 1).

    Returns:
        float: The average overflow pixel count.
    """
    assert mode in [0, 255]
    if mode == 0:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) < min_value, dim=(1, 2, 3)).float().mean()
    else:
        overflow_pixel_n = torch.sum(StochasticRound()(stego, True) > max_value, dim=(1, 2, 3)).float().mean()
    return overflow_pixel_n.item()


def compute_F1_score(pre_mask, target_mask, threshold=0.5):
    """
    Simplified version for binary masks.
    """
    # Binarize masks
    pre_binary = (pre_mask > threshold).float()
    target_binary = (target_mask > threshold).float()

    # Flatten tensors
    pre_flat = pre_binary.view(-1)
    target_flat = target_binary.view(-1)

    # Calculate TP, FP, FN
    TP = (pre_flat * target_flat).sum()
    FP = (pre_flat * (1 - target_flat)).sum()
    FN = ((1 - pre_flat) * target_flat).sum()

    # Calculate Precision and Recall
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.item()


# Function to compute the PSNR between the input and target images
def compute_psnr(input_image, target_image, max_value=1.):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor.
        max_value (float): The maximum allowable pixel value (default is 1).

    Returns:
        float: The average PSNR value.
    """
    # Apply stochastic rounding and clamp images to the range [0, 1]
    average_psnr = kornia.metrics.psnr(input_image.clamp(0., max_value), target_image.clamp(0., max_value),
                                       max_value).mean()
    return average_psnr.item()


def quantize_image(input_image):
    """
    Quantize the input image using stochastic rounding.

    Parameters:
        input_image (Tensor): Input image tensor with values between 0 and 1.

    Returns:
        Tensor: The quantized image tensor after applying stochastic rounding.
    """
    # Apply stochastic rounding to the input image, ensuring the values are within the [0, 1] range.
    input_image = StochasticRound()(input_image.clamp(0., 1.), True)
    return input_image


def quantize_residual_image(input_image, target_image, max_value=1.):
    """
    Quantize the residual image, which is the difference between the input and target image.

    Parameters:
        input_image (Tensor): The input image tensor.
        target_image (Tensor): The target image tensor to compare against.

    Returns:
        Tensor: The quantized residual image after applying stochastic rounding.
        :param max_value:
    """
    # Calculate the residual (difference) between the input image and the target image.
    # Then normalize the residual and apply stochastic rounding.
    input_image = (input_image / max_value - target_image / max_value) + 0.5
    return input_image


def sha256_of_image_array(img_array):
    """
    Compute the SHA-256 hash of a NumPy image array.

    Parameters:
    img_array (np.ndarray): The input image array.

    Returns:
    str: A 64-character hexadecimal SHA-256 hash string.
    """
    # Convert the image array to a raw byte sequence
    img_bytes = img_array.tobytes()
    # Compute the SHA-256 hash of the byte sequence and return the hexadecimal representation
    sha256_hash = hashlib.sha256(img_bytes).hexdigest()
    return sha256_hash


def sha256_to_bitstream(sha256_str):
    """
    Convert a 64-character hexadecimal SHA-256 string to a 256-bit binary stream.

    Each hexadecimal character represents 4 bits (since 16 = 2^4), so 64 hex characters equal 256 bits.

    Parameters:
    sha256_str (str): A 64-character hexadecimal SHA-256 hash string.

    Returns:
    np.ndarray: A NumPy array of 256 binary values (dtype=np.uint8), where each element is 0 or 1.
    """
    # Initialize an array of 256 zeros to store the binary bits
    bits = np.zeros(256, dtype=np.uint8)
    # Iterate through each character in the SHA-256 hex string
    for i, hex_char in enumerate(sha256_str):
        # Convert the hex character to its integer value (0 to 15)
        val = int(hex_char, 16)
        # Convert this 4-bit value to binary and store it in the bitstream
        # The bits are stored in big-endian order (most significant bit first)
        for j in range(4):
            bits[i * 4 + (3 - j)] = (val >> j) & 1
    return bits.tolist()


def find_latest_model(checkpoint_path, train_name):
    pattern = re.compile(rf"model_(\d+)\.pth")
    full_path = os.path.join(checkpoint_path, train_name)
    files = os.listdir(full_path)
    latest_epoch = -1
    latest_file = None
    for file in files:
        match = pattern.match(file)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = file
    if latest_file:
        return os.path.abspath(os.path.join(full_path, latest_file))
    else:
        raise FileNotFoundError("No model files found matching the pattern.")


def cleanup_old_checkpoints(save_path: str, max_checkpoints: int = 1000):
    """
    Delete old checkpoints, keeping only the ones with the largest epoch numbers.

    Args:
        save_path: Directory containing checkpoint files
        max_checkpoints: Maximum number of checkpoints to keep
    """
    # Get all checkpoint files
    files = glob.glob(os.path.join(save_path, "model_*.pth"))
    if len(files) <= max_checkpoints:
        return

    # Extract epoch numbers from filenames
    epoch_files = []
    for f in files:
        match = re.search(r'model_(\d+)', os.path.basename(f))
        if match:
            epoch_files.append((int(match.group(1)), f))

    # Sort by epoch descending (largest first)
    epoch_files.sort(key=lambda x: x[0], reverse=True)

    # Delete files with smaller epochs (beyond max_checkpoints)
    for epoch, f in epoch_files[max_checkpoints:]:
        os.remove(f)
    print(f"Delete {len(epoch_files[max_checkpoints:])} model weight files!")
