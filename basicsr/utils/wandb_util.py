import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
import io
from PIL import Image

def tensor_to_numpy(tensor):
    """Convert a tensor to numpy array for visualization.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (H, W)

    Returns:
        np.array: RGB array of shape (H, W, 3) ready for visualization
    """
    tensor = tensor.detach().cpu()

    # Handle (C,H,W) or (H,W)
    if tensor.ndim == 3:  # C×H×W
        # Handle different channel numbers
        if tensor.shape[0] == 1:  # Grayscale
            arr = tensor[0].float()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            # Convert to RGB by repeating the channel
            rgb = arr.unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0).numpy()
        else:  # RGB or more channels
            # Take first 3 channels and treat as RGB
            rgb = tensor[:3]
            rgb = rgb.permute(1, 2, 0).float()
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = rgb.numpy()
        rgb = (rgb * 255).astype('uint8')
        return rgb

    elif tensor.ndim == 2:  # H×W (grayscale)
        arr = tensor.float()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        # Convert to RGB by repeating
        rgb = arr.unsqueeze(2).repeat(1, 1, 3).numpy()
        return (rgb * 255).astype('uint8')
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def visualize_depth(tensor, cmap='Spectral_r'):
    """Return a uint8 RGB visualisation of *tensor*.

    Works for:
        • single-channel tensors (H×W) – colour-mapped using *cmap*
        • multi-channel tensors (C×H×W) – the first 3 bands are interpreted as
          RGB and min–max-scaled independently.
    """
    tensor = tensor.detach().cpu()

    # Handle (C,H,W) or (H,W)
    if tensor.ndim == 3:  # C×H×W
        # Select first three channels for display (assume BGR → show RGB)
        rgb = tensor[:3]
        rgb = rgb.permute(1, 2, 0).float()
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        rgb = (rgb * 255).byte().numpy()
        return rgb

    elif tensor.ndim == 2:  # H×W
        arr = tensor.float()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        colormap = cm.get_cmap(cmap)
        color = colormap(arr.numpy())[:, :, :3]
        return (color * 255).astype('uint8')

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def create_comparison_image(lr_vis, hr_gt_vis, hr_pred_vis, error_map=None, cmap="inferno"):
    """Create a comparison mosaic of LR, HR-GT, HR-pred and optional error map."""

    ncols = 4 if error_map is not None else 3
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    if ncols == 1:
        axs = [axs]

    axs[0].imshow(lr_vis)
    axs[0].set_title("LR Input")
    axs[0].axis("off")

    axs[1].imshow(hr_gt_vis)
    axs[1].set_title("HR Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(hr_pred_vis)
    axs[2].set_title("SR Prediction")
    axs[2].axis("off")

    if error_map is not None:
        err_np = error_map.detach().cpu().numpy()
        im = axs[3].imshow(err_np, cmap=cmap)
        axs[3].set_title("|GT – Pred|")
        axs[3].axis("off")
        plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    buf.close()

    return img_array


def log_images_to_wandb(wandb, visuals, img_name, current_iter, dataset_name,
                       log_frequency=10, max_images=8):
    """Log LR/HR images and error map to wandb.

    Args:
        wandb: wandb module (should be imported where this is called)
        visuals (dict): Dictionary containing 'lq', 'result', and optionally 'gt' tensors
        img_name (str): Name of the current image
        current_iter (int): Current iteration number
        dataset_name (str): Name of the validation dataset
        log_frequency (int): Log every N images (default: 10)
        max_images (int): Maximum number of images to log per validation (default: 8)
    """
    # Only log a subset of images to avoid overwhelming wandb
    if not hasattr(log_images_to_wandb, 'image_count'):
        log_images_to_wandb.image_count = 0
        log_images_to_wandb.logged_this_val = 0

    # Reset counter for new validation run (simple heuristic)
    if log_images_to_wandb.image_count == 0:
        log_images_to_wandb.logged_this_val = 0

    log_images_to_wandb.image_count += 1

    # Only log every N-th image and limit total images per validation
    if (log_images_to_wandb.image_count % log_frequency != 0 or
        log_images_to_wandb.logged_this_val >= max_images):
        return

    log_images_to_wandb.logged_this_val += 1

    try:
        # Extract tensors (shape: B, C, H, W - take first batch item)
        lr_tensor = visuals['lq'][0]  # (C, H, W)
        sr_tensor = visuals['result'][0]  # (C, H, W)

        # Convert tensors to numpy arrays for visualization
        lr_vis = tensor_to_numpy(lr_tensor)
        sr_vis = tensor_to_numpy(sr_tensor)

        phase = f"val_{dataset_name}"

        # Create wandb Image objects and log
        wandb_images = {
            f"{phase}/LR": wandb.Image(lr_vis, caption=f"LR: {img_name}"),
            f"{phase}/SR": wandb.Image(sr_vis, caption=f"SR: {img_name}"),
        }

        # Add ground truth and error map if available
        if 'gt' in visuals:
            gt_tensor = visuals['gt'][0]  # (C, H, W)
            gt_vis = tensor_to_numpy(gt_tensor)
            wandb_images[f"{phase}/GT"] = wandb.Image(gt_vis, caption=f"GT: {img_name}")

            # Calculate error map (per-pixel L1 difference across channels → mean over channels)
            diff = torch.abs(sr_tensor - gt_tensor)
            error_map = diff.mean(dim=0)  # Mean over channels to get single-channel error

            # Visualize error map
            error_vis = visualize_depth(error_map, cmap='hot')
            wandb_images[f"{phase}/Error"] = wandb.Image(error_vis, caption=f"Error: {img_name}")

            # Create comparison image
            comparison_img = create_comparison_image(lr_vis, gt_vis, sr_vis, error_map, cmap="hot")
            wandb_images[f"{phase}/Comparison"] = wandb.Image(comparison_img, caption=f"Comparison: {img_name}")
        else:
            # Create comparison without GT
            comparison_img = create_comparison_image(lr_vis, sr_vis, sr_vis)  # Use SR as GT placeholder
            wandb_images[f"{phase}/Comparison"] = wandb.Image(comparison_img, caption=f"LR vs SR: {img_name}")

        # Log to wandb
        wandb.log(wandb_images, step=current_iter)

    except Exception as e:
        # Don't let wandb logging break the validation
        print(f"Warning: Failed to log images to wandb: {e}")


def log_training_images_to_wandb(wandb, visuals, current_iter, log_frequency=100, max_images_per_log=2):
    """Log training images to wandb.

    Args:
        wandb: wandb module (should be imported where this is called)
        visuals (dict): Dictionary containing 'lq', 'result', and 'gt' tensors
        current_iter (int): Current iteration number
        log_frequency (int): Log every N iterations (default: 100)
        max_images_per_log (int): Maximum number of images to log per training log (default: 2)
    """
    # Only log at specified frequency
    if current_iter % log_frequency != 0:
        return

    try:
        # Extract tensors from batch (take first batch item, but log multiple if batch size allows)
        batch_size = visuals['lq'].shape[0]
        num_images = min(batch_size, max_images_per_log)

        wandb_images = {}

        for i in range(num_images):
            lr_tensor = visuals['lq'][i]  # (C, H, W)
            sr_tensor = visuals['result'][i]  # (C, H, W)
            gt_tensor = visuals['gt'][i]  # (C, H, W)

            # Convert tensors to numpy arrays for visualization
            lr_vis = tensor_to_numpy(lr_tensor)
            sr_vis = tensor_to_numpy(sr_tensor)
            gt_vis = tensor_to_numpy(gt_tensor)

            # Create wandb Image objects and log
            wandb_images[f"train/LR_{i}"] = wandb.Image(lr_vis, caption=f"LR Training Sample {i}")
            wandb_images[f"train/SR_{i}"] = wandb.Image(sr_vis, caption=f"SR Training Sample {i}")
            wandb_images[f"train/GT_{i}"] = wandb.Image(gt_vis, caption=f"GT Training Sample {i}")

            # Calculate error map (per-pixel L1 difference across channels → mean over channels)
            diff = torch.abs(sr_tensor - gt_tensor)
            error_map = diff.mean(dim=0)  # Mean over channels to get single-channel error

            # Visualize error map
            error_vis = visualize_depth(error_map, cmap='hot')
            wandb_images[f"train/Error_{i}"] = wandb.Image(error_vis, caption=f"Training Error {i}")

            # Create comparison image
            comparison_img = create_comparison_image(lr_vis, gt_vis, sr_vis, error_map, cmap="hot")
            wandb_images[f"train/Comparison_{i}"] = wandb.Image(comparison_img, caption=f"Training Comparison {i}")

        # Log to wandb
        wandb.log(wandb_images, step=current_iter)

    except Exception as e:
        # Don't let wandb logging break the training
        print(f"Warning: Failed to log training images to wandb: {e}")


def reset_wandb_image_counter():
    """Reset the image counter for wandb logging."""
    if hasattr(log_images_to_wandb, 'image_count'):
        log_images_to_wandb.image_count = 0
        log_images_to_wandb.logged_this_val = 0
