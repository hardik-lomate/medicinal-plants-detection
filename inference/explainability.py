"""
inference/explainability.py
===========================
Grad-CAM (Gradient-weighted Class Activation Mapping) Visualizer.

Generates visual attention heatmaps directly from the existing trained EfficientNet-B0
models without any fine-tuning or retraining.
Visualizes which anatomical regions of the leaf/plant (e.g. central vein, margins,
serration, apex, or background) influenced the neural classification decision.
"""

import base64
from io import BytesIO
from typing import Tuple, Optional, Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from inference.preprocessing import preprocess_for_inference, RESIZE_EDGE, IMAGE_SIZE


def apply_colormap_jet(heatmap: np.ndarray) -> np.ndarray:
    """
    Applies a standard Jet colormap to a 2D float array in [0, 1] using pure NumPy.
    Returns RGB uint8 image array of shape (H, W, 3).
    """
    val = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * val - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * val - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * val - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural networks (EfficientNet-B0).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        self.activations = []
        self.gradients = []
        self._handles = []

    def _register_hooks(self):
        self.activations.clear()
        self.gradients.clear()

        def forward_hook(module, input, output):
            self.activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles = [h1, h2]

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(
        self,
        tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes 2D heatmap normalized to [0, 1] for target class.
        """
        self._register_hooks()
        try:
            tensor = tensor.clone().detach().requires_grad_(True)
            self.model.zero_grad()

            logits = self.model(tensor)

            if target_class_idx is None:
                target_class_idx = int(torch.argmax(logits, dim=1).item())

            score = logits[0, target_class_idx]
            score.backward(retain_graph=False)

            if not self.activations or not self.gradients:
                return np.zeros((7, 7), dtype=np.float32)

            act = self.activations[0]  # (1, 1280, 7, 7)
            grad = self.gradients[0]   # (1, 1280, 7, 7)

            # Global average pooling of gradients gives feature channel weights
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, 1280, 1, 1)

            # Weighted linear combination followed by ReLU
            cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, 7, 7)
            cam_np = cam[0, 0].detach().cpu().numpy()

            # Normalize to [0, 1]
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max > c_min:
                norm_cam = (cam_np - c_min) / (c_max - c_min)
            else:
                norm_cam = np.zeros_like(cam_np)

            return norm_cam.astype(np.float32)

        finally:
            self._remove_hooks()
            self.model.zero_grad()


def generate_gradcam_overlay(
    model: nn.Module,
    image: Image.Image,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.50,
    device: Optional[torch.device] = None
) -> Tuple[Image.Image, str, np.ndarray]:
    """
    Generates a Grad-CAM heatmap overlaid on top of the input image.

    Returns:
        (blended_pil_image: PIL.Image, base64_data_uri: str, raw_heatmap: np.ndarray)
    """
    target_device = device if device is not None else next(model.parameters()).device
    tensor = preprocess_for_inference(image, device=target_device)

    gradcam = GradCAM(model)
    heatmap_7x7 = gradcam.generate(tensor, target_class_idx=target_class_idx)

    # Resize 7x7 activation map to original image resolution
    w, h = image.size
    heatmap_pil = Image.fromarray(heatmap_7x7).resize((w, h), Image.Resampling.BICUBIC)
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32)

    # Apply Jet colormap
    heatmap_color = apply_colormap_jet(heatmap_resized)
    heatmap_img = Image.fromarray(heatmap_color)

    # Blend original image with colormapped heatmap
    rgb_orig = image.convert('RGB')
    blended = Image.blend(rgb_orig, heatmap_img, alpha=alpha)

    # Convert to base64 JPEG data URI for web visualization
    buffer = BytesIO()
    blended.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{encoded}"

    return blended, data_uri, heatmap_resized
