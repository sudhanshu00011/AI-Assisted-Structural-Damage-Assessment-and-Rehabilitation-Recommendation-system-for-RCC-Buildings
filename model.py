"""
model.py -- U-Net + ResNet34 (ImageNet encoder) for crack segmentation, plus the
single preprocessing function shared by training and inference.

NOT RUN in the authoring sandbox (PyTorch was unavailable there). Run
`python train.py --help` / `python tests/test_pipeline.py` on your machine or Kaggle first.
"""
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(pretrained=True):
    """pretrained=True downloads ImageNet encoder weights (needs internet; on Kaggle turn Internet ON)."""
    weights = "imagenet" if pretrained else None
    try:
        return smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=1)
    except Exception as e:  # weights download blocked
        print(f"[model] could not load pretrained weights ({type(e).__name__}); using RANDOM init")
        return smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)


def preprocess(image_rgb_uint8):
    """HxWx3 uint8 RGB -> float32 CHW tensor, scaled to [0,1] then ImageNet mean/std normalised."""
    x = image_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def load_weights(model, path, device=None):
    device = device or get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_mask(model, image_bgr, infer_size=512, threshold=0.5, device=None):
    """
    Full-photo inference. The image is resized to infer_size x infer_size for the network,
    the probability map is resized BACK to the original resolution, then thresholded.
    Returns (mask uint8 0/255 at ORIGINAL resolution, probability float32 at original resolution).
    """
    device = device or next(model.parameters()).device
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
    x = preprocess(small).unsqueeze(0).to(device)
    model.eval()
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((prob > threshold).astype(np.uint8) * 255), prob
