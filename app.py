"""
Flask server for the ESRGAN XAAHA Medical Image Upscaler.
Handles image upload, downscaling, super-resolution, metric computation,
and multi-model comparison.
"""

import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template, request, jsonify
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model import Generator
from comparison_models import load_all_models, MODEL_INFO

# ─── Configuration ────────────────────────────────────────────────────────────
SCALE_FACTOR = 4
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model5.pth")
SAMPLE_IMAGE = os.path.join(os.path.dirname(__file__), "00000001_001.png")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGE_SIZE = 2048  # cap input size to avoid OOM

app = Flask(__name__)

# ─── Load ESRGAN XAAHA model ─────────────────────────────────────────────────
print(f"Loading ESRGAN+XAAHA model from {MODEL_PATH} on {DEVICE}...")
generator = Generator(
    in_channels=1, num_features=64, num_rrdb=23,
    growth_rate=32, scale_factor=SCALE_FACTOR, xaaha_reduction=16,
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if "generator" in checkpoint:
    generator.load_state_dict(checkpoint["generator"])
else:
    generator.load_state_dict(checkpoint)
generator.to(DEVICE)
generator.eval()
print("✅ ESRGAN+XAAHA loaded!")

# ─── Load comparison models ──────────────────────────────────────────────────
print("Loading comparison models...")
comparison_models = load_all_models(DEVICE)
print(f"✅ All {len(comparison_models)} comparison models loaded!")

# Combined model registry (for the selector)
ALL_MODELS = {
    "esrgan_xaaha": {
        "name": "ESRGAN + XAAHA",
        "fullname": "ESRGAN with X-Ray Anatomy-Aware Hierarchical Attention",
        "paper": "Our Model (Trained)",
        "description": "23 RRDB blocks + CARP/FDAF attention, trained on X-ray data",
        "trained": True,
        "is_rgb": False,
    }
}
for key, info in MODEL_INFO.items():
    ALL_MODELS[key] = {
        "name": info["name"],
        "fullname": info["fullname"],
        "paper": info["paper"],
        "description": info["description"],
        "trained": info.get("trained", False),
        "is_rgb": info.get("is_rgb", False),
    }


# ─── Helper Functions ─────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prepare_image(pil_img):
    """Prepare image: convert to grayscale, cap size, make divisible by scale factor.
    Returns: (grayscale PIL, gt_np, gt_tensor, lr_tensor, lr_np, w, h, lr_w, lr_h)"""
    img = pil_img.convert("L")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
        img = img.resize((w, h), Image.BICUBIC)
        w, h = img.size

    w = w - (w % SCALE_FACTOR)
    h = h - (h % SCALE_FACTOR)
    img = img.resize((w, h), Image.BICUBIC)

    gt_np = np.array(img).astype(np.float32) / 255.0
    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)

    lr_h, lr_w = h // SCALE_FACTOR, w // SCALE_FACTOR
    lr_tensor = F.interpolate(
        gt_tensor, size=(lr_h, lr_w), mode="bicubic", align_corners=False
    ).clamp(0, 1)
    lr_np = lr_tensor.squeeze().numpy()

    return img, gt_np, gt_tensor, lr_tensor, lr_np, w, h, lr_w, lr_h


def _histogram_match(source, reference):
    """Match the histogram of source to reference (both float32 [0,1] arrays)."""
    src = (source * 255).astype(np.uint8)
    ref = (reference * 255).astype(np.uint8)
    # Compute CDFs
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    # Build lookup table
    lut = np.zeros(256, dtype=np.uint8)
    for s_val in range(256):
        lut[s_val] = np.argmin(np.abs(ref_cdf - src_cdf[s_val]))
    matched = lut[src]
    return matched.astype(np.float32) / 255.0


def upscale_with_model(model_key, lr_tensor, gt_np, w, h):
    """Run SR with a specific model. Returns (sr_np, psnr, ssim)."""
    with torch.no_grad():
        if model_key == "esrgan_xaaha":
            sr_tensor = generator(lr_tensor.to(DEVICE)).clamp(0, 1).cpu()
        else:
            model_info = MODEL_INFO[model_key]
            inp = lr_tensor.to(DEVICE)
            
            # SRCNN needs pre-upscaled input
            if model_info.get("pre_upscale", False):
                inp = F.interpolate(inp, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1)
            
            # Models like RealESRGAN and SwinIR expect 3-channel input
            if model_info.get("is_rgb", False):
                inp = inp.repeat(1, 3, 1, 1)
                
            sr_tensor = comparison_models[model_key](inp).clamp(0, 1).cpu()

            # Reduce back to 1 channel if output is RGB
            if model_info.get("is_rgb", False) and sr_tensor.shape[1] == 3:
                sr_tensor = sr_tensor.mean(dim=1, keepdim=True)

    sr_np = sr_tensor.squeeze().numpy()

    # Resize if shape mismatch
    if sr_np.shape != gt_np.shape:
        sr_pil = Image.fromarray((sr_np * 255).astype(np.uint8))
        sr_pil = sr_pil.resize((gt_np.shape[1], gt_np.shape[0]), Image.BICUBIC)
        sr_np = np.array(sr_pil).astype(np.float32) / 255.0

    # No extra post-processing needed for Real-ESRGAN since the new wrapper handles it natively.

    psnr_val = float(peak_signal_noise_ratio(gt_np, sr_np, data_range=1.0))
    ssim_val = float(structural_similarity(gt_np, sr_np, data_range=1.0))

    return sr_np, psnr_val, ssim_val


def process_single(pil_img, model_key="esrgan_xaaha"):
    """Process image with a single selected model."""
    img, gt_np, gt_tensor, lr_tensor, lr_np, w, h, lr_w, lr_h = prepare_image(pil_img)

    sr_np, psnr_val, ssim_val = upscale_with_model(model_key, lr_tensor, gt_np, w, h)

    gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8), mode="L")
    lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8), mode="L")
    sr_pil = Image.fromarray((sr_np * 255).astype(np.uint8), mode="L")
    lr_display = lr_pil.resize((w, h), Image.NEAREST)

    model_info = ALL_MODELS.get(model_key, ALL_MODELS["esrgan_xaaha"])

    return {
        "model_key": model_key,
        "model_name": model_info["name"],
        "model_trained": model_info["trained"],
        "ground_truth": image_to_base64(gt_pil),
        "low_res": image_to_base64(lr_display),
        "super_res": image_to_base64(sr_pil),
        "psnr": round(psnr_val, 3),
        "ssim": round(ssim_val, 4),
        "dimensions": {
            "original": f"{w}×{h}",
            "low_res": f"{lr_w}×{lr_h}",
            "super_res": f"{sr_np.shape[1]}×{sr_np.shape[0]}",
        },
    }


def process_compare(pil_img):
    """Process image with ALL models for comparison."""
    img, gt_np, gt_tensor, lr_tensor, lr_np, w, h, lr_w, lr_h = prepare_image(pil_img)

    gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8), mode="L")
    lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8), mode="L")
    lr_display = lr_pil.resize((w, h), Image.NEAREST)

    results = []
    for model_key in ALL_MODELS:
        sr_np, psnr_val, ssim_val = upscale_with_model(model_key, lr_tensor, gt_np, w, h)
        sr_pil = Image.fromarray((sr_np * 255).astype(np.uint8), mode="L")

        info = ALL_MODELS[model_key]
        results.append({
            "model_key": model_key,
            "model_name": info["name"],
            "model_fullname": info["fullname"],
            "paper": info["paper"],
            "description": info["description"],
            "trained": info["trained"],
            "super_res": image_to_base64(sr_pil),
            "psnr": round(psnr_val, 3),
            "ssim": round(ssim_val, 4),
        })

    # Sort by PSNR (best first)
    results.sort(key=lambda r: r["psnr"], reverse=True)

    return {
        "ground_truth": image_to_base64(gt_pil),
        "low_res": image_to_base64(lr_display),
        "dimensions": {
            "original": f"{w}×{h}",
            "low_res": f"{lr_w}×{lr_h}",
        },
        "models": results,
    }


# ─── Helper to get PIL image from request ────────────────────────────────────
def get_pil_image():
    if "image" in request.files and request.files["image"].filename:
        return Image.open(request.files["image"].stream)
    elif request.form.get("use_sample") == "true":
        return Image.open(SAMPLE_IMAGE)
    return None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models_list = [{"key": k, **v} for k, v in ALL_MODELS.items()]
    return render_template("index.html", models=models_list)


@app.route("/process", methods=["POST"])
def process():
    """Process with a single selected model."""
    try:
        pil_img = get_pil_image()
        if pil_img is None:
            return jsonify({"error": "No image provided"}), 400

        model_key = request.form.get("model", "esrgan_xaaha")
        if model_key not in ALL_MODELS:
            model_key = "esrgan_xaaha"

        result = process_single(pil_img, model_key)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """Process with ALL models for comparison."""
    try:
        pil_img = get_pil_image()
        if pil_img is None:
            return jsonify({"error": "No image provided"}), 400

        result = process_compare(pil_img)
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def models_list():
    """Return list of available models."""
    return jsonify([{"key": k, **v} for k, v in ALL_MODELS.items()])


@app.route("/sample")
def sample():
    try:
        img = Image.open(SAMPLE_IMAGE).convert("L")
        return jsonify({
            "image": image_to_base64(img),
            "name": os.path.basename(SAMPLE_IMAGE),
            "size": f"{img.size[0]}×{img.size[1]}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
