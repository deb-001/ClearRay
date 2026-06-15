# 🩻 ClearRay: Medical X-Ray Super-Resolution Platform

ClearRay is a web-based medical image enhancement platform developed using Flask and PyTorch. The system enables 4× super-resolution of low-resolution X-ray images and provides comparative evaluation across multiple state-of-the-art super-resolution models.

The project integrates a custom-trained ESRGAN + XAAHA model and allows users to compare its performance with RealESRGAN, SRCNN, SRDenseNet, and SwinIR through a unified interface.

---

## Features

- 4× X-Ray Image Upscaling
- ESRGAN + XAAHA Proposed Model
- Multi-Model Comparison
- PSNR Evaluation
- SSIM Evaluation
- Flask-Based Web Interface
- Real-Time Processing
- Medical Image Enhancement

---

## Available Models

| Model | Description |
|---------|---------|
| ESRGAN + XAAHA | Proposed model developed in this project |
| RealESRGAN | GAN-based super-resolution model |
| SRCNN | CNN-based super-resolution model |
| SRDenseNet | DenseNet-based super-resolution model |
| SwinIR | Transformer-based super-resolution model |

---

## Technology Stack

### Backend

- Python
- Flask
- PyTorch

### Deep Learning

- ESRGAN
- XAAHA
- RealESRGAN
- SRCNN
- SRDenseNet
- SwinIR

### Image Processing

- NumPy
- Pillow
- scikit-image

### Frontend

- HTML
- CSS
- JavaScript

---

## Project Structure

```text
XRayProject/
│
├── model/
├── static/
├── templates/
├── screenshots/
│
├── app.py
├── comparison_models.py
├── model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/deb-001/ClearRay.git
```

Move into the project:

```bash
cd ClearRay
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Download Model Weights

Model files are hosted separately because of GitHub file size limitations.

Download all models from:

**https://drive.google.com/drive/folders/18fWl8noRmQ16muOpds8d_qtilAKzTS30?usp=sharing**

Place all downloaded `.pth` files inside:

```text
model/
```

---

## Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Evaluation Metrics

The platform evaluates super-resolution performance using:

### PSNR

Peak Signal-to-Noise Ratio

### SSIM

Structural Similarity Index

---

## Screenshots

### Homepage

![Homepage](screenshots/homepage.png)

### Enhanced Output

![Output](screenshots/output.png)

### Model Comparison

![Comparison](screenshots/comparison.png)


---

## Research Objective

The objective of this project is to enhance low-resolution medical X-ray images while preserving critical anatomical structures, enabling better visualization and supporting diagnostic analysis.

---

## Authors

Debanjan Kauri

---

## License

Academic and Research Use Only.