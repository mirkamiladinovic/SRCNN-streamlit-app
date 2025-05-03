# SRCNN Image Super-Resolution with Streamlit

This project provides a web interface to apply Super-Resolution Convolutional Neural Networks (SRCNN) to enhance low-resolution images. Built using Streamlit, it allows users to upload images, apply SRCNN, and compare quality metrics visually.

## 📦 Features

- Upload and preview low-resolution images
- Automatically degrade high-res images for testing
- Apply a pre-trained SRCNN model
- Compare original, degraded, and SRCNN-enhanced images
- Visualize PSNR, MSE, and SSIM metrics as tables and graphs
- Download the upscaled image

## 🧪 Installation

```bash
git clone https://github.com/yourusername/srcnn-streamlit-app.git
cd srcnn-streamlit-app
pip install -r requirements.txt
```

## 🚀 Running the App

Ensure you have:
- `source/` folder with high-resolution images
- Pretrained weights file: `model/3051crop_weight_200.h5`

Then run:

```bash
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py
├── src/
│   ├── model.py
│   ├── image_utils.py
│   ├── metrics.py
│   └── predict.py
├── source/
├── images/
├── output/
├── model/
│   └── 3051crop_weight_200.h5
├── requirements.txt
└── README.md
```

## 📊 Example Output

- Visual comparison of image quality
- Metric values and bar charts
