import streamlit as st
import io
from PIL import Image
import tempfile
from src.model import load_model
from src.image_utils import predict
import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_image_download_link(img, filename, label="Download"):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label=label, data=byte_im, file_name=filename, mime="image/png")

st.title("Super-Resolution with SRCNN")
st.write("Upload a degraded image to upscale using a pretrained SRCNN model.")

uploaded_file = st.file_uploader("Choose a low-res image", type=["jpg", "png", "bmp"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.image(Image.open(image_path), caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running SRCNN..."):
        model = load_model()
        ref, degraded, output, scores_degraded, scores_output = predict(image_path, model)

    col1, col2, col3 = st.columns(3)

    col1.image(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB), caption="Original (Ref)", use_container_width=True)
    col2.image(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB), caption="Degraded", use_container_width=True)
    col3.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="SRCNN Output", use_container_width=True)

    st.write("### Metrics Comparison")
    st.table({
        "Metric": ["PSNR", "MSE", "SSIM"],
        "Degraded": [f"{scores_degraded[0]:.2f}", f"{scores_degraded[1]:.2f}", f"{scores_degraded[2]:.4f}"],
        "SRCNN Output": [f"{scores_output[0]:.2f}", f"{scores_output[1]:.2f}", f"{scores_output[2]:.4f}"]
    })

    metric_names = ["PSNR", "MSE", "SSIM"]
    degraded_vals = scores_degraded
    output_vals = scores_output
    colors = ["orange", "green"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i in range(3):
        bars = axs[i].bar(["Degraded", "SRCNN"], [degraded_vals[i], output_vals[i]], color=colors)
        
        axs[i].set_ylabel(metric_names[i])
        axs[i].set_title(f"{metric_names[i]} Comparison")

        for bar in bars:
            height = bar.get_height()
            axs[i].annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

    st.pyplot(fig)




