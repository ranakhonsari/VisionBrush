# 🎨 VisionBrush - AI-Powered Inpainting with LangSAM & ControlNet 🎨

Welcome to **VisionBrush**, an AI-powered image inpainting tool that leverages **LangSAM** for text-to-mask segmentation and **ControlNet with Stable Diffusion** for creative transformations. 🚀

## ✨ Features
- **Text-Guided Masking**: Use natural language to specify areas to modify.
- **AI-Powered Inpainting**: ControlNet replaces masked areas with new content.
- **Web Interface**: Interact via **Streamlit**.
- **API Support**: Process images via **FastAPI**.
- **WandB Integration**: Track experiments and visualize results.

## 📸 Example Use Case
Given an image of a **classic guitar**, you can transform it into an **electric guitar** using text prompts.

## 🛠️ Installation
- pip install -r requirements.txt

## 🚀 Running the Application
### 1️⃣ Start FastAPI Backend
```bash
python app.py
```

### 2️⃣ Start Streamlit Frontend
```bash
streamlit run streamlit_app.py   
```

## 🔬 How It Works
### 1️⃣ **Mask Generation (LangSAM)**
- User provides an **image** and **text prompt**.
- LangSAM predicts a **segmentation mask** based on the prompt.

### 2️⃣ **Inpainting (ControlNet)**
- Masked image and a **replacement prompt** are fed into ControlNet.
- The AI model generates a **new image** by inpainting the selected region.

## 📊 Experiment Tracking with Weights & Biases
This project integrates **WandB** for logging. 
The text prompts and images (uploaded, masked, and inpainted images) are saved for evaluation.


##Enjoy creating with **VisionBrush**! 🖌️
