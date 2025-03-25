import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="VisionBrush", layout="wide")
st.title("🎨 VisionBrush - AI Inpainting 🎨")
st.write("This page uses LangSAM and ControlNet for Diffusion models to perform inpainting! Enjoy!")

# Upload Image
uploaded_file = st.file_uploader("First, let's upload an image", type=["jpg", "png", "jpeg"])

# Input Prompts
mask_prompt = st.text_input("What part do you want to change? (e.g., 'car')")
inpaint_prompt = st.text_input("Time to get creative ✨ What do you want to replace it with? (e.g., 'spaceship')")

if uploaded_file and st.button("Run Inpainting 🚀"):
    # Convert uploaded file to bytes
    files = {"file": uploaded_file.getvalue()}
    data = {"mask_text_prompt": mask_prompt, "final_text_prompt": inpaint_prompt}
    
    response = requests.post("http://localhost:8000/process/", files={"file": uploaded_file}, data=data)
    
    if response.status_code == 200:
        # Display the results side by side
        st.write("Your creative masterpiece is ready! 🎨")

        # Convert uploaded file to an image for display
        original_image = Image.open(uploaded_file)

        # Load inpainted image
        inpainted_image = Image.open(io.BytesIO(response.content))

        # Resize both images to the same width while keeping the aspect ratio
        fixed_width = 512
        def resize_with_aspect(image, width):
            aspect_ratio = image.height / image.width
            new_height = int(width * aspect_ratio)
            return image.resize((width, new_height))

        original_resized = resize_with_aspect(original_image, fixed_width)
        inpainted_resized = resize_with_aspect(inpainted_image, fixed_width)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_resized, caption="Original Image", use_column_width=True)

        with col2:
            st.image(inpainted_resized, caption=f"{inpaint_prompt} instead of {mask_prompt}", use_column_width=True)

    else:
        st.error("❗ Processing failed.")
