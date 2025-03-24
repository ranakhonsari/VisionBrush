import streamlit as st
import requests
import io
from PIL import Image

st.title("AI Image Inpainting")

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Input Prompts
mask_prompt = st.text_input("Enter the mask text prompt", "wheels")
inpaint_prompt = st.text_input("Enter the final text prompt", "meatballs")

if uploaded_file and st.button("Generate"):
    # Send request to FastAPI
    files = {"file": uploaded_file.getvalue()}
    data = {"mask_text_prompt": mask_prompt, "final_text_prompt": inpaint_prompt}
    
    response = requests.post("http://localhost:8000/process/", files={"file": uploaded_file}, data=data)
    
    if response.status_code == 200:
        # Display result
        inpainted_image = Image.open(io.BytesIO(response.content))
        st.image(inpainted_image, caption="Inpainted Image", use_column_width=True)
    else:
        st.error("Processing failed.")

