from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import uvicorn
import os
from pathlib import Path
from masks import MaskGenerator
from inpainting import InpaintingPipeline

app = FastAPI()

# Create directories for storing images
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

mask_generator = MaskGenerator()
inpainting_pipeline = InpaintingPipeline()

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...),
    mask_text_prompt: str = Form(...),
    final_text_prompt: str = Form(...)
):
    """Handle image upload and processing."""
    image_path = f"uploads/{file.filename}"
    
    # Save uploaded file
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the pipeline
    inpainting_pipeline.inpainting(image_path, mask_text_prompt, final_text_prompt, save_results=True)

    # Return the processed image
    output_path = "test_images/inpainted_image.png"
    return FileResponse(output_path, media_type="image/png", filename="inpainted_image.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
