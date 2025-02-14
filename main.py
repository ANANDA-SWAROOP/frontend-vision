from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import io

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to process text using Hugging Face API
def process_text(prompt: str):
    try:
        # Use a pre-trained model for text-based responses
        response = f"{prompt}"
        return {"text": response, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to process images using CLIP and BLIP models
def process_image(image: Image.Image, prompt: str = None):
    try:
        # Convert image to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Generate a caption for the image
        inputs = processor(image, prompt, return_tensors="pt") if prompt else processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        return {"text": caption, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API endpoint to handle text and image inputs
@app.post("/api/chat")
async def chat(
    prompt: str = Form(None),
    images: list[UploadFile] = File(None),
):
    try:
        # Handle text-only input
        if not images:
            return process_text(prompt)

        # Handle image input
        results = []
        for image in images:
            image_data = await image.read()
            image_pil = Image.open(io.BytesIO(image_data))
            result = process_image(image_pil, prompt)
            results.append(result)

        # Combine results if multiple images are provided
        combined_text = " ".join([res["text"] for res in results])
        return {"text": combined_text, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Vision ChatBot API!"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
