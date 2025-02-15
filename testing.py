from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering, AutoTokenizer,
    AutoModelForCausalLM
)
from diffusers import StableDiffusionPipeline
import io
import base64

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
## Image models
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

## Text generation model
text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
text_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

## Image generation model
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

def process_text(prompt: str):
    try:
        if prompt.lower().startswith("generate image:"):
            image_prompt = prompt[len("generate image:"):].strip()
            image = pipe(image_prompt).images[0]
            
            # Convert to base64
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')
            
            return {"text": f"Generated image: {image_prompt}", "image": encoded_image}
        else:
            inputs = text_tokenizer(prompt, return_tensors="pt").to(device)
            outputs = text_model.generate(**inputs, max_length=100)
            response = text_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"text": response, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_image(image: Image.Image, prompt: str = None):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if prompt:
            # Visual Question Answering
            inputs = vqa_processor(image, prompt, return_tensors="pt").to(device)
            outputs = vqa_model.generate(**inputs)
            answer = vqa_processor.decode(outputs[0], skip_special_tokens=True)
            return {"text": answer, "image": None}
        else:
            # Image captioning
            inputs = caption_processor(image, return_tensors="pt").to(device)
            outputs = caption_model.generate(**inputs)
            caption = caption_processor.decode(outputs[0], skip_special_tokens=True)
            return {"text": caption, "image": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(
    prompt: str = Form(None),
    images: list[UploadFile] = File(None),
):
    try:
        # Handle image generation requests
        if not images and prompt and prompt.lower().startswith("generate image:"):
            return process_text(prompt)

        # Handle text-only requests
        if not images:
            if not prompt:
                raise HTTPException(400, "Text prompt required")
            return process_text(prompt)

        # Handle image-based requests
        results = []
        for image in images:
            image_data = await image.read()
            image_pil = Image.open(io.BytesIO(image_data))
            result = process_image(image_pil, prompt)
            results.append(result)

        combined_text = " ".join([res["text"] for res in results])
        return {"text": combined_text, "image": None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Vision ChatBot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
