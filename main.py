import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from fastapi.responses import StreamingResponse, JSONResponse
import json
import requests
import re
import io
import time

# Optional imports for advanced image processing
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from PIL import Image, ImageOps

app = FastAPI()

origins = [
    os.environ.get("FRONTEND_URL", "http://localhost:3000")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure the Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if groq_api_key:
    print(f"GROQ_API_KEY loaded, starting with: {groq_api_key[:4]}...")
else:
    print("GROQ_API_KEY not found!")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
groq_client = Groq(api_key=groq_api_key)

class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.get("/")
def read_root():
    return {"status": "online"}

def stream_generator(response):
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

@app.post("/groq-webhook")
def groq_webhook(request: ChatRequest):
    user_message = request.message
    try:
        messages = [
            {"role": "system", "content": "You are a friendly recovery assistant. Always reply politely and supportively."},
            {"role": "user", "content": user_message}
        ]
        response = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            stream=True
        )
        return StreamingResponse(stream_generator(response), media_type="text/plain")
    except Exception as e:
        print(f"Error generating content: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing your request."})

async def preprocess_image(image_bytes: bytes) -> bytes:
    """Applies advanced preprocessing steps to the image to improve OCR quality."""
    if OPENCV_AVAILABLE:
        try:
            print("Attempting preprocessing with OpenCV...")
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Denoise
            denoised_img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(denoised_img)
            
            # Adaptive Thresholding
            thresh_img = cv2.adaptiveThreshold(clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            _, buffer = cv2.imencode('.png', thresh_img)
            print("Successfully preprocessed image with OpenCV.")
            return buffer.tobytes()
        except Exception as e:
            print(f"OpenCV preprocessing failed: {e}. Falling back to Pillow.")
            # Fall through to Pillow fallback

    # Pillow fallback
    try:
        print("Attempting preprocessing with Pillow...")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("L")
        pil_image = ImageOps.autocontrast(pil_image)
        pil_image = pil_image.point(lambda p: 255 if p > 128 else 0) # Simple thresholding
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        print("Successfully preprocessed image with Pillow.")
        return buffer.getvalue()
    except Exception as e:
        print(f"Pillow preprocessing failed: {e}. Using original image.")
        return image_bytes

@app.post("/debug-image")
async def debug_image_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    processed_bytes = await preprocess_image(image_bytes)
    return StreamingResponse(io.BytesIO(processed_bytes), media_type="image/png")

@app.post("/process_prescription")
async def process_prescription(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        processed_image_bytes = await preprocess_image(image_bytes)

        OCR_API_KEY = os.environ.get("OCR_SPACE_API_KEY")
        if not OCR_API_KEY:
            return JSONResponse(status_code=500, content={"error": "OCR_SPACE_API_KEY is not set."})

        data = {"apikey": OCR_API_KEY, "OCREngine": "3", "isOverlayRequired": False, "detectOrientation": True, "scale": True}
        files = {"file": ("prescription.png", processed_image_bytes, "image/png")}
        
        ocr_data = None
        for attempt in range(4):
            try:
                ocr_response = requests.post("https://api.ocr.space/parse/image", data=data, files=files, timeout=45)
                print(f"OCR Response Status: {ocr_response.status_code}, Text: {ocr_response.text[:100]}")
                if ocr_response.status_code == 200:
                    ocr_data = ocr_response.json()
                    if ocr_data.get("ParsedResults"):
                        break
                time.sleep(1) # Wait before retrying
            except requests.exceptions.RequestException as req_e:
                print(f"OCR attempt {attempt + 1} failed: {req_e}")
                time.sleep(1)

        if not ocr_data or not ocr_data.get("ParsedResults"):
            return JSONResponse(status_code=400, content={"error": "Could not parse image after multiple attempts."})

        parsed_text = ocr_data["ParsedResults"][0]["ParsedText"]
        print("DEBUG Parsed Prescription:\n", parsed_text)
        
        medications = []
        lines = parsed_text.split("\n")

        medicine_keywords = ["tab", "tablet", "cap", "capsule", "syr", "syrup", "inj", "injection", "drop", "drops", "cream", "ointment"]
        frequency_patterns = {
            "1-0-1": ["Morning", "Night"],
            "1-1-1": ["Morning", "Noon", "Night"],
            "0-0-1": ["Night"],
            "0-1-0": ["Noon"],
            "1-0-0": ["Morning"],
            "1-1-0": ["Morning", "Noon"],
            "0-1-1": ["Noon", "Night"],
            "od": ["Morning"], "bd": ["Morning", "Night"], "tds": ["Morning", "Noon", "Night"],
            "hs": ["Night"], "once": ["Morning"], "twice": ["Morning", "Night"], "daily": ["Morning"]
        }

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 3: continue

            lower_line = line_clean.lower()
            
            # Basic check if the line could be a medication
            if not any(k in lower_line for k in medicine_keywords) and not re.search(r'\d', lower_line):
                continue

            # Extract name, dosage
            name_match = re.match(r"^\s*([a-zA-Z0-9\.\s-]+)", line_clean)
            name = name_match.group(1).strip() if name_match else ""
            
            # Clean up name from keywords
            for kw in medicine_keywords:
                name = re.sub(r'\b' + kw + r'\b', '', name, flags=re.IGNORECASE).strip()

            if len(name) < 3: continue

            dosage_match = re.search(r"(\d+\s?(mg|ml|g|mcg))", line_clean, re.IGNORECASE)
            dosage = dosage_match.group(1) if dosage_match else ""

            # Extract frequencies
            times = []
            for key, tlist in frequency_patterns.items():
                if re.search(r'\b' + key + r'\b', lower_line):
                    times = tlist
                    break
            
            if not times:
                if "morning" in lower_line: times.append("Morning")
                if "noon" in lower_line or "afternoon" in lower_line: times.append("Noon")
                if "night" in lower_line: times.append("Night")

            if not times: times = ["As directed"]

            # Avoid adding duplicates
            if not any(med['name'].lower() == name.lower() for med in medications):
                medications.append({"name": name, "dosage": dosage, "timings": times})

        print("DEBUG Final Parsed Output:", {"medications": medications})
        return JSONResponse(content={"medications": medications, "exercises": []})

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})