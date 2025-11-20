import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from fastapi.responses import StreamingResponse, JSONResponse
import json
import requests
import re

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
            model="llama-3.1-8b-instant", # A common Groq model
            stream=True
        )

        return StreamingResponse(stream_generator(response), media_type="text/plain")
    except Exception as e:
        print(f"Error generating content: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing your request."})

class PrescriptionRequest(BaseModel):
    image: UploadFile

@app.post("/process_prescription")
async def process_prescription(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        OCR_API_KEY = os.environ.get("OCR_SPACE_API_KEY")

        payload = {
            "apikey": OCR_API_KEY,
            "language": "eng",
            "isOverlayRequired": False,
            "detectOrientation": True,
            "scale": True
        }
        files = {
            "file": (image.filename, image_bytes, image.content_type)
        }
        ocr_response = requests.post(
            "https://api.ocr.space/parse/image",
            data=payload,
            files=files
        )
        ocr_data = ocr_response.json()

        if not ocr_data.get("ParsedResults"):
            return JSONResponse(status_code=400, content={"error": "Could not parse image."})

        parsed_text = ocr_data["ParsedResults"][0]["ParsedText"]
        
        medications = []

        # More realistic medicine pattern
        medicine_line_pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z0-9\-\s]{2,20})\s*(?P<dosage>\d+\s?(?:mg|ml|MCG|mcg|g))?",
            re.IGNORECASE
        )

        # Frequency patterns found in Indian prescriptions
        frequency_patterns = {
            "1-0-1": ["08:00", "20:00"],
            "1-1-1": ["08:00", "14:00", "20:00"],
            "0-0-1": ["20:00"],
            "0-1-1": ["14:00", "20:00"],
            "1-1-0": ["08:00", "14:00"],
            "od": ["08:00"],
            "bd": ["08:00", "20:00"],
            "tds": ["08:00", "14:00", "20:00"],
            "hs": ["20:00"],
            "once": ["08:00"],
            "twice": ["08:00", "20:00"],
            "daily": ["08:00"]
        }

        lines = parsed_text.split("\n")

        for line in lines:
            line_clean = line.strip()

            if len(line_clean) < 3:
                continue

            # Match medicine name + dosage
            med_match = medicine_line_pattern.search(line_clean)
            if not med_match:
                continue

            name = med_match.group("name").strip()
            dosage = med_match.group("dosage") or ""

            # Default times if no frequency found
            times = []

            # Search for freq pattern
            for key, tlist in frequency_patterns.items():
                if key.lower() in line_clean.lower():
                    times = tlist
                    break

            # If no explicit time, check for raw HH:MM
            time_match = re.findall(r"([0-2]?\d:[0-5]\d)", line_clean)
            if time_match:
                times = time_match

            # If still no times → default OD (once daily)
            if not times:
                times = ["08:00"]

            medications.append({
                "name": name,
                "dosage": dosage,
                "timings": times
            })

        return JSONResponse(content={"medications": medications, "exercises": []}) # exercises are not handled here

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})