import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
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

client = genai.Client()

class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.get("/")
def read_root():
    return {"status": "online"}

def stream_generator(response):
    for chunk in response:
        print("STREAM CHUNK:", chunk)
        if hasattr(chunk, "delta") and hasattr(chunk.delta, "content") and chunk.delta.content:
            yield chunk.delta.content
        else:
            yield str(chunk) + "\n"

@app.post("/gemini-webhook")
def gemini_webhook(request: ChatRequest):
    user_message = request.message

    try:
        messages = [
            {
                "author": "user",
                "content": (
                    "System instruction: You are a friendly recovery assistant. "
                    "Always reply politely and supportively.\n\n"
                    f"User message: {user_message}"
                )
            }
        ]
        
        response = client.models.generate_content_stream(
            model="gemini-2.0-turbo",
            messages=messages,
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
            "isOverlayRequired": False
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

        try:
            parsed_text = ocr_data["ParsedResults"][0]["ParsedText"]
        except Exception:
            parsed_text = ""

        medications = []

        med_pattern = re.compile(
            r"(?P<name>[A-Za-z0-9\-]+)\s*(?P<dosage>\d+mg|\d+ml)?\s*(?P<frequency>\d+\s*times\s*a\s*day|\w+:\w+)?",
            re.IGNORECASE
        )

        for match in med_pattern.finditer(parsed_text):
            med_name = match.group("name")
            dosage = match.group("dosage") or ""
            frequency = match.group("frequency") or ""

            times = []
            if "once a day" in frequency.lower():
                times = ["08:00"]
            elif "twice a day" in frequency.lower():
                times = ["08:00", "20:00"]
            elif "three times a day" in frequency.lower():
                times = ["08:00", "14:00", "20:00"]
            elif re.match(r"\d{1,2}:\d{2}", frequency):
                times = [frequency]

            medications.append({
                "name": med_name,
                "dosage": dosage,
                "times": times
            })

        return JSONResponse(content={"medications": medications})

    except Exception as e:
        print("Error in /process_prescription:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})