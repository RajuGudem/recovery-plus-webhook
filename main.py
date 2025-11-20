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
            r"(?P<name>[A-Za-z0-9\-]{2,})\s*(?P<dosage>\d+(?:mg|ml))?\s*(?P<frequency>(?:once|twice|three times|every \d+ hours|[0-2]?\d:[0-5]\d))?",
            re.IGNORECASE
        )

        for match in med_pattern.finditer(parsed_text):
            med_name = match.group("name")
            dosage = match.group("dosage") or ""
            frequency = match.group("frequency") or ""

            if not med_name or len(med_name) < 2:
                continue

            times = []
            freq_lower = frequency.lower()
            if "once" in freq_lower:
                times = ["08:00"]
            elif "twice" in freq_lower:
                times = ["08:00", "20:00"]
            elif "three" in freq_lower:
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