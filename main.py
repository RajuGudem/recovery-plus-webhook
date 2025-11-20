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
        
        # Keywords to identify lines with medication info
        med_keywords = ['mg', 'ml', 'tablet', 'capsule', 'daily', 'twice', 'night', 'morning', 'afternoon', 'noon']
        
        # Regex to find medication name, dosage, and frequency on a single line
        med_pattern = re.compile(
            r"^(?P<name>[a-zA-Z0-9\s.-]+?)\s*(?:\((?P<generic_name>[a-zA-Z\s]+)\))?\s*(?P<dosage>\d+\s*(?:mg|ml|g))?",
            re.IGNORECASE
        )
        
        # Regex for 1-0-1 style frequency
        freq_pattern = re.compile(r'(\d)\s*-\s*(\d)\s*-\s*(\d)')

        lines = parsed_text.split('\n')
        for line in lines:
            line = line.strip()
            if not any(keyword in line.lower() for keyword in med_keywords):
                continue

            match = med_pattern.match(line)
            if not match:
                continue

            med_name = match.group("name").strip()
            if len(med_name) < 3: # Filter out very short, likely incorrect matches
                continue

            dosage = match.group("dosage") or ""
            times = []

            # Check for 1-0-1 format
            freq_match = freq_pattern.search(line)
            if freq_match:
                morning, noon, night = [int(i) > 0 for i in freq_match.groups()]
                if morning:
                    times.append("Morning")
                if noon:
                    times.append("Noon")
                if night:
                    times.append("Night")
            else:
                # Check for keyword frequencies
                line_lower = line.lower()
                if "once" in line_lower or "daily" in line_lower:
                    times = ["Morning"]
                elif "twice" in line_lower:
                    times = ["Morning", "Night"]
                elif "three times" in line_lower:
                    times = ["Morning", "Noon", "Night"]
                
                if not times: # If no frequency words found, check for time of day
                    if "morning" in line_lower:
                        times.append("Morning")
                    if "noon" in line_lower or "afternoon" in line_lower:
                        times.append("Noon")
                    if "night" in line_lower or "evening" in line_lower:
                        times.append("Night")

            # Avoid adding duplicates
            is_duplicate = False
            for med in medications:
                if med['name'].lower() == med_name.lower():
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                medications.append({
                    "name": med_name,
                    "dosage": dosage.strip(),
                    "timings": times if times else ["As directed"]
                })

        return JSONResponse(content={"medications": medications, "exercises": []}) # exercises are not handled here

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})