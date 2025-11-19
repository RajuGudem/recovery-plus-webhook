import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from fastapi.responses import StreamingResponse, JSONResponse
import json
import requests

# Set the environment variable for Google Cloud credentials
# This is crucial for the Dialogflow client to find the credentials file.
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dialogflow-credentials.json"

app = FastAPI()

# Add CORS middleware
origins = [
    os.environ.get("FRONTEND_URL", "http://localhost:3000")  # In production, replace with your Flutter app's domain.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini Client
# The client automatically gets the API key from the GEMINI_API_KEY environment variable.
client = genai.Client()

# Define the request body model
class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.get("/")
def read_root():
    return {"status": "online"}

def stream_generator(response):
    for chunk in response:
        print("STREAM CHUNK:", chunk)
        # Try direct text
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text
        # Try candidates → content → text (common Gemini v1beta fallback)
        elif hasattr(chunk, "candidates"):
            try:
                candidate = chunk.candidates[0]
                part = candidate.content.parts[0]
                text_val = getattr(part, "text", None)
                if text_val:
                    yield text_val
                else:
                    yield str(chunk) + "\n"
            except Exception:
                yield str(chunk) + "\n"
        # Fallback: yield whole object for inspection
        else:
            yield str(chunk) + "\n"

@app.post("/gemini-webhook")
def gemini_webhook(request: ChatRequest):
    user_message = request.message

    try:
        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "System instruction: You are a friendly recovery assistant. "
                            "Always reply politely and supportively.\n\n"
                            f"User message: {user_message}"
                        )
                    }
                ]
            }
        ]
        
        response = client.models.generate_content_stream(
            model="models/gemini-2.0-flash",
            contents=contents
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

        return JSONResponse(content={"text": parsed_text})

    except Exception as e:
        print("Error in /process_prescription:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})