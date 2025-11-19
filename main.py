import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from fastapi.responses import StreamingResponse, JSONResponse
import json

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
                "role": "system",
                "parts": [
                    {"text": "Hello! I'm here to support you through your recovery. How are you feeling today?"}
                ]
            },
            {
                "role": "user",
                "parts": [
                    {"text": user_message}
                ]
            },
        ]
        
        response = client.models.generate_content_stream(
            model="gemini-1.0-pro",
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

        # Prepare the prompt for the vision model
        prompt = """
        Analyze the attached prescription image and extract the following information in JSON format:
        - A list of medications with their name, dosage, and timings.
        - A list of exercises with their name, duration, and frequency.
        
        Example JSON output:
        {
          "medications": [
            {
              "name": "Medication Name",
              "dosage": "Dosage",
              "timings": ["Morning", "Afternoon", "Night"]
            }
          ],
          "exercises": [
            {
              "name": "Exercise Name",
              "duration": "Duration",
              "frequency": "Frequency"
            }
          ]
        }
        """

        contents = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": image.content_type,
                            "data": image_bytes
                        }
                    }
                ]
            }
        ]

        # Generate content with the vision model
        response = client.models.generate_content(
            model="gemini-1.0-pro-vision",
            contents=contents
        )
        
        # Extract the JSON from the response
        # The response might contain markdown, so we need to clean it
        raw = response.text or ""
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        json_response = cleaned
        
        return JSONResponse(content=json.loads(json_response))

    except Exception as e:
        print("Error in /process_prescription:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})