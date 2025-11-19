import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from fastapi.responses import StreamingResponse, JSONResponse
import json
from PIL import Image
import io

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
        if hasattr(chunk, "text"):
            yield chunk.text

@app.post("/gemini-webhook")
def gemini_webhook(request: ChatRequest):
    user_message = request.message

    try:
        contents = [
            {
                "role": "model",
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
        
        response = client.models.generate_content(
            model="gemini-pro",
            contents=contents,
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
        # Read the image data
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))

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

        contents = [prompt, img]

        # Generate content with the vision model
        response = client.models.generate_content(
            model="gemini-pro-vision",
            contents=contents
        )
        
        # Extract the JSON from the response
        # The response might contain markdown, so we need to clean it
        json_response = response.text.strip().replace('```json', '').replace('```', '')
        
        return JSONResponse(content=json.loads(json_response))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})