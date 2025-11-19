import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
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

# Configure the Gemini API key from environment variable
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Define the request body model
class ChatRequest(BaseModel):
    sessionId: str
    message: str

# Initialize Gemini API
# The API key will be provided via an environment variable in Render
model = genai.GenerativeModel(
    'gemini-pro',
    system_instruction=(
        "You are RecoveryPlus Doctor Assistant, a professional and knowledgeable virtual doctor who supports "
        "post-surgery and post-operation patients. You provide medically accurate, clear, and structured guidance "
        "in a professional tone. Focus on recovery advice, wound care instructions, pain management, medication "
        "adherence, mobility exercises, diet, and hygiene. Always explain information in a clinical yet patient-friendly way. "
        "If patients describe symptoms such as severe pain, fever, infection signs, bleeding, or breathing difficulties, "
        "firmly instruct them to immediately contact your doctor or emergency services. Never provide formal diagnoses, "
        "prescriptions, or replace real medical consultations. Always remind patients that your guidance is supplementary "
        "and their surgeon/doctor’s advice takes priority."
    )
)

@app.get("/")
def read_root():
    return {"status": "online"}

async def stream_generator(response):
    async for chunk in response:
        if hasattr(chunk, "text"):
            yield chunk.text

@app.post("/gemini-webhook")
async def gemini_webhook(request: ChatRequest):
    user_message = request.message

    try:
        response = await model.generate_content_async(
            [
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
            ],
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

        # Initialize the Gemini Pro Vision model
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        
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

        # Generate content with the vision model
        response = await vision_model.generate_content_async([prompt, img])
        
        # Extract the JSON from the response
        # The response might contain markdown, so we need to clean it
        json_response = response.text.strip().replace('```json', '').replace('```', '')
        
        return JSONResponse(content=json.loads(json_response))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
