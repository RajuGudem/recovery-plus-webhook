import os
import base64
import re
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from deepseek import DeepSeekClient
from fastapi.responses import StreamingResponse, JSONResponse
import json

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

# Configure Groq API key
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
groq_client = Groq(api_key=groq_api_key)

# Configure DeepSeek API key
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")
deepseek_client = DeepSeekClient(api_key=deepseek_api_key)


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


@app.post("/process_prescription")
async def process_prescription(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Use DeepSeek-OCR to extract text
        ocr_response = deepseek_client.ocr(
            image_base64=base64_image,
            language="eng"
        )

        if not ocr_response or "text" not in ocr_response:
            print(f"Error: DeepSeek OCR did not return any text. Response: {ocr_response}")
            return JSONResponse(status_code=500, content={"error": "Failed to extract text from prescription."})

        extracted_text = ocr_response["text"]
        print(f"DEBUG Extracted Text: {extracted_text}")

        # Parse medications from extracted text
        medications = []
        lines = extracted_text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'([A-Za-z0-9\-]+)\s*(.*)', line)
            if match:
                name = match.group(1)
                dosage = match.group(2).strip() if match.group(2) else ""
                timings = []
                if re.search(r'1-0-1|BD|twice daily', line, re.IGNORECASE):
                    timings = ["Morning", "Night"]
                elif re.search(r'1-1-1|TDS|three times daily', line, re.IGNORECASE):
                    timings = ["Morning", "Noon", "Night"]
                elif re.search(r'HS|night', line, re.IGNORECASE):
                    timings = ["Night"]
                else:
                    timings = ["As directed"]

                medications.append({
                    "name": name,
                    "dosage": dosage,
                    "timings": timings
                })

        parsed_json = {
            "medications": medications,
            "exercises": []
        }

        print(f"DEBUG Final Parsed Output: {parsed_json}")
        return JSONResponse(content=parsed_json)

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})