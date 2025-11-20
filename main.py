import os
import base64
import re
import json
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from deepseek import DeepSeekClient
from deepseek_ocr import DeepSeekOCR
from fastapi.responses import StreamingResponse, JSONResponse

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

# Configure Groq API key for chatbot
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
groq_client = Groq(api_key=groq_api_key)

# Configure DeepSeek API key for prescription scanning
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")
deepseek_client = DeepSeekClient(api_key=deepseek_api_key)
ocr_client = DeepSeekOCR(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1/ocr"
)


class ChatRequest(BaseModel):
    sessionId: str
    message: str

@app.get("/")
def read_root():
    return {"status": "online"}

# Stream generator for the Groq chatbot
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
        print(f"Error in /groq-webhook: {e}")
        return JSONResponse(status_code=500, content={"error": "An error occurred while processing your request."})


@app.post("/process_prescription")
async def process_prescription(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        
        # Save uploaded file to a temporary file
        suffix = os.path.splitext(image.filename)[1] if image.filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        # Step 1: OCR
        ocr_result = ocr_client.parse(tmp_path)
        if isinstance(ocr_result, dict):
            extracted_text = ocr_result.get("text", "")
        else:
            extracted_text = str(ocr_result)

        if not extracted_text.strip():
            return JSONResponse(status_code=400, content={"error": "No text detected in the prescription image."})

        # Step 2: Prepare prompt for JSON parsing
        prompt = f"""
        You are an expert medical prescription parser. Analyze the following prescription text and extract medication information.
        Return ONLY a valid JSON object with a single key "medications".
        The value of "medications" should be a list of objects, where each object has three keys: "name", "dosage", and "timings".
        Use the following rules for timings:
        - "1-0-1": ["Morning", "Night"]
        - "1-1-1" or "TDS": ["Morning", "Noon", "Night"]
        - "0-0-1" or "HS": ["Night"]
        - "1-0-0" or "OD": ["Morning"]
        - "BD" or "twice daily": ["Morning", "Night"]
        - If no timing specified: ["As directed"]

        Prescription Text:
        {extracted_text}
        """

        # Step 3: DeepSeek chat completion
        response = deepseek_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-vl-chat",
            max_tokens=3000,
            stream=False
        )

        response_text = response.choices[0].message.content

        # Step 4: Extract JSON safely
        parsed_json = {}
        try:
            # Find first {...} block in response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
            else:
                # Fallback: try json.loads directly
                parsed_json = json.loads(response_text)
        except Exception as json_err:
            print(f"JSON parsing error: {json_err}. Response text: {response_text}")
            return JSONResponse(status_code=500, content={"error": "Failed to parse prescription JSON from AI response."})

        # Step 5: Ensure required keys
        if "medications" not in parsed_json:
            parsed_json["medications"] = []
        if "exercises" not in parsed_json:
            parsed_json["exercises"] = []

        return JSONResponse(content=parsed_json)

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})