import os
import base64
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

        prompt = """
        Analyze the attached prescription image and extract medication information.
        Return a JSON object with a single key "medications".
        The value of "medications" should be a list of objects, where each object has three keys: "name", "dosage", and "timings".
        - "name": The name of the medication.
        - "dosage": The dosage (e.g., "500mg", "1 tablet"). If not present, use an empty string.
        - "timings": A list of strings representing the time of day. Use "Morning", "Noon", or "Night". If the prescription says "1-0-1", use ["Morning", "Night"]. If it says "BD" or "twice daily", use ["Morning", "Night"]. If no timing is specified, use ["As directed"].

        Example output:
        {
          "medications": [
            {
              "name": "Dolo",
              "dosage": "650mg",
              "timings": ["Morning", "Night"]
            }
          ]
        }
        """

        response = deepseek_client.chat.completions.create(
            model="deepseek-vl-chat",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                },
            ],
            max_tokens=2000,
            stream=False  # We need the full JSON response at once
        )

        response_text = response.choices[0].message.content
        
        # Clean the response to get only the JSON part
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print(f"Error: No JSON found in DeepSeek response. Response was: {response_text}")
            return JSONResponse(status_code=500, content={"error": "Failed to parse prescription from AI response."})

        json_string = json_match.group(0)
        parsed_json = json.loads(json_string)

        # Ensure the response has the expected structure
        if "medications" not in parsed_json:
            parsed_json = {"medications": [], "exercises": []}
        else:
            parsed_json["exercises"] = []


        return JSONResponse(content=parsed_json)

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})