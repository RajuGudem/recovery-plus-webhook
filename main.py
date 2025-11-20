import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from fastapi.responses import StreamingResponse, JSONResponse
import json
import requests
import re
from PIL import Image, ImageEnhance
from PIL import ImageFilter
from PIL import ImageOps
import io

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
        # --- Preprocess image to improve OCR accuracy ---
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))

            # --- Convert to grayscale ---
            pil_image = pil_image.convert("L")

            # --- Auto contrast enhancement ---
            from PIL import ImageOps
            pil_image = ImageOps.autocontrast(pil_image)

            # --- Remove noise using stronger median filter ---
            pil_image = pil_image.filter(ImageFilter.MedianFilter(size=5))

            # --- Apply Unsharp Mask for better text edges ---
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))

            # --- Adaptive thresholding (manual Otsu-like method) ---
            histogram = pil_image.histogram()
            total = sum(histogram)
            sumB = 0
            wB = 0
            maximum = 0.0
            sum1 = sum(i * histogram[i] for i in range(256))

            threshold = 0
            for i in range(256):
                wB += histogram[i]
                if wB == 0:
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += i * histogram[i]
                mB = sumB / wB
                mF = (sum1 - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between >= maximum:
                    threshold = i
                    maximum = between

            # Apply threshold
            pil_image = pil_image.point(lambda p: 255 if p > threshold else 0)

            # --- Save processed image back to bytes ---
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format="PNG")
            image_bytes = img_buffer.getvalue()
        except Exception as e:
            print("Preprocessing failed, using original image. Error:", e)
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
        print("DEBUG Parsed Prescription:\n", parsed_text)
        
        medications = []
        lines = parsed_text.split("\n")

        # More realistic medicine pattern
        medicine_line_pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z0-9\-\s]{1,50})\s*(?P<dosage>\d{1,4}\s?(mg|ml|mcg|MCG|g))",
            re.IGNORECASE
        )

        # Frequency patterns found in Indian prescriptions
        frequency_patterns = {
            "1-0-1": ["08:00", "20:00"],
            "1-1-1": ["08:00", "14:00", "20:00"],
            "0-0-1": ["20:00"],
            "0-1-1": ["14:00", "20:00"],
            "1-1-0": ["08:00", "14:00"],
            "od": ["08:00"],
            "bd": ["08:00", "20:00"],
            "tds": ["08:00", "14:00", "20:00"],
            "hs": ["20:00"],
            "once": ["08:00"],
            "twice": ["08:00", "20:00"],
            "daily": ["08:00"]
        }

        # --- Improved medicine detection (even without dosage) ---
        medicine_name_only_pattern = re.compile(
            r"\b([A-Za-z][A-Za-z0-9\-]{2,30})\b",
            re.IGNORECASE
        )

        medicine_keywords = [
            "tab", "tablet", "cap", "capsule", "syr", "syrup",
            "inj", "injection", "drop", "drops", "cream", "ointment"
        ]

        for line in lines:
            line_clean = line.strip()
            if len(line_clean) < 2:
                continue

            lower_line = line_clean.lower()

            # Step 1: Detect lines that look like medicine instructions
            is_medicine_line = False

            # Contains medicine keyword
            if any(k in lower_line for k in medicine_keywords):
                is_medicine_line = True

            # Contains frequency
            if any(freq in lower_line for freq in frequency_patterns.keys()):
                is_medicine_line = True

            # Contains mg/ml but name may be missing
            if re.search(r"\d+\s?(mg|ml|mcg|g)", lower_line):
                is_medicine_line = True

            if not is_medicine_line:
                continue

            # Step 2: Extract dosage (if present)
            dosage_match = re.search(r"(\d{1,4}\s?(mg|ml|mcg|g))", line_clean, re.IGNORECASE)
            dosage = dosage_match.group(1) if dosage_match else ""

            # Step 3: Extract name even without dosage
            name = ""
            tokens = line_clean.split()
            for t in tokens:
                if t.lower() in medicine_keywords:
                    continue
                if re.match(r"[A-Za-z][A-Za-z0-9\-]{2,30}", t):
                    name = t
                    break
            if not name:
                name_match = medicine_name_only_pattern.search(line_clean)
                if name_match:
                    name = name_match.group(1)

            if not name:
                continue

            # Step 4: Extract frequencies
            times = []
            for key, tlist in frequency_patterns.items():
                if key.lower() in lower_line:
                    times = tlist
                    break

            # Step 5: Look for HH:MM manual times
            time_match = re.findall(r"([0-2]?\d:[0-5]\d)", line_clean)
            if time_match:
                times = time_match

            # Step 6: Default time
            if not times:
                times = ["08:00"]

            medications.append({
                "name": name.strip(),
                "dosage": dosage.strip() if dosage else "",
                "timings": times
            })

        print("DEBUG Final Parsed Output:", {"medications": medications, "exercises": []})
        return JSONResponse(content={"medications": medications, "exercises": []}) # exercises are not handled here

    except Exception as e:
        print(f"Error in /process_prescription: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})