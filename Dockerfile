# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Install Tesseract OCR and its language data
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port available (Render will inject $PORT)
EXPOSE $PORT

# Use the shell form of CMD to allow environment variable expansion
CMD uvicorn main:app --host 0.0.0.0 --port $PORT