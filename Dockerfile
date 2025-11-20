# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye


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