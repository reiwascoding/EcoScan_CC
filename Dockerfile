# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask API code into the container
COPY app.py .

# Expose the port on which the Flask API will run
EXPOSE 5000

# Set the entrypoint command to run the Flask API
CMD ["python", "app.py"]
