# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc (equivalent to python -B)
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr (equivalent to python -u)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set the working directory in the container
WORKDIR /app

# Install dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Run the application using python (app.py handles the Waitress server)
CMD ["python", "app.py"]
