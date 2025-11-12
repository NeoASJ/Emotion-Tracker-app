# Use a lightweight base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app
COPY emotion_model_sep_ravdess.h5 /app/
COPY emotion_model_sep_crema-d.h5 /app/
COPY emotion_model_sep_emodb.h5 /app/
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
