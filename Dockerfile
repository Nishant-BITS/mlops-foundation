FROM --platform=linux/amd64 python:3.9
WORKDIR /app
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "src.app:app"]