FROM python:3.9.13-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/* .

ENTRYPOINT ["python", "./main.py"]