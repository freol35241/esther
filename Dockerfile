FROM python:3.9.13-slim

RUN adduser --system --no-create-home nonroot

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

USER nonroot

ENTRYPOINT ["python", "-m", "app.main"]
CMD [ "--help" ]