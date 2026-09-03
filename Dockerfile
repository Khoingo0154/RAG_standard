FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/
COPY telegram_bot/ telegram_bot/
COPY ingestion/ ingestion/
COPY retrieval/ retrieval/
COPY shared/ shared/
COPY evals/ evals/

EXPOSE 8000

CMD ["python", "-m", "api.main"]
