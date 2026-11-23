# The image a Hugging Face Docker Space builds: the app, the scoring code and the two saved models.
FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app.py gate.py decision.py ./
COPY --chown=user models/baseline.joblib models/thresholds.json models/
COPY --chown=user models/distilbert models/distilbert
EXPOSE 7860
CMD ["python", "app.py"]
