FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TRUTHLENS_PUBLIC=1 PORT=8000
RUN pip install --no-cache-dir "anthropic>=0.40"
COPY truthlens ./truthlens
COPY web ./web
COPY data ./data
COPY server.py ./
EXPOSE 8000
CMD ["python", "server.py"]
