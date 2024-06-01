FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
ENV PORT=8501
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT"]