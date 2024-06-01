FROM python:3.8-slim
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
RUN mkdir -p /content/drive/MyDrive/Bootcamp/Day 33 - Checkpoint 1/sentiment
COPY ./model/tf_model.h5 /content/drive/MyDrive/Bootcamp/Day 33 - Checkpoint 1/sentiment/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]