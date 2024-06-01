import uvicorn

import tensorflow as tf

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

app = FastAPI(title="Sentiment Analysis")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


print('load model')
# load the pretrained model from image
model = TFDistilBertForSequenceClassification.from_pretrained("/content/drive/MyDrive/Bootcamp/Day 33 - Checkpoint 1/sentiment/tf_model.h5")


@app.post('/content/drive/MyDrive/Bootcamp/Day 33 - Checkpoint 1/sentiment')
async def predict(request: Request):
    body = await request.json()

    # batch would be possible if required
    text = body['text']

    tf_batch = tokenizer([text], max_length=128, padding=True,
                         truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['negative', 'positive', 'neutral']
    confidences = tf_predictions[0]

    negative_confidence = confidences[0]
    positive_confidence = confidences[1]
    neutral_confidence = confidences[2]

    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()

    return {'text': text, 'sentiment': labels[label[0]], 'confidence': {'negative': float(negative_confidence), 'positive': float(positive_confidence),  'neutral': float(neutral_confidence)}}


if __name__ == "__main__":
    uvicorn.run(app, debug=True)