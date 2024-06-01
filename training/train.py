from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
from google.cloud import storage

import tensorflow as tf
import json
import pandas as pd
import numpy as np
from io import StringIO

print('load distilbert')
# load model and tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")


def train():
    print('start training')
    file = tf.io.gfile.GFile(
        'gs://machine-learning-samples/datasets/sentiment/imdb/csv/dataset.csv', mode='r').read()
    df = pd.read_csv(StringIO(file))

    sentiments = df['sentiment'].values.tolist()
    reviews = df['review'].values.tolist()

    training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(
    reviews, 
    sentiments, 
    test_size=0.2, 
    stratify=sentiments  
    )

    train_encodings = tokenizer(training_sentences, 
                            truncation=True, 
                            padding=True,
                            max_length=512, 
                            return_tensors="tf")

    val_encodings = tokenizer(validation_sentences, 
                          truncation=True, 
                          padding=True, 
                          max_length=512, 
                          return_tensors="tf")

    train_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask']
    }, training_labels)).shuffle(100).batch(16)

    val_dataset = tf.data.Dataset.from_tensor_slices(({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask']
    }, validation_labels)).shuffle(100).batch(16)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, epsilon=1e-08),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
    )

    model.fit(
        train_dataset,  # Dataset already shuffled and batched
        epochs=10,
        validation_data=val_dataset
    )

    model.save_pretrained("./sentiment")

    upload_blob('machine-learning-samples',
                './sentiment/config.json', 'models/sentiment/config.json')
    upload_blob('machine-learning-samples',
                './sentiment/tf_model.h5', 'models/sentiment/tf_model.h5')


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client(project='machine-learning-firda')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


if __name__ == '__main__':
    print('main')
    train()