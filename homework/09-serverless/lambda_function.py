import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

MODEL_PATH = 'hair_classifier_empty.onnx'
INPUT_SIZE = (200, 200)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)

def predict(url):
    img = download_image(url)
    img = prepare_image(img, INPUT_SIZE)
    x = preprocess_image(img)
    result = session.run([output_name], {input_name: x})
    return float(result[0][0][0])

def lambda_handler(event, context):
    url = event['url']
    prediction = predict(url)
    return {'prediction': prediction}