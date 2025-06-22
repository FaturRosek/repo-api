import numpy as np
import tensorflow.lite as tflite
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = FastAPI()

# Load model TFLite
interpreter = tflite.Interpreter(model_path="mobilenetv2_batik.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    'Motif Bhajit', 'Motif Bhang Kopi', 'Motif Koceng Arenduh', 
    'Motif Kuaci', 'Motif Malate Rok-Rok', 'Motif Mata Ikan', 
    'Motif Ompay', 'Motif Perreng', 'Motif Ramok', 'Motif Sek Mlayah'
]

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output_data)
    return class_names[predicted_class], float(np.max(output_data))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    label, confidence = predict_image(img)

    return {
        "class": label,
        "confidence": confidence
    }
