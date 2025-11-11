import os
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
app.template_folder = 'templates'

# Load the pre-trained model
model_path = 'tumor_detection.h5'
model = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        
        # Ensure the file is an image
        if file.filename == '':
            return "No selected file"
        if not file:
            return "File not found"

        # Convert the file to a PIL image
        img = Image.open(file.stream)

        # Preprocess the image
        img = img.resize((128, 128))
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape

        # Make a prediction
        prediction = model.predict(img_array)

        # Debugging output: print prediction to check the model output
        print("Model prediction output:", prediction)

        # Process the prediction
        predicted_class = np.argmax(prediction[0])  # Get the index of the highest probability
        result = 'Normal Cell' if predicted_class == 0 else 'Tumor Cell'

        return result

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
