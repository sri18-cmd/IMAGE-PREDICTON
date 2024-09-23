from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import base64
import os

app = Flask(__name__)

# Load your trained model
model_path = 'Cifar-10_Image_Classification_Using_CNNs-master\cnn_100_epochs.h5'

# Load your trained model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print("Model file not found. Please check the file path.")
    exit()

# Define the labels of the dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def prepare_image_for_display(image):
    image = np.squeeze(image, axis=0)  # Remove batch dimension
    image = (image * 255).astype(np.uint8)  # De-normalize to [0, 255]
    return image

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image, target_size=(32, 32))  # Adjust target size as per your model
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = labels[predicted_class]
            
            # Prepare image for display
            display_image = prepare_image_for_display(processed_image)

            # Convert image to base64 string for rendering in HTML
            fig, ax = plt.subplots()
            ax.imshow(display_image)
            ax.axis('off')
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png')
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')

            return render_template("index.html", prediction=predicted_label, img_data=img_base64)
    return render_template("index.html", prediction=None, img_data=None)

if __name__ == "__main__":
    app.run(debug=True)
