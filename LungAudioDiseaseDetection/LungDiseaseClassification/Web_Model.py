from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from keras import models

class Web_Model:
    def __init__(self, model_path, classes):
        self.model = models.load_model(model_path)
        self.classes = classes

    def preprocess_image(self, image):
        image = image.resize((224, 224))  # Resize to required input size
        image = np.array(image) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def predict(self, image):
        prediction = self.model.predict(self.preprocess_image(image))
        predicted_class = self.classes[np.argmax(prediction)]
        
        return predicted_class