import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/random_forest_mnist.pkl")

def predict_digit(image):
    image = image.reshape(1, -1)
    prediction = model.predict(image)
    return f"Predicted Digit: {prediction[0]}"

# Launch app
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=True, source="canvas"),
    outputs="text",
    title="MNIST Digit Classifier"
)

interface.launch()
