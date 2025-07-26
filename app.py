import gradio as gr
import numpy as np

def classify_digit(image):
    image = image.resize((28, 28)).convert("L")  # Resize to 28x28 and convert to grayscale
    image_arr = np.array(image).reshape(1, -1)   # Flatten image to 1D array
    prediction = rf_clf.predict(image_arr)[0]    # Predict using trained classifier
    return f"Predicted Digit: {prediction}"

app = gr.Interface(
    fn=classify_digit,
    inputs=gr.Image(type="pil", image_mode='L'), # Accept PIL image, grayscale mode
    outputs="text",
    title="MNIST Digit Recognizer",
    description="Draw or upload a digit image (28x28 grayscale)"
)

app.launch()
