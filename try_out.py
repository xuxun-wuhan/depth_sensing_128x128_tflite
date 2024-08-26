import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="test.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_image(image_path):
    # Load image and resize to 128x128
    image = Image.open(image_path)
    image = image.resize((128, 128))
    image_np = np.array(image)
    
    # Ensure the image is in uint8 format
    image_np = image_np.astype(np.uint8)
    
    return image_np

def run_model(image_np):
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image_np, axis=0))

    # Run inference
    interpreter.invoke()

    # Get output tensor
    depth_map = interpreter.get_tensor(output_details[0]['index'])

    # Remove batch dimension and ensure it is uint8
    depth_map = np.squeeze(depth_map).astype(np.uint8)

    return depth_map

def display_images(original, depth_map):
    # Display original and depth map side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth_map, cmap='gray')
    plt.axis("off")

    plt.show()

def main(image_path):
    image_np = load_image(image_path)
    depth_map = run_model(image_np)
    display_images(image_np, depth_map)

if __name__ == "__main__":
    # Replace 'out00001.png' with your image path
    main("out00001.png")
    #main("out00002.png")
    #main("out00003.png")
