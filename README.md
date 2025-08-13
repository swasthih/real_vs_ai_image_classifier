Real vs AI-Generated Image Classification
This project is a deep learning-based system that detects whether an image is Real or AI-generated.
It uses a Convolutional Neural Network (CNN) built with PyTorch and is deployed as a Flask web application with a simple, responsive frontend.

Dataset Used
CIFAKE Dataset — A publicly available dataset containing real images and AI-generated synthetic images.
CIFAKE on Kaggle

How It Works
Images are preprocessed (resized, normalized, and converted to tensors).

The trained CNN model classifies the image into Real or AI-generated.

The Flask backend processes image uploads and sends predictions to the frontend.

The result is displayed on a clean, responsive web interface.
