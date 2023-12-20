# Inception-Image-Classifier
his repository contains a PyTorch implementation of an image classification model leveraging an Inception-based architecture. The model employs a modified encoder from the VGG network, followed by several Inception modules, and a final classifier to categorize images into different classes.

# Overview
The Inception Image Classifier is designed to handle complex image classification tasks by extracting robust features through a combination of convolutional and inception layers. The use of Inception modules allows the model to capture information at various scales, improving its ability to recognize patterns in images.

# Model Architecture
![image](https://github.com/ZanePrance/Inception-Image-Classifier/assets/141082203/b443165f-423a-4bee-9896-0d33f20411e5)

The model consists of:

1. Encoder: Based on VGG network's first few layers for initial feature extraction.
2. Inception Modules: Capture and concatenate features from different kernel sizes.
3. Batch Normalization: Applied to stabilize learning and normalize feature distribution.
4. Adaptive Average Pooling: Reduces the spatial dimensions to a fixed size.
5. Classifier: Fully connected layers with dropout and batch normalization for the final classification.

# Customization
You can customize the model to fit different datasets or modify the architecture by changing the number of Inception modules or other hyperparameters.
