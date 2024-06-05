# Handwritten Digit Recognition using PyTorch
## Overview
In this project, I implemented a convolutional neural network (CNN) using PyTorch to recognize handwritten digits from the MNIST dataset. The goal was to achieve high accuracy in classifying the digits (0-9) present in the images.

## Dataset
I used the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits along with their corresponding labels. The images were converted into tensors and normalized witha mean 0f 0.5 and standard deviation of 0.5 to center them so that theycan be scaled.The dataset was divided into training, validation, and test sets with a batch size of 64.

## Model Architecture
I designed a neural network architecture that is a feedforward neural network (FNN) with multiple hidden layers. Specifically, it is a fully connected neural network designed for classifying handwritten digits from the MNIST dataset. It consists of four hidden layers with ReLU activation functions and dropout layers to prevent overfitting. The input layer has 28 * 28 = 784 neurons, corresponding to the flattened input images. The output layer has 10 neurons, representing the 10 possible digit classes (0-9). Each hidden layer progressively reduces the dimensionality of the input data, starting from 512 neurons in the first hidden layer and ending with 64 neurons before the output layer.

## Training
The model was trained using the training set and validated using the validation set. We used the Adam optimizer with cross-entropy loss as the loss function. The training process involved iterating over multiple epochs, adjusting hyperparameters, and monitoring performance as you can see from the below training and validation loss graph.
[Training Validation Loss](TVLoss.png)

## Evaluation
After training, the model was evaluated using the test set to assess its accuracy and loss on unseen data. The evaluation metrics provided insights into the model's performance and generalization ability.

## Results
The trained model achieved a test accuracy of approximately 97%, demonstrating its effectiveness in recognizing handwritten digits. We compared our results with benchmark performances and discussed the implications.

## Conclusion
In conclusion, we successfully implemented a CNN-based digit recognition system using PyTorch. The project showcased the power of deep learning in solving classification tasks and provided a foundation for further research and applications in image recognition.

