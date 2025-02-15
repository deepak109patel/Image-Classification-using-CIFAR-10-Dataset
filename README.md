# CIFAR-10 Image Classification with TensorFlow

This project focuses on classifying images from the CIFAR-10 dataset using various neural network architectures, including an Artificial Neural Network (ANN) and Convolutional Neural Networks (CNNs). The project also explores the use of data augmentation to improve model performance.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

### Classes
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Project Steps

1. **Data Loading and Preprocessing**
   - Loaded the CIFAR-10 dataset using TensorFlow's `datasets.cifar10.load_data()`.
   - Normalized the pixel values to be between 0 and 1 by dividing by 255.
   - Reshaped the labels to be 1D arrays.

2. **Exploratory Data Analysis (EDA)**
   - Visualized some of the images along with their labels to understand the dataset.

3. **Artificial Neural Network (ANN)**
   - Built a simple ANN model using TensorFlow's `Sequential` API.
   - Compiled the model with the SGD optimizer and sparse categorical cross-entropy loss.
   - Trained the model for 100 epochs.
   - Evaluated the model on the test set and calculated accuracy.

4. **Convolutional Neural Network (CNN)**
   - Built a CNN model with two convolutional layers followed by max-pooling layers, a flatten layer, and two dense layers.
   - Compiled the model with the Adam optimizer and sparse categorical cross-entropy loss.
   - Trained the model for 20 epochs.
   - Evaluated the model on the test set and calculated accuracy.

5. **Data Augmentation**
   - Used `ImageDataGenerator` to apply data augmentation techniques such as rotation, shifting, shearing, zooming, and flipping.
   - Built a more complex CNN model with an additional convolutional layer and dropout for regularization.
   - Compiled the model with the Adam optimizer and sparse categorical cross-entropy loss.
   - Trained the model with early stopping to prevent overfitting.
   - Evaluated the model on the test set and calculated accuracy.

6. **Model Saving**
   - Saved the trained models (`ann_model.h5`, `cnn_model.h5`, `cnn_augmented_model.h5`) for future use.

7. **Model Evaluation**
   - Calculated accuracy, confusion matrix, and classification report for the models.
   - Compared the performance of the ANN, CNN, and CNN with data augmentation.

## Results

- **ANN Model**: Achieved an accuracy of approximately X% on the test set.
- **CNN Model**: Achieved an accuracy of approximately Y% on the test set.
- **CNN with Data Augmentation**: Achieved an accuracy of approximately Z% on the test set.

## Usage

To run the code:

1. Ensure you have TensorFlow and other required libraries installed.
2. Run the provided Python script in a Jupyter notebook or Google Colab environment.
3. The models will be trained and evaluated, and the results will be displayed.

## Files

- `ann_model.h5`: Saved ANN model.
- `cnn_model.h5`: Saved CNN model.
- `cnn_augmented_model.h5`: Saved CNN model with data augmentation.

## Dependencies

- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Documentation](https://keras.io/api/)
