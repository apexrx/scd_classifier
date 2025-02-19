# SCD Classifier

A neural network model for detecting Sickle Cell Disease (SCD) from images of red blood cell (RBC) samples.

## Description

This project presents a machine learning approach to identify SCD by analyzing RBC images. Utilizing the **erythrocytesIDB** dataset, the model processes microscopic images to extract key features and employs a neural network for classification.

## Dataset

The **erythrocytesIDB** dataset, developed by the University of the Balearic Islands, comprises 626 images of individual RBCs categorized into:
- **Normal Erythrocytes:** 202 images
- **Sickle Cells:** 211 images
- **Other Deformations:** 213 images

Each image is in JPG format with a resolution of 80x80 pixels.

More Info: http://erythrocytesidb.uib.es/

## Preprocessing

### Image Preprocessing

The preprocessing pipeline includes:
- **Edge Detection:** Identifies cell boundaries.
- **Object Segmentation:** Isolates individual RBCs.
- **Binary Fill Operations:** Refines segmented regions.
- **Size Filtering:** Retains structures exceeding a predefined size threshold.

### Feature Extraction

From each processed image, the following features are extracted and stored in a CSV file:
- **Area:** The number of pixels within the cell boundary.
- **Perimeter:** The length of the cell boundary.
- **Circularity:** Indicates how close the shape is to a perfect circle.

## Data Splitting

The dataset is divided as follows:
- **60% Training**
- **20% Validation**
- **20% Testing**

## Neural Network Model

Implemented using TensorFlow/Keras, the feedforward neural network comprises:
- **Input Layer:** 3 neurons corresponding to the extracted features.
- **Two Hidden Layers:** Each with 32 neurons and ReLU activation.
- **Output Layer:** Single neuron with sigmoid activation for binary classification.
- **Loss Function:** Binary cross-entropy.
- **Optimizer:** Adam.

## Performance

The model achieved an accuracy of **96%** on the test dataset.

## Future Improvements

- **Dataset Expansion:** Incorporate additional data to enhance model generalization.
- **Exploration of CNN Architectures:** Investigate models like ResNet-50 and MobileNet for direct image classification. citeturn0search6
- **Hyperparameter Optimization:** Fine-tune parameters to boost accuracy.