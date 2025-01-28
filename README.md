# SCD Classifier

A neural network model for detecting sickle cell disease (SCD) from images of red blood cell (RBC) samples.

## Dataset
This model utilizes the **ErythrocytesIDB** dataset, which contains microscopic images of RBCs.

## Preprocessing
Before training, the RBC images are preprocessed to extract useful features.

### Image Preprocessing
The images undergo edge detection and object segmentation to identify and isolate individual RBCs. A binary fill operation helps to refine the segmented regions, and only significant structures are retained based on a predefined size threshold.

### Feature Extraction
From each preprocessed image, the following features are extracted and stored in a CSV file:
- **Area**
- **Perimeter**
- **Circularity**

### Data Splitting
The dataset is split as follows:
- **60% Training**
- **20% Validation**
- **20% Testing**

## Neural Network Model
A feedforward neural network is implemented using TensorFlow/Keras with the following characteristics:
- **Input Layer:** 3 features (area, perimeter, circularity)
- **Two Hidden Layers:** Each with 32 neurons and ReLU activation
- **Output Layer:** Single neuron with sigmoid activation for binary classification
- **Loss Function:** Binary cross-entropy
- **Optimizer:** Adam

## Performance
The model achieved **96% accuracy** on the test dataset.

## Future Improvements
- Increase dataset size for better generalization.
- Experiment with convolutional neural networks (CNNs) for direct image classification.
- Fine-tune hyperparameters for improved accuracy.

## License
This project is licensed under the MIT License.

