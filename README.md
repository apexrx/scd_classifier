# SCD Classifier: Sickle Cell Disease Detection from Red Blood Cell Images

[![Accuracy](https://img.shields.io/badge/Accuracy-96%25-green)](https://img.shields.io/badge/Accuracy-96%25-green)

A machine learning model for detecting Sickle Cell Disease (SCD) from images of red blood cell (RBC) samples. This project utilizes a combination of image processing techniques, feature extraction, and neural network classification to identify sickle cells.  An ensemble method combining the neural network with a KNN classifier is also implemented to boost performance.

## Overview

This project presents a machine learning approach to identify SCD by analyzing RBC images.  We process microscopic images to extract key features, train a neural network (along with a KNN model), and combine their predictions for a more accurate classification of cells as either healthy or sickle-shaped.  This could be useful in preliminary screening or in resource-limited settings.

## Table of Contents

- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Data Splitting](#data-splitting)
- [Neural Network Model](#neural-network-model)
- [KNN Classifier](#knn-classifier)
- [Ensemble Method](#ensemble-method)
- [Performance](#performance)
- [Usage](#usage)
- [Example Results](#example-results)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

## Dataset

The **erythrocytesIDB** dataset, developed by the University of the Balearic Islands, is used in this project. It comprises 626 images of individual RBCs categorized into:

-   **Normal Erythrocytes:** 202 images
-   **Sickle Cells:** 211 images
-   **Other Deformations:** 213 images

Each image is in JPG format with a resolution of 80x80 pixels.

More Info: [http://erythrocytesidb.uib.es/](http://erythrocytesidb.uib.es/)

## Image Preprocessing

Before feature extraction, images undergo several preprocessing steps to enhance segmentation and analysis:

1.  **Grayscale Conversion:** The input image is converted to grayscale for simplified processing.  Here's an example input image:

    ![Input Image](rbc.jpeg)

2.  **Edge Detection (Canny):**  The Canny edge detection algorithm identifies cell boundaries within the grayscale image.

3.  **Binary Fill Operations:** Fills holes and imperfections within the detected cell regions, creating solid objects.

4.  **Object Segmentation:** Isolates individual RBCs from the background.

5.  **Size Filtering:** Retains structures exceeding a predefined size threshold to remove noise and small artifacts. This step results in a binary mask.

    ![Binary Mask](binary_mask.jpg)

## Feature Extraction

After preprocessing, key features are extracted from each segmented cell:

*   **Area:** The number of pixels within the cell boundary.
*   **Perimeter:** The length of the cell boundary.
*   **Circularity:** Indicates how close the shape is to a perfect circle. This is calculated as  `(4 * pi * Area) / (Perimeter^2)`.

These features are stored in a CSV file (`cell_features.csv`), which is then used for training the machine learning models.

## Data Splitting

The dataset is divided into three subsets:

*   **Training Set (60%):** Used for training the neural network and KNN models.
*   **Validation Set (20%):** Used for tuning hyperparameters and monitoring the model's performance during training.
*   **Testing Set (20%):** Used for evaluating the final model's performance on unseen data.

Data augmentation using RandomOverSampler is performed on the training data to address class imbalance, ensuring the model is robust and doesn't overfit to the majority class.  A `StandardScaler` is fit on the training data and used to scale all the data.  The scaler is saved and reloaded for use in prediction on new images.

## Neural Network Model

The neural network model is implemented using TensorFlow/Keras and consists of the following layers:

*   **Input Layer:** 3 neurons corresponding to the extracted features (area, perimeter, circularity).
*   **Hidden Layers:** Multiple dense layers with ReLU activation, batch normalization, and dropout for regularization.
*   **Output Layer:** Single neuron with sigmoid activation for binary classification (0 for healthy, 1 for sickle cell).

The model is compiled with:

*   **Loss Function:** Binary cross-entropy.
*   **Optimizer:** Adam.
*   **Metrics:** Accuracy, Precision, Recall, and AUC.

## KNN Classifier

A K-Nearest Neighbors (KNN) classifier is also trained using the extracted features. This provides a complementary approach to classification.  The optimal number of neighbors (k) is determined through experimentation, and the trained KNN model is saved for later use.

## Ensemble Method

To improve prediction accuracy, we use an ensemble method that combines the predictions of the neural network and the KNN classifier. The final prediction is made based on a majority voting scheme: if at least one of the models predicts a sickle cell, the cell is classified as sickle.

## Performance

The ensemble model achieved the following performance metrics on the test dataset:

*   **Accuracy:**  87%
*   **Precision (Sickle Cell):** 63%
*   **Recall (Sickle Cell):** 96%

The neural network alone achieved an accuracy of **96%** on the test dataset.  However, we found that an ensemble approach yields higher recall.

## Usage

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas scikit-learn opencv-python scikit-image tensorflow imbalanced-learn joblib
    ```

2.  **Prepare Data:** Ensure the `cell_features.csv` file is generated using the provided preprocessing and feature extraction scripts.

3.  **Run Prediction:**
    ```python
    import os
    import cv2
    import numpy as np
    import joblib
    import tensorflow as tf
    from skimage.feature import canny
    from scipy import ndimage as ndi
    import math
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from google.colab.patches import cv2_imshow
    def image_prep(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Error: Unable to read image at {path}")
        edges = canny(img / 255.0)
        fill_img = ndi.binary_fill_holes(edges)
        label_objects, nb_labels = ndi.label(fill_img)
        mask_size = 400
        sizes = np.bincount(label_objects.ravel())
        mask_sizes = sizes > mask_size
        mask_sizes[0] = 0
        img_cleaned = mask_sizes[label_objects]
        labeled_img, num_features = ndi.label(img_cleaned)
        binary_mask = (labeled_img > 0).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return labeled_img, num_features, img, binary_mask
    def numofneighbour(mat, i, j, searchValue):
        count = 0
        if i > 0 and mat[i - 1][j] == searchValue:
            count += 1
        if j > 0 and mat[i][j - 1] == searchValue:
            count += 1
        if i < len(mat) - 1 and mat[i + 1][j] == searchValue:
            count += 1
        if j < len(mat[i]) - 1 and mat[i][j + 1] == searchValue:
            count += 1
        return count
    def findperimeter(mat, num_features):
        perimeter = [0] * (num_features + 1)
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] != 0:
                    perimeter[mat[i][j]] += (4 - numofneighbour(mat, i, j, mat[i][j]))
        return perimeter
    def extract_area_perim(img, num_features):
        area = [0] * (num_features + 1)
        for i in range(len(img)):
            for j in range(len(img[i])):
                value = img[i][j]
                if value != 0:
                    area[value] += 1
        return area, findperimeter(img, num_features)
    def extract_circularity(area, perimeter):
        return [
            (4 * math.pi * area[i]) / (math.pow(perimeter[i], 2)) if perimeter[i] != 0 else 0
            for i in range(len(area))
        ]
    def convert_to_relative(cellAreas, cellPerims):
        avg_area = np.mean(cellAreas) if cellAreas else 1
        avg_perim = np.mean(cellPerims) if cellPerims else 1
        return [area / avg_area for area in cellAreas], [perim / avg_perim for perim in cellPerims]
    def removeOutliers(area, perim, img):
        if not area:
            return [], [], img
        mean = np.mean(area)
        std = np.std(area)
        new_area, new_perim = [], []
        new_img = img.copy()
        for i in range(len(area)):
            if area[i] <= mean + 3.5 * std:
                new_area.append(area[i])
                new_perim.append(perim[i])
            else:
                new_img[new_img == i + 1] = 0
        if not new_area and area:
            max_index = np.argmax(area)
            new_area.append(area[max_index])
            new_perim.append(perim[max_index])
        return new_area, new_perim, new_img
    def train_knn_model(X_train, y_train, n_neighbors=5):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        return knn
    def save_knn_model(knn_model, model_path='knn_model.pkl'):
        joblib.dump(knn_model, model_path)
        print(f"KNN model saved to {model_path}")
    def load_knn_model(model_path='knn_model.pkl'):
        return joblib.load(model_path)
    def ensemble_predict(nn_model, knn_model, features_scaled):
        if features_scaled.size == 0:
            return np.array([])
        nn_predictions = nn_model.predict(features_scaled)
        nn_labels = (nn_predictions > 0.5).astype(int).flatten()
        knn_predictions = knn_model.predict(features_scaled)
        ensemble_predictions = []
        for nn_pred, knn_pred in zip(nn_labels, knn_predictions):
            ensemble_predictions.append(int(nn_pred + knn_pred >= 1))
        return np.array(ensemble_predictions)
    def predict_sickle_cells(image_path, nn_model_path='sickle_cell_model.keras',
                            knn_model_path='knn_model.pkl', scaler_path='scaler.pkl'):
        nn_model = tf.keras.models.load_model(nn_model_path)
        knn_model = load_knn_model(knn_model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        labeled_img, num_features, original_img_gray, binary_mask = image_prep(image_path)
        cv2.imwrite('binary_mask.jpg', binary_mask)
        areaArray, perimArray = extract_area_perim(labeled_img, num_features)
        areaArray, perimArray = areaArray[1:], perimArray[1:] if len(areaArray) > 1 else ([], [])
        circularityArray = extract_circularity(areaArray, perimArray)
        relativeAreaArray, relativePerimArray = convert_to_relative(areaArray, perimArray)
        _, _, img_no_outliers = removeOutliers(relativeAreaArray, relativePerimArray, labeled_img.copy())
        contours = []
        for i in range(1, num_features + 1):
            mask = np.uint8(labeled_img == i)
            cell_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.append(cell_contours)
        features_for_prediction = np.array([
            [relativeAreaArray[i], relativePerimArray[i], circularityArray[i]]
            for i in range(len(relativeAreaArray))
        ])
        if features_for_prediction.size:
            features_scaled = scaler.transform(features_for_prediction) if scaler else features_for_prediction
        else:
            features_scaled = np.array([])  # Handle the case where no features are extracted
        predicted_labels = ensemble_predict(nn_model, knn_model, features_scaled)
        original_img_color = cv2.cvtColor(original_img_gray, cv2.COLOR_GRAY2BGR)
        marked_image = original_img_color.copy()
        mask_colored = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        for i, cell_contours in enumerate(contours[:len(predicted_labels)]):
            color = (0, 0, 255) if predicted_labels[i] == 1 else (0, 255, 0)
            cv2.drawContours(marked_image, cell_contours, -1, color, 2)
            cv2.drawContours(mask_colored, cell_contours, -1, color, 2)
        cv2.imwrite('sickle_cells_marked.jpg', marked_image)
        cv2.imwrite('sickle_cells_mask_marked.jpg', mask_colored)
        return marked_image, mask_colored, binary_mask
    def main():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import RandomOverSampler
        cols = ["area", "per", "circ", "class"]
        df = pd.read_csv("cell_features.csv", names=cols)
        df = df.iloc[1:].reset_index(drop=True)
        df["class"] = (df['class'] == '1').astype(int)
        train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
        def scale_dataset(dataframe, oversample=False):
            X = dataframe[dataframe.columns[:-1]].values
            y = dataframe[dataframe.columns[-1]].values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            if oversample:
                ros = RandomOverSampler()
                X, y = ros.fit_resample(X, y)
            if dataframe is train:
                joblib.dump(scaler, 'scaler.pkl')
            data = np.hstack((X, np.reshape(y, (-1, 1))))
            return data, X, y
        train, X_train, y_train = scale_dataset(train, oversample=True)
        valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
        test, X_test, y_test = scale_dataset(test, oversample=False)
        knn_model = train_knn_model(X_train, y_train)
        save_knn_model(knn_model)
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = knn_model.predict(X_test)
        print("KNN Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
        nn_model = tf.keras.models.load_model('sickle_cell_model.keras')
        ensemble_preds = ensemble_predict(nn_model, knn_model, X_test)
        print("\nEnsemble Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, ensemble_preds)}")
        print(classification_report(y_test, ensemble_preds))
        input_image_path = 'rbc.jpeg'
        print(f"\nRunning ensemble prediction on {input_image_path}")
        marked_image, mask_colored, binary_mask = predict_sickle_cells(input_image_path)
        cv2_imshow(marked_image)
        cv2_imshow(mask_colored)
        cv2_imshow(binary_mask)
        print("Sickle cell detection and image marking complete. Images saved and displayed.")
    if __name__ == '__main__':
        main()

    ```

## Example Results

The prediction script marks the original image and the binary mask to highlight detected sickle cells:

*   **Marked Image:**  Sickle cells are outlined in red, and healthy cells are outlined in green.

    ![Marked Image](sickle_cells_marked.jpg)

*   **Marked Mask:** The binary mask with sickle cells outlined in red and healthy cells in green.

    ![Marked Mask](sickle_cells_mask_marked.jpg)

## Future Improvements

*   **Dataset Expansion:** Incorporate additional data to enhance model generalization. A larger and more diverse dataset could significantly improve the model's robustness.
*   **Exploration of CNN Architectures:** Investigate Convolutional Neural Network (CNN) models like ResNet-50 and MobileNet for direct image classification. These architectures are designed to automatically learn relevant features from images, potentially eliminating the need for manual feature extraction.
*   **Hyperparameter Optimization:** Fine-tune the hyperparameters of both the neural network and KNN models to further boost accuracy and performance.  Techniques like grid search or Bayesian optimization could be used.
*   **Advanced Image Preprocessing:** Experiment with more sophisticated image preprocessing techniques, such as adaptive thresholding, morphological operations, and noise reduction filters, to improve the quality of cell segmentation.
*   **Incorporate Cell Shape Analysis:** Explore more advanced shape features beyond circularity, such as elongation, convexity, and solidity, to better differentiate between healthy and sickle cells.

## Acknowledgements

*   The **erythrocytesIDB** dataset provided by the University of the Balearic Islands.
*   The TensorFlow and Keras libraries for providing the deep learning framework.
*   The scikit-learn library for machine learning tools.
