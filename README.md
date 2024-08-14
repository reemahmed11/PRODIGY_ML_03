# PRODIGY_ML_03  
**SVM-Based Image Classification for Cats and Dogs**

## Overview

This project focuses on implementing a Support Vector Machine (SVM) model to classify images of cats and dogs. The dataset used contains 25,000 labeled images, and the goal is to accurately distinguish between the two classes using image preprocessing and dimensionality reduction techniques like PCA (Principal Component Analysis).

## Dataset

The dataset used for this project is the Cats and Dogs dataset from Kaggle. It contains 12,500 images of cats and 12,500 images of dogs, all in `.jpg` format.

- **Dataset Link:** [Cats and Dogs Dataset](https://www.kaggle.com/c/dogs-vs-cats)](https://www.kaggle.com/c/dogs-vs-cats/data)

## SVM and PCA Implementation

### Algorithm Description

Support Vector Machine (SVM) is a supervised learning algorithm commonly used for classification tasks. It works by finding the hyperplane that best separates the classes in the feature space.

#### Steps Involved:

1. **Image Preprocessing:**
    - Each image is resized to a uniform size of 50x50 pixels.
    - The pixel values are normalized and flattened into a 1D array to serve as features for the model.

2. **Dimensionality Reduction using PCA:**
    - PCA is applied to reduce the dimensionality of the feature space, retaining the most important features while speeding up the training process.

3. **Model Training:**
    - An SVM model is trained on the preprocessed images using GridSearchCV to find the optimal hyperparameters.

4. **Evaluation:**
    - The model's performance is evaluated using accuracy, confusion matrix, and classification report metrics.

### Project Implementation

#### Data Preparation

- **Load and Preprocess the Data:**
    - Images from the dataset are loaded, resized, normalized, and flattened into feature vectors.
  
- **Train-Test Split:**
    - The dataset is split into training and testing sets, with 80% used for training and 20% for testing.

#### Model Training

- **PCA and SVM Pipeline:**
    - A pipeline is created that first applies PCA for dimensionality reduction and then trains an SVM model.
  
- **Hyperparameter Tuning:**
    - GridSearchCV is used to find the best combination of PCA components and SVM parameters.

#### Model Evaluation

- **Accuracy:**
    - The model's accuracy is calculated on the test set.

- **Confusion Matrix:**
    - A confusion matrix is plotted to visualize the classification performance.

- **Classification Report:**
    - A detailed classification report is generated, showing precision, recall, and F1-score for each class.

#### Unseen Image Prediction

- **Prediction on Unseen Data:**
    - The trained model is used to predict the class of an unseen image, with the results visualized along with the input image.

## Results and Insights

- **Model Performance:**
    - The SVM model achieved high accuracy in classifying the images, demonstrating the effectiveness of PCA and SVM for this task.

- **Visualizations:**
    - Confusion matrix and sample predictions are provided to illustrate the model's performance.

## Contact

For any questions or further information, please feel free to reach out:

- **Email:** reemahmedm501@gmail.com

## Contributing

If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

