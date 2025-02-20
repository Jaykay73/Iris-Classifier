# Iris Species Classification with Random Forest

This project demonstrates how to build a machine learning model to classify iris flowers into different species using the Random Forest algorithm. The dataset used is the famous Iris dataset, which contains measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers: Setosa, Versicolor, and Virginica.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Model Evaluation](#model-evaluation)
- [Feature Importance](#feature-importance)
- [Prediction Function](#prediction-function)
- [Example Prediction](#example-prediction)
- [License](#license)

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Clone the repository or download the script.
2. Ensure you have the `iris.csv` file in the same directory as the script.
3. Run the script using Python:

```bash
python iris_classification.py
```

## Code Overview

The script performs the following steps:

1. **Load the Data**: The Iris dataset is loaded from `iris.csv`.
2. **Separate Features and Target**: The features (sepal length, sepal width, petal length, petal width) are separated from the target (species).
3. **Split the Data**: The data is split into training and testing sets using an 70-30 split.
4. **Scale the Features**: The features are scaled using `StandardScaler` to normalize the data.
5. **Train the Model**: A Random Forest Classifier is trained on the scaled training data.
6. **Make Predictions**: The model makes predictions on the scaled test data.
7. **Evaluate the Model**: The model's performance is evaluated using a classification report and confusion matrix.
8. **Feature Importance**: The importance of each feature is calculated and displayed.
9. **Prediction Function**: A function is provided to make predictions on new data.

## Model Evaluation

The model's performance is evaluated using the following metrics:

- **Classification Report**: Provides precision, recall, F1-score, and support for each class.
- **Confusion Matrix**: Shows the number of correct and incorrect predictions for each class.

```python
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
```

## Feature Importance

The importance of each feature in the model is calculated and displayed in descending order:

```python
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='importance', ascending=False))
```

## Prediction Function

A function `predict_iris` is provided to make predictions on new data. The function takes four arguments: sepal length, sepal width, petal length, and petal width. It returns the predicted species and the confidence level.

```python
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return {
        'predicted_species': prediction[0],
        'probability': max(probabilities[0]) * 100
    }
```

## Example Prediction

An example prediction is made using the `predict_iris` function:

```python
example = predict_iris(5.1, 3.5, 1.4, 0.2)
print("\nExample Prediction:")
print(f"Predicted Species: {example['predicted_species']}")
print(f"Confidence: {example['probability']:.2f}%")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive overview of the project, including installation instructions, usage, code explanation, and model evaluation. It also includes a prediction function and an example prediction to demonstrate how the model can be used in practice.
