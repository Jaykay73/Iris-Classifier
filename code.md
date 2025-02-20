import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('iris.csv')

# Separate features and target
X = data.drop('species', axis=1)
y = data['species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print model evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='importance', ascending=False))

# Function to make predictions on new data
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Create a numpy array with the input features
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return {
        'predicted_species': prediction[0],
        'probability': max(probabilities[0]) * 100
    }

# Example usage of the prediction function
example = predict_iris(5.1, 3.5, 1.4, 0.2)
print("\nExample Prediction:")
print(f"Predicted Species: {example['predicted_species']}")
print(f"Confidence: {example['probability']:.2f}%")
