#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv("churn.csv")
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())


# In[10]:


# Check for missing values
print("Missing values in the dataset:")
print(data.isnull().sum())


# In[11]:


# Perform one-hot encoding for 'Geography' column
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# Binary encode 'Gender' column (Female: 0, Male: 1)
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})

# Display the first few rows of the updated dataset
print("First few rows of the updated dataset:")
print(data.head())


# In[12]:


from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Select numerical features for scaling
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Scale the numerical features
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Display the first few rows of the scaled dataset
print("First few rows of the scaled dataset:")
print(data.head())


# In[14]:


from sklearn.model_selection import train_test_split

# Split the dataset into features (X) and target variable (y)
X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = data['Exited']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize and train the logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = logistic_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[16]:


from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (L1 or L2)
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Inverse of regularization strength
}

# Initialize grid search
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Initialize logistic regression model with the best hyperparameters
best_logistic_model = LogisticRegression(**best_params, random_state=42)

# Train the model with the best hyperparameters
best_logistic_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_best = best_logistic_model.predict(X_test)

# Evaluate the model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("\nAccuracy with Best Hyperparameters:", accuracy_best)
print("\nClassification Report with Best Hyperparameters:")
print(classification_report(y_test, y_pred_best))
print("\nConfusion Matrix with Best Hyperparameters:")
print(confusion_matrix(y_test, y_pred_best))


# In[17]:


from sklearn.ensemble import RandomForestClassifier

# Initialize and train the random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy with Random Forest:", accuracy_rf)
print("\nClassification Report with Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix with Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))


# In[18]:


# Define hyperparameters to tune
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize grid search for random forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')

# Perform grid search
grid_search_rf.fit(X_train, y_train)

# Get the best hyperparameters
best_params_rf = grid_search_rf.best_params_
print("Best Hyperparameters for Random Forest:", best_params_rf)

# Initialize random forest model with the best hyperparameters
best_rf_model = RandomForestClassifier(**best_params_rf, random_state=42)

# Train the model with the best hyperparameters
best_rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_best_rf = best_rf_model.predict(X_test)

# Evaluate the model
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print("\nAccuracy with Best Hyperparameters for Random Forest:", accuracy_best_rf)
print("\nClassification Report with Best Hyperparameters for Random Forest:")
print(classification_report(y_test, y_pred_best_rf))
print("\nConfusion Matrix with Best Hyperparameters for Random Forest:")
print(confusion_matrix(y_test, y_pred_best_rf))


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

# Extract feature importances from the trained random forest model
feature_importances = best_rf_model.feature_importances_

# Create a DataFrame to display feature importances
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[20]:


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Define SMOTE and random forest pipeline
smote = SMOTE(random_state=42)
rf_pipeline = Pipeline([('smote', smote), ('rf', best_rf_model)])

# Train the model with SMOTE for oversampling
rf_pipeline.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf_smote = rf_pipeline.predict(X_test)

# Evaluate the model with SMOTE
accuracy_rf_smote = accuracy_score(y_test, y_pred_rf_smote)
print("Accuracy with SMOTE:", accuracy_rf_smote)
print("\nClassification Report with SMOTE:")
print(classification_report(y_test, y_pred_rf_smote))
print("\nConfusion Matrix with SMOTE:")
print(confusion_matrix(y_test, y_pred_rf_smote))


# In[21]:


pip install xgboost


# In[22]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("Accuracy with XGBoost:", accuracy_xgb)

# Classification report
print("\nClassification Report with XGBoost:")
print(classification_report(y_test, y_pred_xgb))

# Confusion matrix
print("\nConfusion Matrix with XGBoost:")
print(confusion_matrix(y_test, y_pred_xgb))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_model.feature_importances_, y=X.columns)
plt.title('Feature Importances with XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[23]:


# Saving best model
import joblib
# Save the model to disk
filename = 'best_rf_model.sav'  
joblib.dump(best_rf_model, filename)


# In[24]:


# Load the saved model from disk
import joblib
model = joblib.load('best_rf_model.sav')


# In[44]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the trained model
model = joblib.load('best_rf_model.sav')

# Define the Streamlit app
def main():
    st.title("Churn Prediction App")
    st.write("This app predicts whether a customer will churn or not.")

    # Add file uploader to upload CSV data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded data into a DataFrame
        data = pd.read_csv(uploaded_file)

        # Preprocess the data
        data = preprocess_data(data)

        # Make predictions
        predictions = model.predict(data)

        # Display predictions
        st.write("Predictions:")
        st.write(predictions)

# Function to preprocess the data
def preprocess_data(data):
    # Perform one-hot encoding for 'Geography' column
    data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

    # Binary encode 'Gender' column (Female: 0, Male: 1)
    data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})

    # Scale the numerical features
    scaler = StandardScaler()
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

# Run the Streamlit app
if __name__ == "__main__":
    main()


# In[ ]:




