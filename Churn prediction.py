#!/usr/bin/env python
# coding: utf-8

# Step 1 = Data Loading

# In[1]:


import pandas as pd

# Load the dataset
data = pd.read_csv("churn.csv")

# Display the first few rows of the dataset to understand its structure
print(data.head())


# Step 2 = Exploratory Data Analysis (EDA)

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of the target variable
sns.countplot(x='Exited', data=data)
plt.title('Distribution of Exited')
plt.show()

# Visualize the distribution of numerical features
sns.pairplot(data=data, vars=['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary'], hue='Exited')
plt.show()

# Visualize the distribution of categorical features
sns.countplot(x='Geography', hue='Exited', data=data)
plt.title('Distribution of Exited by Geography')
plt.show()

sns.countplot(x='Gender', hue='Exited', data=data)
plt.title('Distribution of Exited by Gender')
plt.show()


# Step 3 = Data preprocessing 

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Split data into features and target variable
X = data.drop(columns=['Exited'])
y = data['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define numerical and categorical features
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
categorical_features = ['Geography', 'Gender']

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing pipeline to training and testing sets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Output the processed training and testing sets
print("Processed training data shape:", X_train_processed.shape)
print("Processed testing data shape:", X_test_processed.shape)


# Step 4 = Model selection

# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize models
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()


# Step 4 = Model Selection and Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define hyperparameters grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Initialize GridSearchCV with Random Forest model
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=param_grid,
                              scoring='accuracy',
                              cv=5,
                              n_jobs=-1)

# Perform Grid Search to find the best hyperparameters
rf_grid_search.fit(X_train_processed, y_train)

# Get the best hyperparameters and model
best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest Model Parameters:")
print(rf_grid_search.best_params_)

# Get the mean cross-validated score of the best estimator
print("Best Mean Cross-validated Score:", rf_grid_search.best_score_)


# Step 5 = Model training

# In[ ]:


# Train Logistic Regression model
logistic_regression.fit(X_train_processed, y_train)

# Train Random Forest model
random_forest.fit(X_train_processed, y_train)


# Step 6 = Model Evaluation

# In[ ]:


# Evaluate Logistic Regression model
y_pred_lr = logistic_regression.predict(X_test_processed)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Model Accuracy:", lr_accuracy)
print("Classification Report for Logistic Regression Model:")
print(classification_report(y_test, y_pred_lr))

# Evaluate Random Forest model
y_pred_rf = random_forest.predict(X_test_processed)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Model Accuracy:", rf_accuracy)
print("Classification Report for Random Forest Model:")
print(classification_report(y_test, y_pred_rf))


# Step 6 = Model Evaluation (using the best model from Grid Search)

# In[ ]:


y_pred_best_rf = best_rf_model.predict(X_test_processed)
best_rf_accuracy = accuracy_score(y_test, y_pred_best_rf)
print("\nBest Random Forest Model Accuracy:", best_rf_accuracy)
print("Classification Report for Best Random Forest Model:")
print(classification_report(y_test, y_pred_best_rf))


# In[ ]:




