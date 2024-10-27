import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load the data
data = pd.read_csv('Churn_Modelling.csv')  

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Drop irrelevant columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Convert categorical variables into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Separate features and target variable
X = data.drop('Exited', axis=1)
y = data['Exited']

# Visualize the distribution of the target variable
sns.countplot(x='Exited', data=data)
plt.title('Distribution of Churn (Exited)')
plt.show()

# Visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# Evaluation
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, y_pred_gb))

# AUC-ROC scores
print("Logistic Regression AUC-ROC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))
print("Random Forest AUC-ROC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
print("Gradient Boosting AUC-ROC:", roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1]))
