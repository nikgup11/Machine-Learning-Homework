import numpy as np
import pandas as pd
import seaborn as sns # Used only for generating the heatmap
import matplotlib.pyplot as plt

# Perform Chi-Square test of independence between two categorical columns in a DataFrame.
def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str):
    contingency_table = pd.crosstab(df[col_a], df[col_b])
    observed = contingency_table.values

    # Calculate expected frequencies
    expected = np.outer(contingency_table.sum(axis=1), contingency_table.sum(axis=0)) / observed.sum()

    # Calculate chi-square statistic
    chi2 = ((observed - expected) ** 2 / expected).sum()

    return chi2

# STEP 1: DATA PREPROCESSING

# Load data set
df = pd.read_csv('Churn_Modelling.csv')


# Histogram only for numeric columns
df.drop(columns=['RowNumber', 'CustomerId', 'Exited']).hist(figsize=(12, 10), bins=20)
plt.tight_layout()
# plt.show() # DISPLAY HISTOGRAM

# Box plot for important features
selected_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

plt.figure(figsize=(12, 10))
plt.boxplot([df[col] for col in selected_cols], tick_labels=selected_cols)
plt.title('Boxplots of Selected Numerical Features')
plt.ylabel('Value')
# plt.show() # DISPLAY BOXPLOT

# Check missing values
if df.isnull().sum().sum():
    print("MISSING VALUES FOUND")
    # Imputation (fill missing numeric with median)
    for col in df.select_dtypes(include=['float', 'int']).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Imputation (fill categorical columns with mode)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)


# Categorical Encoding
# Gender
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Geography (one-hot encoding with France as baseline - implied as true when both Germany and Spain are false)
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Drop last name and customer id since they aren't relevant to predictions
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Feature scaling (standardization)
features_to_scale = ['Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
for col in features_to_scale:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std


# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)

# Compute correlation values between numerical feats and target
corr_matrix = df.corr(numeric_only=True)

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
plt.title("Correlation Matrix Heatmap")

# Since data set is really large, show only half via mask since matrix is symmetric 
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
# plt.show() # DISPLAY HEATMAP

# Get feature importance of numerical data with correlation to target
corr_with_target = df.corr(numeric_only=True)['Exited'].sort_values(ascending=False)
print('Numerical Data Correlation:\n', corr_with_target)


# Get feature importance of categorical data via chi-square test
chi2_gender = chi_square_test(df, 'Gender', 'Exited')
chi2_crc = chi_square_test(df, 'HasCrCard', 'Exited')
chi2_active = chi_square_test(df, 'IsActiveMember', 'Exited')

print("\nFeature Importance\nGender FI: ", chi2_gender)
print('Has Credit Card FI: ', chi2_crc)
print('Active Member FI: ', chi2_active)

# STEP 3: TRAIN-TEST SPLIT

# Shuffle the entire DataFrame randomly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index for 80% training data
split = int(0.8 * len(df_shuffled))

# Slice into training and test sets 
train_df = df_shuffled.iloc[:split]
test_df = df_shuffled.iloc[split:] # 20% testing

print('\nTrain-Test Split')
print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(test_df)}")

# Task 4: LOGISTIC REGRESSION MODELING

from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

feature_cols = [col for col in df.columns if col != 'Exited']

X_train = train_df.drop('Exited', axis=1) # Features for training
y_train = train_df['Exited'] # Target variable for training
X_test = test_df.drop('Exited', axis=1) # Features for testing
y_test = test_df['Exited'] # Target variable for testing

# Create logistic regression model
model = linear_model.LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train, y_train) # Train the model
y_pred = model.predict(X_test) # Make predictions on test set

# To do: Hyperparameter tuning and Regularization yet to be implemented

# TASK 5: MODEL EVALUATION
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
report = classification_report(y_test, y_pred) # Get precision, recall, f1-score
precision = report.split()[-6] # Extract precision for positive class from report
recall = report.split()[-4] # Extract recall for positive class from report
f1_score = report.split()[-2] # Extract f1-score for positive class from report

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print(f"\nLogistic Regression Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")

print("\nClassification Report:\n", classification_report(y_test, y_pred))

# STEP 6: MODEL INTERPRETATION (Feature Coefficients)

coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0]
})

coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='AbsCoefficient', ascending=False)

print("\nFeature Coefficients (sorted by absolute importance):")
print(coefficients)

# STEP 7: MODEL OPTIMIZATION — Hyperparameter Tuning & Regularization

from sklearn.model_selection import GridSearchCV

log_reg = linear_model.LogisticRegression(max_iter=10000, solver='liblinear')

param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\nBest Hyperparameters:", grid.best_params_)

# Evaluate optimized model
y_pred_opt = best_model.predict(X_test)
print("\nOptimized Model Performance:")
print(classification_report(y_test, y_pred_opt))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_opt))

# STEP 7 (Continued): Threshold Adjustment

# Use best_model instead of the original model
y_prob = best_model.predict_proba(X_test)[:, 1]
threshold = 0.4  # adjust as needed
y_pred_thresh = (y_prob >= threshold).astype(int)

print(f"\nClassification Report with Threshold = {threshold}:")
print(classification_report(y_test, y_pred_thresh))
print("Confusion Matrix with Threshold:")
print(confusion_matrix(y_test, y_pred_thresh))


# STEP 8: VISUALIZATION — Churn Probability Distribution

plt.figure(figsize=(8, 5))
plt.hist(y_prob, bins=20, edgecolor='black')
plt.title('Churn Probability Distribution')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Frequency')
plt.show()

# STEP 8: Important Feature Visualization

top_features = coefficients.nlargest(10, 'AbsCoefficient')

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.title('Top Logistic Regression Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()





# To do: ROC Curve and AUC yet to be implemented, handle class imbalance if needed

# ROC Curve and AUC
y_prob = best_model.predict_proba(X_test)[:, 1]# Get predicted probabilities for positive class
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--') # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show() # DISPLAY ROC CURVE