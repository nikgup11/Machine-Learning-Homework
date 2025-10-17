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


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression training with Gradient Descent
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(epochs):
        # Forward pass
        linear_model = np.dot(X, weights) + bias
        y_hat = sigmoid(linear_model)

        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_hat - y))
        db = (1/m) * np.sum(y_hat - y)

        # Update parameters
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

# Train our model
weights, bias = train_logistic_regression(X_train.values, y_train.values, lr=0.01, epochs=5000)

# Predictions
def predict(X, weights, bias, threshold=0.5):
    linear_model = np.dot(X, weights) + bias
    y_hat = sigmoid(linear_model)
    return (y_hat >= threshold).astype(int), y_hat

y_pred, y_prob = predict(X_test.values, weights, bias, threshold=0.5)

# STEP 6: Feature Coefficients
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': weights
})
coefficients['AbsCoefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values(by='AbsCoefficient', ascending=False)

print("\nFeature Coefficients (sorted by absolute importance):")
print(coefficients)

# STEP 7: Model Evaluation (manual metrics)

def confusion_matrix_manual(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP],
                     [FN, TP]])

cm = confusion_matrix_manual(y_test.values, y_pred)
accuracy = np.mean(y_pred == y_test.values)
precision = cm[1,1] / (cm[0,1] + cm[1,1] + 1e-10)
recall = cm[1,1] / (cm[1,0] + cm[1,1] + 1e-10)
f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

print("\nConfusion Matrix:\n", cm)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# STEP 7 (continued): Try different threshold
threshold = 0.4
y_pred_thresh, _ = predict(X_test.values, weights, bias, threshold=threshold)
cm_thresh = confusion_matrix_manual(y_test.values, y_pred_thresh)
print(f"\nConfusion Matrix with threshold={threshold}:\n", cm_thresh)

# STEP 8: Visualizations

# Churn probability distribution
plt.figure(figsize=(8, 5))
plt.hist(y_prob, bins=20, edgecolor='black')
plt.title('Churn Probability Distribution')
plt.xlabel('Predicted Probability of Churn')
plt.ylabel('Frequency')
plt.show()

# Important feature coefficients
top_features = coefficients.nlargest(10, 'AbsCoefficient')
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.title('Top Logistic Regression Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# ROC Curve
def roc_curve_manual(y_true, y_scores):
    thresholds = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        cm = confusion_matrix_manual(y_true, y_pred)
        TP, FN = cm[1,1], cm[1,0]
        FP, TN = cm[0,1], cm[0,0]
        tpr.append(TP / (TP + FN + 1e-10))
        fpr.append(FP / (FP + TN + 1e-10))
    return fpr, tpr

fpr, tpr = roc_curve_manual(y_test.values, y_prob)

plt.figure()
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Manual)")
plt.legend()
plt.show()
