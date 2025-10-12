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