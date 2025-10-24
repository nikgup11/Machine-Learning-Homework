import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# Accuracy score calculation
def accuracy(y_true, y_pred):
    correct = 0
    # If y_true matches prediction then its correct, return percentage of correct out of number of y_true instances
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)

# F1 Score Calculation
def f1_score(y_true, y_pred):
    # Index mapping for dataset
    D = list(range(len(y_true)))
    # Observed positive and negative sets
    P = set(i for i, yt in enumerate(y_true) if yt == 1)

    # Sweep through thresholds (1.0 down to 0.0)
    thresholds = np.arange(1.0, -0.01, -0.01)
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0  # <-- Now accounting for false negatives too

        # Evaluate each instance
        for i in D:
            # Predicted positive
            if y_scores[i] >= threshold:
                if i in P:
                    tp += 1  # Correctly identified positive
                else:
                    fp += 1  # Incorrectly identified positive
            else:
                if i in P:
                    fn += 1  # Missed positive (False Negative)

        # Compute precision and recall safely
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Handle division-by-zero for F1
        if (precision + recall) == 0:
            return 0.0

        # Compute F1 score
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

# ROC Calculation - TPR & FPR
def compute_ROC(y_true, y_scores):
    # Test set
    D = list(range(len(y_true)))
    P = set(i for i, yt in enumerate(y_true) if yt == 1)  # Observed positive instances
    N = set(i for i, yt in enumerate(y_true) if yt == 0)  # Observed negative instances
    num_P = len(P)  # number of positives
    num_N = len(N)  # number of negatives

    # Collection of all fpr, tpr pairs; gets split into respective lists at end
    roc_points = []

    # Thresholds: from 1 to 0 in steps of -0.01
    thresholds = np.arange(1.0, -0.01, -0.01)
    for threshold in thresholds:
        fp = 0
        tp = 0
        for i in D:
            # If prediction at i >= threshold, count as positive prediction, otherwise it is false positive
            if y_scores[i] >= threshold:
                if i in P:
                    tp += 1
                else:
                    fp += 1
        # Rate calculation
        tpr = tp / num_P if num_P > 0 else 0
        fpr = fp / num_N if num_N > 0 else 0
        roc_points.append((fpr, tpr))

    # Separate lists for plotting (done in bins loop block)
    fpr_list, tpr_list = zip(*roc_points)
    return np.array(fpr_list), np.array(tpr_list)


# *************************************
# DATA LOAD & PREPROCESSING
# *************************************

# Load the dataset
df = pd.read_csv('Churn_Modelling.csv')

# Drop last name and customer id since they aren't relevant to predictions
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)

# Categorical Encoding
# Geography (one-hot encoding with France as baseline)
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Gender map
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Separate features (X) and target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Feature scaling (standardization)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_features:
    mean = df[col].mean()
    std = df[col].std()
    df[col] = (df[col] - mean) / std


# *************************************
# ID3 METRICS
# *************************************

# Define discretization bin settings
bin_sizes = [5, 10, 15, 20]
num_splits = 5
test_size = 0.33

accuracies_per_split = []
f1_scores_per_split = []

for split in range(num_splits):
    # 33% and random train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split
    )
    
    # Accuracies and F1 score lists for prediction collection
    split_accuracies = []
    split_f1_scores = []

    # Create figure for each split (4 ROC bin curves per split)
    plt.figure(figsize=(8, 6))

    for bins in bin_sizes:
        # Copy over training and testing to discretize numeric features
        X_train_binned = X_train.copy()
        X_test_binned = X_test.copy()

        # Discretize all numerical features into bins [5, 10, 15, 20]
        for col in numeric_features:
            bin = np.linspace(X_train[col].min(), X_train[col].max(), bins + 1)
            X_train_binned[col] = np.digitize(X_train[col], bin[:-1])
            X_test_binned[col] = np.digitize(X_test[col], bin[:-1])

        # Train decision tree (using sklearn for training & prediction)
        tree = DecisionTreeClassifier(criterion='entropy', random_state=split)
        tree.fit(X_train_binned, y_train)

        # Predict on test set
        y_pred = tree.predict(X_test_binned)
        
        # Predict probabilities
        y_scores = tree.predict_proba(X_test_binned)[:, 1]
        
        # Calculate accuracy, F1, ROC (FPR & TPR)
        acc = accuracy(y_test.tolist(), y_pred.tolist())
        f1 = f1_score(y_test, y_pred)
        fpr, tpr = compute_ROC(y_test.tolist(), y_scores)
        
        # Plot ROC curve for current bin size and split
        label = f'Split {split+1}, Bins {bins}'
        plt.plot(fpr, tpr, linewidth=2, label=label)

        split_f1_scores.append(f1)
        split_accuracies.append(acc)

    accuracies_per_split.append(split_accuracies)
    f1_scores_per_split.append(split_f1_scores)

    # Display ROC Curve for each split of bin sizes
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curves for ID3 Decision Tree (Split {split + 1})')
    plt.legend()
    plt.grid(True)

# Convert to np array
accuracies_per_split = np.array(accuracies_per_split)
f1_scores_per_split = np.array(f1_scores_per_split)

# Get max, min, avg accuracy scores
max_accuracy = accuracies_per_split.max(axis=0)
min_accuracy = accuracies_per_split.min(axis=0)
avg_accuracy = accuracies_per_split.mean(axis=0)

# Accuracy output
for i, bins in enumerate(bin_sizes):
    acc_list = [f'{acc:.4f}' for acc in accuracies_per_split[:, i]]
    print(f"Bins: {bins}")
    print(f"\tAccuracies: {acc_list}")
    print(f"\tMin Acc: {min_accuracy[i]:.4f}")
    print(f"\tMax Acc: {max_accuracy[i]:.4f}")
    print(f"\tAvg Acc: {avg_accuracy[i]:.4f}")

# Plotting Accuracy Score
plt.figure(figsize=(10, 6))

# Plot scores for each split
for i in range(num_splits):
    plt.plot(bin_sizes, accuracies_per_split[i], marker='o', label=f'Split {i + 1}')
plt.plot(bin_sizes, avg_accuracy, 'k--', linewidth=2, label='Average Accuracy')

plt.xlabel('Number of bins')
plt.xticks([5, 10, 15, 20])
plt.ylabel('Test accuracy')
plt.title("ID3 Decision Tree Accuracy Vs Bin Sizes")
plt.legend()
plt.grid(True)

# Plotting F1 Scores
plt.figure(figsize=(10, 6))

# Plot scores for each split
for i in range(num_splits):
    plt.plot(bin_sizes, f1_scores_per_split[i], marker='o', label=f'Split {i + 1}')

plt.xlabel('Number of bins')
plt.xticks([5, 10, 15, 20])
plt.ylabel('F1 Score on Test Set')
plt.title('ID3 Decision Tree F1 Score Vs Bin Size')
plt.legend()
plt.grid(True)
plt.show()