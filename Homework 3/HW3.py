import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)



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
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

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

# -------------------------------
# Naive Bayes Implementation
# -------------------------------

def naive_bayes_train(X_train, y_train, bins):
    # train NB on discretized data
    X_train = X_train.copy()
    y_train = np.array(y_train)
    classes = np.unique(y_train)
    priors, cond_probs = {}, {}

    for c in classes:
        priors[c] = np.mean(y_train == c)

    for feat in X_train.columns:
        cond_probs[feat] = {}
        vals = np.unique(X_train[feat])
        for c in classes:
            sub = X_train[y_train == c][feat]
            counts, _ = np.histogram(sub, bins=np.arange(vals.min()-0.5, vals.max()+1.5))
            probs = (counts + 1) / (len(sub) + len(vals))  # Laplace smoothing
            cond_probs[feat][c] = dict(zip(vals, probs))
    return priors, cond_probs


def naive_bayes_predict(X_test, priors, cond_probs):
    # predict labels and probs for ROC
    classes = list(priors.keys())
    y_pred, y_scores = [], []

    for _, row in X_test.iterrows():
        scores = {}
        for c in classes:
            log_p = np.log(priors[c])
            for feat in X_test.columns:
                val = row[feat]
                prob = cond_probs[feat][c].get(val, 1e-6)
                log_p += np.log(prob)
            scores[c] = log_p

        pred = max(scores, key=scores.get)
        y_pred.append(pred)

        exp_s = np.exp(list(scores.values()))
        y_scores.append(exp_s[1] / np.sum(exp_s))
    return np.array(y_pred), np.array(y_scores)


# ---------------------------------
# Naive Bayes Metrics
# ---------------------------------

acc_nb, f1_nb = [], []

for split in range(num_splits):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=split
    )

    split_acc, split_f1 = [], []
    plt.figure(figsize=(8, 6))

    for bins in bin_sizes:
        Xtr, Xte = X_train.copy(), X_test.copy()
        for col in numeric_features:
            edges = np.linspace(Xtr[col].min(), Xtr[col].max(), bins + 1)
            Xtr[col] = np.digitize(Xtr[col], edges[:-1])
            Xte[col] = np.digitize(Xte[col], edges[:-1])

        priors, cond_probs = naive_bayes_train(Xtr, y_train, bins)
        y_pred, y_scores = naive_bayes_predict(Xte, priors, cond_probs)

        acc = accuracy(y_test.tolist(), y_pred.tolist())
        f1 = f1_score(y_test.tolist(), y_pred.tolist())
        fpr, tpr = compute_ROC(y_test.tolist(), y_scores)

        plt.plot(fpr, tpr, label=f'Split {split+1}, Bins {bins}', lw=2)
        split_acc.append(acc)
        split_f1.append(f1)

    acc_nb.append(split_acc)
    f1_nb.append(split_f1)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'Naive Bayes ROC (Split {split+1})')
    plt.legend(); plt.grid(True)

acc_nb, f1_nb = np.array(acc_nb), np.array(f1_nb)
avg_acc_nb = acc_nb.mean(axis=0)
min_acc_nb = acc_nb.min(axis=0)
max_acc_nb = acc_nb.max(axis=0)

for i,b in enumerate(bin_sizes):
    vals = [f"{x:.4f}" for x in acc_nb[:,i]]
    print(f"Bins {b}: Accuracies={vals}, Min={min_acc_nb[i]:.4f}, Max={max_acc_nb[i]:.4f}, Avg={avg_acc_nb[i]:.4f}")

plt.figure(figsize=(10,6))
for i in range(num_splits):
    plt.plot(bin_sizes, acc_nb[i], 'o-', label=f'Split {i+1}')
plt.plot(bin_sizes, avg_acc_nb, 'k--', lw=2, label='Avg')
plt.xlabel('Bins'); plt.ylabel('Accuracy'); plt.title('Naive Bayes Accuracy vs Bins')
plt.legend(); plt.grid(True)

plt.figure(figsize=(10,6))
for i in range(num_splits):
    plt.plot(bin_sizes, f1_nb[i], 'o-', label=f'Split {i+1}')
plt.xlabel('Bins'); plt.ylabel('F1'); plt.title('Naive Bayes F1 vs Bins')
plt.legend(); plt.grid(True)
plt.show()


# -----------------------------
# Cross-Model F1 Comparison
# -----------------------------

f1_nb_vs_dt, f1_dt_vs_nb = [], []

for bins in bin_sizes:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in numeric_features:
        edges = np.linspace(X_train[col].min(), X_train[col].max(), bins + 1)
        X_train[col] = np.digitize(X_train[col], edges[:-1])
        X_test[col] = np.digitize(X_test[col], edges[:-1])

    dt = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train, y_train)
    priors, cond_probs = naive_bayes_train(X_train, y_train, bins)

    y_dt = dt.predict(X_test)
    y_nb, _ = naive_bayes_predict(X_test, priors, cond_probs)

    f1_nb_vs_dt.append(f1_score(y_dt, y_nb))
    f1_dt_vs_nb.append(f1_score(y_nb, y_dt))

plt.figure(figsize=(8,6))
plt.plot(bin_sizes, f1_nb_vs_dt, 'o-', label='NB vs DT GT')
plt.plot(bin_sizes, f1_dt_vs_nb, 's-', label='DT vs NB GT')
plt.xlabel('Bins'); plt.ylabel('F1')
plt.title('Cross-Model F1 Comparison')
plt.legend(); plt.grid(True); plt.show()
