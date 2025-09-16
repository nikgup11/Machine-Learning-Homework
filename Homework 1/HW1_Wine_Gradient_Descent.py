from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]

def batch_without_reg(
    X_train,
    Y_train,
    learning_rate = 1e-5,
    tolerance = 1e-6,
    max_iters = 10000,
):
    mse_history = []
    # Combine features and targets for splitting, so outputs will have m x (n+1) shape
    m, n = X_train.shape

    # Add bias to X_train: shape (m, n+1)
    X_train_aug = np.hstack([np.ones((m, 1)), X_train])

    # Step 2: Randomly initialize w = [w0, w1, ..., wn] for (n+1)-dim
    w = np.random.randn(n + 1, 1)

    for iteration in range(max_iters):
        # Step 3: Compute predictions Y_hat = X_train_aug (dot) w
        Y_hat = X_train_aug @ w  # shape (m, 1)
        # Step 4: Compute training MSE
        mse = np.mean((Y_train - Y_hat) ** 2)
        # Step 5: Compute gradient
        grad_mse = (2/m) * (X_train_aug.T @ (Y_hat - Y_train))  # shape (n+1, 1)
        # Step 6: Weight update
        w = w - learning_rate * grad_mse

        # Step 7: Terminate if change in error is less than tolerance
        if np.linalg.norm(grad_mse) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())
    
    # Y_hat for saving and inspection
    np.savetxt("Yhat_training.csv", Y_hat, delimiter=",")
    return w, mse_history



def batch_with_l2_reg(
    X_train,
    Y_train,
    alpha=0.1,
    learning_rate=1e-5,
    tolerance=1e-6,
    max_iters=10000
):
    # Combine features and targets for splitting, so outputs will have m x (n+1) shape
    m, n = X_train.shape
    
    # Add bias to X_train: shape (m, n+1)
    X_aug = np.hstack([np.ones((m, 1)), X_train])
    
    # Step 2: Randomly initialize w = [w0, w1, ..., wn] for (n+1)-dim
    w = np.random.randn(n + 1, 1)
    
    # Track MSE history to remove smallest weight likewise
    mse_history = []

    for iteration in range(max_iters):
        # Step 3: Compute predictions Y_hat = X_train_aug (dot) w
        Y_hat = X_aug @ w
        # Step 4: Compute training MSE
        mse = np.mean((Y_train - Y_hat) ** 2)
        mse_history.append(mse)
        # Step 5: Compute gradient
        grad_mse = (2 / m) * (X_aug.T @ (Y_hat - Y_train))
        # Do not regularize the bias term (w[0])
        grad_l2 = 2 * alpha * np.vstack([[0], w[1:]])
        grad = grad_mse + grad_l2
        # Step 6: Weight update
        w = w - learning_rate * grad

        # Step 7: Terminate if change in error is less than tolerance
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())
    
    # Output Y_hat to csv
    np.savetxt("Yhat_training_l2.csv", Y_hat, delimiter=",")
    return w, mse_history

def retrain_without_smallest_weight(
    X_train, Y_train, w, 
    learning_rate=1e-5, tolerance=1e-6, max_iters=10000, verbose=True
):
    # Exclude bias; find attribute index with smallest weight
    abs_w = np.abs(w[1:].ravel())
    min_idx = np.argmin(abs_w)
    print(f"Removing feature index {min_idx} with weight {w[min_idx+1,0]}")
    
    # Remove that column (attribute) from X_train
    X_new = np.delete(X_train, min_idx, axis=1)
    
    # Retrain using original batch gradient (without regularization)
    w_retrained, hist = batch_without_reg(X_new, Y_train, learning_rate, tolerance, max_iters)
    return w_retrained, hist, min_idx


def batch_with_l1_reg(
    X_train,
    Y_train,
    alpha=0.1,
    learning_rate=1e-5,
    tolerance=1e-6,
    max_iters=10000,
):
    # Combine features and targets for splitting, so outputs will have m x (n+1) shape
    m, n = X_train.shape
    
    # Add bias to X_train: shape (m, n+1)
    X_aug = np.hstack([np.ones((m, 1)), X_train])
    
    # Step 2: Randomly initialize w = [w0, w1, ..., wn] for (n+1)-dim
    w = np.random.randn(n + 1, 1)
    
    # Track MSE history to remove smallest component (value 0) likewise
    mse_history = []

    for iteration in range(max_iters):
        # Step 3: Compute predictions Y_hat = X_train_aug (dot) w
        Y_hat = X_aug @ w
        
        # Step 4: Compute training MSE
        mse = np.mean((Y_train - Y_hat) ** 2)
        mse_history.append(mse)

        # Step 5: Compute gradient
        grad_mse = (2 / m) * (X_aug.T @ (Y_hat - Y_train))
        
        # L1 gradient: sign(w) for w (do not regularize bias term w0)
        w_sign = np.vstack([[0], np.sign(w[1:])])
        grad_l1 = 2 * alpha * w_sign

        # Step 6: update weight
        grad = grad_mse + grad_l1
        w = w - learning_rate * grad

        # Step 7: Terminate if change in error is less than tolerance
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())
    np.savetxt("L1_Yhat_training.csv", Y_hat, delimiter=",")

    # Find zero-weight features (excluding bias)
    zero_weight_indices = np.where(np.isclose(w[1:].ravel(), 0, atol=1e-8))[0]
    if zero_weight_indices.size > 0:
        print(f"Features with zero weights to eliminate: {zero_weight_indices}")
    else:
        print("No features with zero weights found.")

    return w, mse_history, zero_weight_indices

def retrain_without_zero_weights(
    X_train, Y_train, zero_weight_indices,
    learning_rate=1e-5, tolerance=1e-6, max_iters=10000
):
    # Remove columns corresponding to zero-weight features
    X_reduced = np.delete(X_train, zero_weight_indices, axis=1)
    
    # Retrain using original batch gradient
    w_retrained, hist = batch_without_reg(X_reduced, Y_train, learning_rate, tolerance, max_iters)
    return w_retrained, hist, zero_weight_indices

# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data
X = wine_quality.data.features
y = wine_quality.data.targets

data = X.copy()
data['quality'] = y

# INSPECT/ANALYZE DATA SET

# Choose percentages,  rest is testing
t = 0.6  # 60% for training
v = 0.2  # 20% for validation

# Split into temp (train+valid) and test
data_temp, X_test = train_test_split(data, test_size=(1.0 - t - v), random_state=42)

# Split temp into train and validation using the adjusted ratio
X_train, X_valid = train_test_split(data_temp, test_size=(v / (t + v)), random_state=42)

# Ground truths
Y_train = X_train['quality']
Y_valid = X_valid['quality']
Y_test = X_test['quality']

if Y_train.ndim == 1:
    Y_train = Y_train.values.reshape(-1, 1)

print(X_train.shape)
print(Y_train.shape)


# MODEL EXECUTION (uncomment sections likewise to run): 
print("\nBATCH")
batch_without_reg(X_train, Y_train)

# print("\L2 BATCH")
# w_l2, mse_hist = batch_with_l2_reg(X_train, Y_train, alpha=0.05)
# w_retrained, mse_hist_retrained, dropped_col = retrain_without_smallest_weight(X_train, Y_train, w_l2)

# print("\L1 BATCH")
# w_l1, mse_hist_l1, zeros = batch_with_l1_reg(X_train, Y_train, alpha=0.08)
# if zeros.size > 0:
#     w_retrained, mse_retrained, dropped = retrain_without_zero_weights(X_train, Y_train, zeros)


