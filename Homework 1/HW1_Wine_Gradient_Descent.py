from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt



# Batch Without Regularization
def batch_without_reg(
    X_train,
    Y_train,
    learning_rate = 1e-5,
    tolerance = 1e-6,
    max_iters = 10000,
    ind_run = True    # Flag set for independent runs where batch w/o regression is run on it's own, retrained data from L1, L2 has it's own plotting functionality
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
    
    if ind_run:
        largest_idx = np.argmax(np.abs(w[1:].ravel()))

        # Get actual feature name from DataFrame columns
        feature_name = X_train.columns[largest_idx]

        # Extract feature and target values as numpy arrays
        feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
        target_values = Y_train.ravel()

        # Plot scatter of feature vs target
        plt.scatter(feature_values, target_values, label='Data points')

        # Plot regression line using learned weights
        regression_line = w[0, 0] + w[largest_idx + 1, 0] * feature_values
        plt.plot(feature_values, regression_line, color='red', label='Regression line')

        plt.xlabel(feature_name)
        plt.ylabel('Quality')
        plt.title('Batch Gradient Descent (Without Regularization)')
        plt.legend()
        plt.show()

    return w, mse_history

# Mini-Batch Without Regularization
def mini_batch_without_reg(
    X_train,
    Y_train,
    batch_size=64,
    learning_rate=1e-5,
    tolerance=1e-6,
    max_epochs=1000,
    ind_run = True
):
    mse_history = []
    m, n = X_train.shape

    # Add bias
    X_train_aug = np.hstack([np.ones((m, 1)), X_train])
    w = np.random.randn(n + 1, 1)
    history = []

    num_batches = int(np.ceil(m / batch_size))

    for epoch in range(max_epochs):
        # Shuffle indices at each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_shuf = X_train_aug[indices]
        Y_shuf = Y_train[indices]

        for batch in range(num_batches):
            start = batch * batch_size
            end = min(start + batch_size, m)
            X_batch = X_shuf[start:end]
            Y_batch = Y_shuf[start:end]

            # Predictions and gradient for mini-batch
            Y_hat_batch = X_batch @ w
            grad = (2 / X_batch.shape[0]) * (X_batch.T @ (Y_hat_batch - Y_batch))
            w = w - learning_rate * grad

        # Compute full training MSE after each epoch
        Y_hat = X_train_aug @ w
        mse = np.mean((Y_train - Y_hat) ** 2)
        history.append(mse)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mse}")

        # Optional: stop if last gradient very small (converged)
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {epoch} epochs.")
            break

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())

    if ind_run:
        largest_idx = np.argmax(np.abs(w[1:].ravel()))

        # Get actual feature name from DataFrame columns
        feature_name = X_train.columns[largest_idx]

        # Extract feature and target values as numpy arrays
        feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
        target_values = Y_train.ravel()

        # Plot scatter of feature vs target
        plt.scatter(feature_values, target_values, label='Data points')

        # Plot regression line using learned weights
        regression_line = w[0, 0] + w[largest_idx + 1, 0] * feature_values
        plt.plot(feature_values, regression_line, color='red', label='Regression line')

        plt.xlabel(feature_name)
        plt.ylabel('Quality')
        plt.title('Mini-Batch Gradient Descent (Without Regularization)')
        plt.legend()
        plt.show()

    return w, mse_history


# L2 Batch
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
    X_new_np = np.delete(X_train, min_idx, axis=1)
    X_new = pd.DataFrame(X_new_np)
    
    # Retrain using original batch gradient (without regularization)
    w_retrained, hist = batch_without_reg(X_new, Y_train, learning_rate, tolerance, max_iters, False)

    largest_idx = np.argmax(np.abs(w_retrained[1:].ravel()))

   # Get actual feature name from DataFrame columns
    feature_name = X_train.columns[largest_idx]

    # Extract feature and target values as numpy arrays
    feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
    target_values = Y_train.ravel()

    # Plot scatter of feature vs target
    plt.scatter(feature_values, target_values, label='Data points')

    # Plot regression line using learned weights
    regression_line = w_retrained[0, 0] + w_retrained[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color='red', label='Regression line')

    plt.xlabel(feature_name)
    plt.ylabel('Quality')
    plt.title('L2 Regularization (Batch)')
    plt.legend()
    plt.show()

    return w_retrained, hist, min_idx


# L2 Mini-Batch
def mini_batch_with_l2_reg(
    X_train,
    Y_train,
    alpha=0.1,
    learning_rate=1e-5,
    tolerance=1e-6,
    max_epochs=100,
    batch_size=32
):
    m, n = X_train.shape
    X_aug = np.hstack([np.ones((m, 1)), X_train])
    w = np.random.randn(n + 1, 1)
    mse_history = []

    for epoch in range(max_epochs):
        # Shuffle data at the start of epoch
        perm = np.random.permutation(m)
        X_aug_shuffled = X_aug[perm]
        Y_train_shuffled = Y_train[perm]

        for i in range(0, m, batch_size):
            X_batch = X_aug_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]

            # Compute prediction and loss on mini-batch
            Y_hat = X_batch @ w
            mse = np.mean((Y_batch - Y_hat) ** 2)
            mse_history.append(mse)

            # Gradient calculation on mini-batch
            batch_m = X_batch.shape[0]
            grad_mse = (2 / batch_m) * (X_batch.T @ (Y_hat - Y_batch))
            grad_l2 = 2 * alpha * np.vstack([[0], w[1:]])
            grad = grad_mse + grad_l2

            # Update weights
            w = w - learning_rate * grad

        # Termination criteria after full epoch
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {epoch} epochs.")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Latest MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())

    return w, mse_history

def retrain_without_smallest_weight_mini_batch(
    X_train, Y_train, w,
    learning_rate=1e-5, tolerance=1e-6, max_epochs=100, batch_size=32, verbose=True
):
    abs_w = np.abs(w[1:].ravel())
    min_idx = np.argmin(abs_w)
    print(f"Removing feature index {min_idx} with weight {w[min_idx+1,0]}")

    X_new_np = np.delete(X_train, min_idx, axis=1)
    X_new = pd.DataFrame(X_new_np)

    # Replace batch_without_reg with mini_batch_with_l2_reg or your mini-batch version
    w_retrained, hist = mini_batch_with_l2_reg(X_new.values, Y_train, learning_rate=learning_rate, tolerance=tolerance, max_epochs=max_epochs, batch_size=batch_size)

    largest_idx = np.argmax(np.abs(w_retrained[1:].ravel()))

   # Get actual feature name from DataFrame columns
    feature_name = X_train.columns[largest_idx]

    # Extract feature and target values as numpy arrays
    feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
    target_values = Y_train.ravel()

    # Plot scatter of feature vs target
    plt.scatter(feature_values, target_values, label='Data points')

    # Plot regression line using learned weights
    regression_line = w_retrained[0, 0] + w_retrained[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color='red', label='Regression line')

    plt.xlabel(feature_name)
    plt.ylabel('Quality')
    plt.title('L2 Regularization (Mini-Batch)')
    plt.legend()
    plt.show()

    return w_retrained, hist, min_idx

# L1 Batch ***NEEDS FIXING***
def batch_with_l1_reg(
    X_train,
    Y_train,
    alpha=0.5,
    learning_rate=1e-5,
    tolerance=1e-7,
    max_iters=10000,
):
    m, n = X_train.shape
    X_aug = np.hstack([np.ones((m, 1)), X_train])
    w = np.random.randn(n + 1, 1)
    mse_history = []

    for iteration in range(max_iters):
        Y_hat = X_aug @ w
        mse = np.mean((Y_train - Y_hat) ** 2)
        mse_history.append(mse)

        grad_mse = (2 / m) * (X_aug.T @ (Y_hat - Y_train))
        grad_l1 = alpha * np.vstack([[0], np.sign(w[1:])])  # no reg on bias
        grad = grad_mse + grad_l1

        w = w - learning_rate * grad

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())

    zero_weight_indices = np.where(np.isclose(w[1:].ravel(), 0, atol=1e-2))[0]
    if zero_weight_indices.size > 0:
        print(f"Features with ~zero weights to eliminate: {zero_weight_indices}")
    else:
        print("No features with zero weights found (within tolerance).")

    # Always plot
    largest_idx = np.argmax(np.abs(w[1:].ravel()))
    feature_name = X_train.columns[largest_idx]
    feature_values = X_train.iloc[:, largest_idx].to_numpy()
    target_values = Y_train.ravel()

    plt.scatter(feature_values, target_values, label="Data points")
    regression_line = w[0, 0] + w[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color="red", label="Regression line")
    plt.xlabel(feature_name)
    plt.ylabel("Quality")
    plt.title("L1 Regularization (Batch)")
    plt.legend()
    plt.show()

    return w, mse_history, zero_weight_indices


def retrain_without_zero_weights(
    X_train, Y_train, zero_weight_indices,
    learning_rate=1e-5, tolerance=1e-6, max_iters=10000
):
    # Remove columns corresponding to zero-weight features
    X_reduced_np = np.delete(X_train, zero_weight_indices, axis=1)
    X_reduced = pd.DataFrame(X_reduced_np)
    
    # Retrain using original batch gradient
    w_retrained, hist = batch_without_reg(X_reduced, Y_train, learning_rate, tolerance, max_iters)
    
    largest_idx = np.argmax(np.abs(w_retrained[1:].ravel()))

   # Get actual feature name from DataFrame columns
    feature_name = X_train.columns[largest_idx]

    # Extract feature and target values as numpy arrays
    feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
    target_values = Y_train.ravel()

    # Plot scatter of feature vs target
    plt.scatter(feature_values, target_values, label='Data points')

    # Plot regression line using learned weights
    regression_line = w_retrained[0, 0] + w_retrained[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color='red', label='Regression line')

    plt.xlabel(feature_name)
    plt.ylabel('Quality')
    plt.title('L1 Regularization (Batch)')
    plt.legend()
    plt.show()

    return w_retrained, hist, zero_weight_indices

# L1 Mini-Batch 
def mini_batch_with_l1_reg(
    X_train,
    Y_train,
    alpha=0.1,            # L1 regularization strength
    learning_rate=1e-5,
    tolerance=1e-6,
    max_epochs=100,
    batch_size=32,
):
    m, n = X_train.shape
    X_aug = np.hstack([np.ones((m, 1)), X_train])  # add bias
    w = np.random.randn(n + 1, 1)
    mse_history = []

    for epoch in range(max_epochs):
        perm = np.random.permutation(m)
        X_aug_shuffled = X_aug[perm]
        Y_train_shuffled = Y_train[perm]

        for i in range(0, m, batch_size):
            X_batch = X_aug_shuffled[i:i + batch_size]
            Y_batch = Y_train_shuffled[i:i + batch_size]

            Y_hat = X_batch @ w
            mse = np.mean((Y_batch - Y_hat) ** 2)
            mse_history.append(mse)

            batch_m = X_batch.shape[0]   # Gradient of MSE
            grad_mse = (2 / batch_m) * (X_batch.T @ (Y_hat - Y_batch))
            grad_l1 = alpha * np.vstack([[0], np.sign(w[1:])])  # no reg on bias

            # Subgradient for L1 (skip bias term)
            grad_l1 = np.vstack([np.array([[0]]), alpha * np.sign(w[1:])])

            # Update weights
            w = w - learning_rate * (grad_mse + grad_l1)

        if np.linalg.norm(grad) < tolerance:
            print(f"Converged after {epoch} epochs.")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Latest MSE: {mse}")

    print("Final training MSE:", mse)
    print("Learned weight vector:", w.ravel())

    zero_weight_indices = np.where(np.isclose(w[1:].ravel(), 0, atol=1e-2))[0]
    if zero_weight_indices.size > 0:
        print(f"Features with ~zero weights to eliminate: {zero_weight_indices}")
    else:
        print("No features with zero weights found (within tolerance).")

    # Always plot
    largest_idx = np.argmax(np.abs(w[1:].ravel()))
    feature_name = X_train.columns[largest_idx]
    feature_values = X_train.iloc[:, largest_idx].to_numpy()
    target_values = Y_train.ravel()

    plt.scatter(feature_values, target_values, label="Data points")
    regression_line = w[0, 0] + w[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color="red", label="Regression line")
    plt.xlabel(feature_name)
    plt.ylabel("Quality")
    plt.title("L1 Regularization (Mini-Batch)")
    plt.legend()
    plt.show()

    return w, mse_history, zero_weight_indices


def retrain_without_zero_weights_mini_batch(
    X_train, Y_train, zero_weight_indices,
    learning_rate=1e-5, tolerance=1e-7, max_epochs=100, batch_size=32
):
    X_reduced_np = np.delete(X_train, zero_weight_indices, axis=1)
    X_reduced = pd.DataFrame(X_reduced_np)

    w_retrained, hist, _ = mini_batch_with_l1_reg(
        X_reduced.values, Y_train,
        learning_rate=learning_rate,
        tolerance=tolerance,
        max_epochs=max_epochs,
        batch_size=batch_size
    )

    largest_idx = np.argmax(np.abs(w_retrained[1:].ravel()))
    feature_name = X_train.columns[largest_idx]

    feature_values = X_train.iloc[:, largest_idx].to_numpy() if hasattr(X_train, "iloc") else X_train[:, largest_idx]
    target_values = Y_train.ravel()

    plt.scatter(feature_values, target_values, label='Data points')
    regression_line = w_retrained[0, 0] + w_retrained[largest_idx + 1, 0] * feature_values
    plt.plot(feature_values, regression_line, color='red', label='Regression line')

    plt.xlabel(feature_name)
    plt.ylabel('Quality')
    plt.title('L1 Regularization (Mini-Batch)')
    plt.legend()
    plt.show()

    return w_retrained, hist, zero_weight_indices


# Fetch dataset
wine_quality = fetch_ucirepo(id=186)

# Data
X = wine_quality.data.features
y = wine_quality.data.targets

data = X.copy()
data['quality'] = y

# INSPECT/ANALYZE DATA SET

# Shuffle data in place
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Set percentages, rest is testing
n = len(data)
train_end = int(n * 0.6) # 60% for training
valid_end = train_end + int(n * 0.2) # 20% for validation

# Split into training, validation, testing
X_train = data[:train_end]
X_valid = data[train_end:valid_end]
X_test = data[valid_end:]

# Ground truths
Y_train = X_train['quality']
Y_valid = X_valid['quality']
Y_test = X_test['quality']

if Y_train.ndim == 1:
    Y_train = Y_train.values.reshape(-1, 1)


# MODEL EXECUTION (uncomment sections as necessary to run specific regression models): 

print("\nBATCH")
batch_without_reg(X_train, Y_train)
print("\nBATCH")
batch_without_reg(X_train, Y_train)

print("\nMINI-BATCH")
mini_batch_without_reg(X_train, Y_train)
print("\nMINI-BATCH")
mini_batch_without_reg(X_train, Y_train)

print("\nL2 BATCH")
w_l2, mse_hist = batch_with_l2_reg(X_train, Y_train, alpha=0.05)
w_retrained, mse_hist_retrained, dropped_col = retrain_without_smallest_weight(X_train, Y_train, w_l2)
print("\nL2 BATCH")
w_l2, mse_hist = batch_with_l2_reg(X_train, Y_train, alpha=0.05)
w_retrained, mse_hist_retrained, dropped_col = retrain_without_smallest_weight(X_train, Y_train, w_l2)

print("\nL2 MINI-BATCH")
w_l2, mse_hist = mini_batch_with_l2_reg(X_train, Y_train, alpha=0.05)
w_retrained, mse_hist_retrained, dropped_col = retrain_without_smallest_weight_mini_batch(X_train, Y_train, w_l2)
print("\nL2 MINI-BATCH")
w_l2, mse_hist = mini_batch_with_l2_reg(X_train, Y_train, alpha=0.05)
w_retrained, mse_hist_retrained, dropped_col = retrain_without_smallest_weight_mini_batch(X_train, Y_train, w_l2)


print("\nL1 BATCH")
w_l1, mse_hist_l1, zeros = batch_with_l1_reg(X_train, Y_train, alpha=0.08)
if zeros.size > 0:
    w_retrained, mse_retrained, dropped = retrain_without_zero_weights(X_train, Y_train, zeros)

#-----------------------------------------------------------------------------------------------------------------------------
# ***TO-DO ITEM***: L1 model not able to find 0 weights in features, thus cannot plot, could be an issue with the equations/math
# -----------------------------------------------------------------------------------------------------------------------------
print("\nL1 MINI-BATCH")
w_l1, mse_hist_l1, zeros = mini_batch_with_l1_reg(X_train, Y_train, alpha=0.08)
if zeros.size > 0:
    w_retrained, mse_retrained, dropped = retrain_without_zero_weights_mini_batch(X_train, Y_train, zeros)






