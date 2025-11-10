import numpy as np
import matplotlib.pyplot as plt
import time

batch = False
incremental = False
while True:
    batch_or_incremental = input("Enter 'bat' for batch delta rule, or 'incr' for incremental delta rule: ")
    if(batch_or_incremental == 'bat'):
        batch = True
        break
    elif(batch_or_incremental == 'incr'):
        incremental = True
        break
    else:
        print("INVALID INPUT")
    
# Parameters
learning_rate = 0.1
num_epochs = 100
n_samples = 200

# Sigmoid function and its derivative
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_prime(t):
    s = sigmoid(t)
    return s * (1 - s)

# Generate random samples in range [-5, 5]
x = np.random.uniform(-5, 5, size=(n_samples, 2))
x_1 = x[:,0]
x_2 = x[:,1]

# Assign targets using the concept: 1 if x_1 + 3 * x_2 - 2 > 0, else 0
targets = np.where(x_1 + 3 * x_2 - 2 > 0, 1, 0)

# Initialize weights and bias
w = np.random.uniform(-0.5, 0.5, size=(2,))
bias = np.random.uniform(-0.5, 0.5)

# Decision surface epoch iterations
plot_epochs = [5, 10, 50, 100]

# Store epoch errors
epoch_errors = []

# Track updates and start timer
updates = 0
start_time = time.time()

# Delta rule training
for epoch in range(1, num_epochs + 1):
    if (incremental): # INCREMENTAL TRAINING
        total_error = 0
        for i in range(n_samples):
            # Get sample input and target
            inp = x[i]
            target = targets[i]
            
            # (1) Calculate net input (w dot input + bias)
            # (2) Calculate predicted output layer via sigmoid (1 / 1 + e^-t)
            # (3) Calculate error (diff between target and prediction)
            # (4) Calculate gradient using error and derived sigmoid function 
            net = np.dot(w, inp) + bias
            out = sigmoid(net)
            error = target - out
            grad = error * sigmoid_prime(net)
            
            # Update weight per sample using learning rate, gradient, and sample input
            w += learning_rate * grad * inp
            # Update bias with learning rate and gradient
            bias += learning_rate * grad
            
            # Accumulate squared error for MSE at end of epoch
            total_error += (error**2)
            updates += 1
        
        # Average error per epoch
        epoch_errors.append(total_error / n_samples)

    elif (batch): # BATCH TRAINING
        total_dw = np.zeros_like(w) # Set array of zeroes matching shape and size of w
        total_db = 0
        epoch_error = 0

        for i in range(n_samples): # Update each weight per epoch by calculating running sum of weight and bias gradient, then update weight per epoch
            inp = x[i]
            target = targets[i]
            
            net = np.dot(w, inp) + bias
            out = sigmoid(net)
            error = target - out
            grad = error * sigmoid_prime(net)
            
            total_dw += grad * inp  # sum up weight gradient
            total_db += grad        # sum up bias gradient
            
            epoch_error += error**2
        
        # Update weights and bias per epoch
        w += learning_rate * total_dw
        bias += learning_rate * total_db
        epoch_errors.append(epoch_error / n_samples)
        updates += 1
    
    # End timer
    end_time = time.time()
    exec_time = end_time - start_time
    
    # Plot decision surface at epochs [5, 10, 50, 100]
    if epoch in plot_epochs:
        plt.figure()
        
        # Separate positive and negative points
        pos = x[targets == 1]
        neg = x[targets == 0]
        plt.scatter(pos[:, 0], pos[:, 1], color='blue', label='Class 1')
        plt.scatter(neg[:, 0], neg[:, 1], color='red', label='Class 0')

        # Plot the decision line (w1*x1 + w2*x2 + b = 0)
        x1_line = np.linspace(-5, 5, 100)

        # Avoid division by zero
        if w[1] != 0:
            x2_line = -(w[0] * x1_line + bias) / w[1]
            plt.plot(x1_line, x2_line, 'k--', label='Decision boundary')

        plt.title(f"Decision Boundary after {epoch} Epochs")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.legend()
        plt.grid(True)


# Plot error vs epochs
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), epoch_errors, marker='o')
plt.title("Error vs Epochs (Delta Rule Training)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (E)")
plt.grid(True)
plt.show()

# Evaluate training accuracy
outputs = sigmoid(np.dot(x, w) + bias)
pred_labels = (outputs > 0.5).astype(int)
accuracy = np.mean(pred_labels == targets)

print(f"Learned weights: {w}")
print(f"Learned bias: {bias}")
print(f"Training accuracy: {accuracy:.2f}")

# Show first 10 predictions and their targets
for i in range(10):
    print(f"x: {x[i]}, Predicted: {pred_labels[i]}, Target: {targets[i]}")

# Compare delta rule batch training vs incremental training with execution time, weight updates, 
print(f"Execution time: {exec_time:.4f} seconds")
print(f"Number of weight updates: {updates}")

# ----------------------------------------------------------
# PROBLEM 2: VARIABLE LEARNING RATES (DECAYING + ADAPTIVE)
# ----------------------------------------------------------

# Reuse the same dataset (x, targets) from Problem 1

# Common parameters
num_epochs = 100
initial_lr = 0.3
decay_factor = 0.8
t = 0.03
d = 0.9
D = 1.02

# Initialize new weights and bias
w_decay = np.random.uniform(-0.5, 0.5, size=(2,))
b_decay = np.random.uniform(-0.5, 0.5)

w_adapt = np.copy(w_decay)
b_adapt = b_decay

# --------------------------
# (a) DECAYING LEARNING RATE
# --------------------------
lr = initial_lr
decay_errors = []

for epoch in range(1, num_epochs + 1):
    total_dw = np.zeros_like(w_decay)
    total_db = 0
    epoch_error = 0

    for i in range(n_samples):
        inp = x[i]
        target = targets[i]
        net = np.dot(w_decay, inp) + b_decay
        out = sigmoid(net)
        error = target - out
        grad = error * sigmoid_prime(net)

        total_dw += grad * inp
        total_db += grad
        epoch_error += error**2

    w_decay += lr * total_dw
    b_decay += lr * total_db
    decay_errors.append(epoch_error / n_samples)
    print(f"Epoch {epoch:3d} | η = {lr:.5f} | Mean Error = {decay_errors[-1]:.5f}")


    lr *= decay_factor  # multiply learning rate by 0.8 each epoch

print("\n--- Decaying Learning Rate Training Complete ---")

# Plot decaying learning rate error curve
plt.figure()
plt.plot(range(1, num_epochs + 1), decay_errors, label="Decaying η")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (E)")
plt.title("Error vs Epochs (Decaying Learning Rate)")
plt.grid(True)
plt.legend()
plt.show()

# --------------------------
# (b) ADAPTIVE LEARNING RATE
# --------------------------
eta = 0.3
prev_error = float('inf')
adapt_errors = []
eta_values = []

for epoch in range(1, num_epochs + 1):
    total_dw = np.zeros_like(w_adapt)
    total_db = 0
    epoch_error = 0

    for i in range(n_samples):
        inp = x[i]
        target = targets[i]
        net = np.dot(w_adapt, inp) + b_adapt
        out = sigmoid(net)
        error = target - out
        grad = error * sigmoid_prime(net)

        total_dw += grad * inp
        total_db += grad
        epoch_error += error**2

    w_adapt += eta * total_dw
    b_adapt += eta * total_db
    mean_error = epoch_error / n_samples
    adapt_errors.append(mean_error)
    eta_values.append(eta)
    print(f"Epoch {epoch:3d} | η = {eta:.5f} | Mean Error = {mean_error:.5f}")


    # Adaptive learning rule
    if mean_error > prev_error * (1 + t):
        eta *= d   # decrease
    elif mean_error < prev_error:
        eta *= D   # increase
    prev_error = mean_error

print("\n--- Adaptive Learning Rate Training Complete ---")

# Plot adaptive learning rate error curve
plt.figure()
plt.plot(range(1, num_epochs + 1), adapt_errors, label="Adaptive η")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (E)")
plt.title("Error vs Epochs (Adaptive Learning Rate)")
plt.grid(True)
plt.legend()
plt.show()

# visualize how η changes over epochs
plt.figure()
plt.plot(range(1, num_epochs + 1), eta_values, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate (η)")
plt.title("Adaptive Learning Rate Changes Over Time")
plt.grid(True)
plt.show()
