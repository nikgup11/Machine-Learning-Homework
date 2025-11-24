import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(filename):
    # Loads [feature1, feature2, label] where label is 0 or 1
    data = np.loadtxt(filename)
    X = data[:, :2]
    y_raw = data[:, 2]
    
    # SVM expects labels as -1 and 1
    y = np.where(y_raw == 0, -1, 1)
    return X, y

def init_alphas(y, seed=None):
    m = len(y)
    rng = np.random.default_rng(seed)
    
    # random values >= 0
    alphas = rng.uniform(low=0, high=1, size=m) 
    
    # Project onto constraint: summation y_i * alpha_i = 0
    y = y.astype(np.float64)
    correction = np.dot(y, alphas) / np.dot(y, y)
    alphas_corrected = alphas - y * correction
    
    # Ensure values are positive after correction
    alphas_corrected = np.clip(alphas_corrected, 0, None)
    
    # If there is still a tiny residual - project again
    correction2 = np.dot(y, alphas_corrected) / np.dot(y, y)
    alphas_corrected -= y * correction2
    alphas_corrected = np.clip(alphas_corrected, 0, None)
    return alphas_corrected


def compute_weight_vector(alphas, y, X):
    # Ensure all arrays are numpy arrays
    alphas = np.asarray(alphas)
    y = np.asarray(y)
    X = np.asarray(X)
    
    # Compute w as the weighted sum over all samples
    w = np.sum((alphas * y)[:, np.newaxis] * X, axis=0)
    return w

def kkt_conditions(alphas, y, X, w, b):
    # Compute y_i (w dot x_i + b) - 1 for all i
    margins = y * ((X @ w) + b) - 1
    
    # Element-wise product with corresponding alpha_i
    kkt_vals = alphas * margins
    return kkt_vals

def compute_errors(alphas, y, X, b, kernel=lambda x1, x2: np.dot(x1, x2)):
    # Number of samples
    m = X.shape[0]
    # Init error array
    E = np.zeros(m)
    
    # Compute kernel values between all j, i, then error for sample i 
    for i in range(m):
        Kji = np.array([kernel(X[j], X[i]) for j in range(m)])
        E[i] = np.sum(alphas * y * Kji) + b - y[i]
    return E

def calc_ei(alphas, y, X, i_1):
    # Number of samples
    m = X.shape[0]
    # Compute K_j1 and K_ji for all j, i_1
    K_j1 = np.array([np.dot(X[j], X[i_1]) for j in range(m)])  # shape (m,)
    
    # init e values array
    e_values = []
    
    for i in range(m):
        # For all training samples compute kernel for all j, i
        K_ji = np.array([np.dot(X[j], X[i]) for j in range(m)])
        
        # summation over j
        sum_term = np.sum(alphas * y * (K_j1 - K_ji))
        
        # Calculate temporary error difference for i then store in e_values
        e_i = sum_term + y[i] - y[i_1]
        e_values.append(e_i)
    return np.array(e_values)

def quadratic_kernel(x1, x2):
    return (np.dot(x1, x2) + 1) ** 2

# Load data
X, y = load_data("SMO_Data-1-1.txt")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

max_iters = 200     # max number of SMO iterations
tolerance = 1e-5     # minimum alpha change threshold to declare convergence
b = 0
alphas = init_alphas(y)

# Prompt user for kernel type
kernel_choice = input("Type in 'lin' for linear kernel or 'quad' for quadratic kernel: ")
if kernel_choice not in {'lin', 'quad'}:
    print("Invalid choice. Defaulting to linear kernel.")
    kernel_choice = 'lin'

# SMO Algorithm
for iteration in range(max_iters):
    # Store alphas original copy for checking convergence
    alpha_prev = alphas.copy()

    # Step 2 - weight vector
    w = compute_weight_vector(alphas, y, X)

    # Step 3 - KKT Conditions
    kkt_vals = kkt_conditions(alphas, y, X, w, b)
    if(kernel_choice == 'lin'): # Linear Kernel
        E = compute_errors(alphas, y, X, b)
    elif(kernel_choice == 'quad'): # Quadratic Kernel
        E = compute_errors(alphas, y, X, b, kernel=quadratic_kernel)
    
    # Step 4 - Pick x_1, x_2
    i_1 = np.argmax(kkt_vals)
    x_1 = X[i_1]
    e_vals = calc_ei(alphas, y, X, i_1)

    i_2 = np.argmax(e_vals)
    x_2 = X[i_2]

    # Calculate k = K_11 + K_22 − 2K_12
    k_i1i1 = np.dot(x_1, x_1)
    k_i2i2 = np.dot(x_2, x_2)
    k_i1i2 = np.dot(x_1, x_2)
    k = k_i1i1 + k_i2i2 - 2 * k_i1i2

    # Step 5: Update alpha_i2 (for index i_2)
    alpha_i2 = alphas[i_2] + (y[i_2] * e_vals[i_2]) / k
    alphas_updated = alphas.copy()
    alphas_updated[i_2] = alpha_i2

    # Step 6: update alpha_i1 (for index i_1)
    alpha_i1 = alphas[i_1] + y[i_1] * y[i_2] * (alphas[i_2] - alpha_i2)
    alphas_updated[i_1] = alpha_i1

    # Step 7: for i = 1, ..., m, if alpha_i < epsilon, set alpha_i = 0
    alphas_updated = np.where(alphas_updated < tolerance, 0, alphas_updated)

    # Step 8: Select alpha_i > 0, compute b
    support_vector_indices = np.where(alphas_updated > 0)[0]

    # NOTICE: This uses mean of values from all support vectors
    # So consider use only 1 if error happens
    if len(support_vector_indices) > 0:
        b_values = []
        for sv_index in support_vector_indices:
            b_sv = y[sv_index] - np.dot(w, X[sv_index])
            b_values.append(b_sv)
        b = np.mean(b_values)
        
    # Step 9: test for classification
    y_pred = np.sign(np.dot(X, w) + b)      # Predict labels for all training samples
    correctly_classified = np.all(y_pred == y)  # Check if all points classified correctly
    
    # Print progress/statistics
    acc = np.mean(y_pred == y)
    print(f"Iteration {iteration+1}: accuracy = {acc:.4f}, converged = {correctly_classified}")
    
    # Early stopping if all samples are correctly classified
    if correctly_classified:
        print(f"Converged after {iteration+1} iterations!")
        break

else:
    print("Max iterations reached. Classified")
