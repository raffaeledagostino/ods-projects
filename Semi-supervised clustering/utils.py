from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import heapq
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score
import time
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from sklearn.decomposition import PCA

def generate_points(n=1000, n_features=2, centers=2, plot=False, pairplot=False):
    X, y = make_blobs(n_samples=n, n_features=n_features, random_state=0, centers=centers)
    y = 2 * y - 1  # transform labels to -1/+1

    if plot:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X[:, 0], X[:, 1],
            c=y, cmap='bwr', edgecolor='k', s=60, alpha=0.8
        )
        plt.title("Generated Points", fontsize=16, pad=15)
        plt.xlabel("Feature 1", fontsize=14)
        plt.ylabel("Feature 2", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    if pairplot:
        # Create DataFrame for seaborn
        df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(n_features)])
        df['Label'] = y

        sns.pairplot(
            df,
            vars=df.columns[:-1],
            hue='Label',
            palette='bwr',
            plot_kws={'s': 30, 'edgecolor': 'k', 'alpha': 0.7}
        )
        plt.suptitle("Pairplot of Generated Features", fontsize=16, y=1.02)
        plt.show()

    return X, y


def download_bank(plot=False, pairplot=False, pca_analysis=True):
    # Fetch dataset
    bank = fetch_ucirepo(id=267)

    # Data (as pandas DataFrames)
    X = bank.data.features.copy()
    y = bank.data.targets.copy()

    y.columns = ['target']
    y['target'] = 2 * y['target'] - 1  # Transform to -1 / 1

    # Join X and y for plotting convenience
    df = pd.concat([X, y], axis=1)

    if pca_analysis:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        print("\n=== PCA Analysis ===")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"Component {i+1}: {var*100:.2f}% variance explained")

        features = X.columns
        for i, component in enumerate(pca.components_):
            print(f"\nTop features per PC{i+1}:")
            sorted_features = sorted(zip(features, component),
                                   key=lambda x: abs(x[1]),
                                   reverse=True)
            for feat, weight in sorted_features[:3]:
                print(f"- {feat}: {weight:.3f}")

    if plot:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X.iloc[:, 0], X.iloc[:, 1],
            c=y['target'], cmap='bwr', edgecolor='k', s=60, alpha=0.8
        )
        plt.title("First Two Features Colored by Target", fontsize=16, pad=15)
        plt.xlabel(X.columns[0], fontsize=14)
        plt.ylabel(X.columns[1], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    if pairplot:
        sns.set(style="whitegrid")
        g = sns.pairplot(
            df,
            vars=X.columns,  # Plot all features
            hue='target',
            palette='bwr',
            plot_kws={'s': 30, 'edgecolor': 'k', 'alpha': 0.7}
        )
        g.fig.suptitle("Pairplot of Features by Class", fontsize=16, y=1.02)
        plt.show()

    return X.to_numpy(), y['target'].to_numpy()


def gauss_kernel(X, Y, k=1):
  return np.exp(-k * (cdist(X, Y)**2))


def data_preparation(X, y, labeled_size=0.1, plot=False):
    # Split points into labeled/unlabeled
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, train_size=0.1, random_state=0)

    # Compute distances for unlabeled-unlabeled
    W_unlabeled = gauss_kernel(X_unlabeled, X_unlabeled)

    # Compute distances for labeled-unlabeled
    W_labeled = gauss_kernel(X_labeled, X_unlabeled)

    if plot:
        plt.figure(figsize=(8, 6))

        # Unlabeled data (light gray)
        plt.scatter(
            X_unlabeled[:, 0], X_unlabeled[:, 1],
            c='lightgrey', edgecolor='k', label='Unlabeled', s=40, alpha=0.5, marker='o'
        )

        # Labeled data (colored by class, with clear boundary)
        scatter = plt.scatter(
            X_labeled[:, 0], X_labeled[:, 1],
            c=y_labeled, cmap='bwr', edgecolor='k', label='Labeled', s=60, marker='o'
        )

        # Styling
        plt.title("Labeled and Unlabeled Data", fontsize=16, pad=15)
        plt.xlabel("Feature 1", fontsize=14)
        plt.ylabel("Feature 2", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(frameon=True, fontsize=12, loc='best')
        plt.tight_layout()

        plt.show()

    return X_labeled, X_unlabeled, y_labeled, y_unlabeled, W_labeled, W_unlabeled


def hessian_matrix(w_labeled_unlabeled, w_unlabeled_unlabeled, y_unlabeled):
    mat = np.copy(-w_unlabeled_unlabeled)
    for i in range(len(y_unlabeled)):
        mat[i][i] = np.sum(w_labeled_unlabeled[:,i]) + np.sum(w_unlabeled_unlabeled[:,i]) - w_unlabeled_unlabeled[i][i]
    return mat * 2

def estimate_lipschitz_constant(hessian):
    return eigh(hessian, subset_by_index=(len(hessian)-1, len(hessian)-1))[0][0]

def objective_function(y_labeled, W_labeled, W_unlabeled, y_unlabeled):

    def f(y):
      Y = np.copy(y).astype("float64").reshape((-1,1))
      Y_bar = np.copy(y_labeled).astype("float64").reshape((-1,1))

      Y_minus_Y_bar = Y-Y_bar.T
      Y_minus_Y_bar_sq = np.power(Y_minus_Y_bar, 2)
      labeled_unlabeled_loss_matrix = Y_minus_Y_bar_sq * W_labeled.T
      labeled_unlabeled_loss = np.sum(labeled_unlabeled_loss_matrix)

      Y_minus_Y = Y-Y.T
      Y_minus_Y_sq = np.power(Y_minus_Y, 2)
      unlabeled_unlabeled_loss_matrix = Y_minus_Y_sq * W_unlabeled.T
      unlabeled_unlabeled_loss = np.sum(unlabeled_unlabeled_loss_matrix)

      return labeled_unlabeled_loss + unlabeled_unlabeled_loss/2

    hessian = hessian_matrix(W_labeled, W_unlabeled, y_unlabeled)

    def grad_f(y):
      vec1 = np.zeros(len(y))
      for k in range(len(y)):
          vec1[k] = 2 * np.sum(W_labeled[:,k] * (y[k] - y_labeled))

      vec2 = np.zeros(len(y))
      for k in range(len(y)):
          vec2[k] = 2 * np.sum(W_unlabeled[k,:] * (y[k] - y))

      return vec1 + vec2

    def fast_updates_f(fy, gy, y, stepsize, idx):

        fy = fy - stepsize * gy[idx]**2 + 0.5 * stepsize**2 * gy[idx]**2 * hessian[idx, idx]

        gy = gy - stepsize * gy[idx] * hessian[:, idx]

        return fy, gy

    # Function for safe updates of the objective function
    def safe_updates_f(y):
        gy = grad_f(y)
        fy = f(y)
        return fy, gy

    return safe_updates_f, fast_updates_f, f, hessian


def GD_update(safe_updates_f, y0, y_unlabeled, hessian, num_iters=100, tol=1e-6):
    # Initialize the solution
    y = y0.copy().astype(float)
    n = len(y)

    # Initialize the gradient
    fy, grad = safe_updates_f(y)
    f_history = [fy]
    accuracy_history = [accuracy_score(y0, y_unlabeled)]
    time_history = [0]

    step_size = 1 / estimate_lipschitz_constant(hessian)

    total_time = 0.0  # Track cumulative time

    for i in tqdm(range(num_iters), desc=f"Running GD"):
        start_time = time.process_time()

        y -= step_size * grad

        fx, grad = safe_updates_f(y)
        f_history.append(fx)

        y_pred = np.sign(y)
        accuracy_history.append(accuracy_score(y_pred, y_unlabeled))

        iter_time = time.process_time() - start_time
        total_time += iter_time
        time_history.append(total_time)

        if abs(f_history[i+1] - f_history[i]) / abs(f_history[i]) < tol:
            print(f"\nConverged after {i} iterations")
            break

    return y, f_history, accuracy_history, time_history

class Heap:
    def __init__(self, gradient, f_priorities):
        # Initialize the heap with priorities calculated from the given function
        self.f_priorities = f_priorities
        self.heap = [(-f_priorities(val, idx), idx) for idx, val in enumerate(gradient)]
        heapq.heapify(self.heap)  # Convert the list into a heap structure
        # Update the heap structure with positive values for priorities
        self.heap = [(-vali[0], vali[1]) for vali in self.heap]
        # Dictionary containing the index of each element in the heap
        self.dict = {vali[1]: i for (i, vali) in enumerate(self.heap)}

    def get_max(self):
        # Return the element with the maximum priority (first element in the heap)
        return self.heap[0]

    def update_priority(self, idx, new_val):
        # Update the priority of an element at the given index with a new value
        new_priority = self.f_priorities(new_val, idx)
        i = self.dict[idx]  # Get the index of the element in the heap
        old_priority, _ = self.heap[i]  # Get the old priority of the element
        self.heap[i] = (new_priority, idx)  # Update the priority with the new value and index
        if new_priority > old_priority:
            # New priority is higher than the old one, swift up
            while i > 0:
                parent_i = (i - 1) // 2  # Calculate the index of the parent node
                if self.heap[parent_i] < self.heap[i]:
                    # Swap the current node with its parent if the priority is higher
                    heap_parent_i = self.heap[parent_i]
                    self.heap[parent_i] = self.heap[i]
                    self.heap[i] = heap_parent_i
                    # Update the dictionary with the new indices
                    self.dict[self.heap[parent_i][1]], self.dict[self.heap[i][1]] = parent_i, i
                    i = parent_i  # Move to the parent node
                else:
                    break  # Exit the loop if the heap property is satisfied
        else:
            # New priority is lower or equal to the old one, swift down
            while True:
                left_child_i = 2 * i + 1  # Calculate the index of the left child node
                right_child_i = 2 * i + 2  # Calculate the index of the right child node

                # Find the index of the child with the maximum priority
                max_child_i = i
                if left_child_i < len(self.heap) and self.heap[left_child_i] > self.heap[max_child_i]:
                    max_child_i = left_child_i
                if right_child_i < len(self.heap) and self.heap[right_child_i] > self.heap[max_child_i]:
                    max_child_i = right_child_i
                if max_child_i == i:
                    break  # Exit the loop if the heap property is satisfied

                # Swap the current node with the maximum child
                heap_max_child_i = self.heap[max_child_i]
                self.heap[max_child_i] =  self.heap[i]
                self.heap[i] = heap_max_child_i
                # Update the dictionary with the new indices
                self.dict[self.heap[max_child_i][1]], self.dict[self.heap[i][1]] = max_child_i, i
                i = max_child_i  # Move to the maximum child node

def BCGD_GS(fast_updates_f, safe_updates_f, y0, y_unlabeled, hessian, num_iters=20, tol=1e-6):
    """
    Block Coordinate Gradient Descent with Gauss-Southwell updates.
    """
    # Initialize the solution
    y = y0.copy().astype(float)
    n = len(y)

    def abs_val(val, idx): #function to pass to the heap to retrieve the max element of the gradient
      return abs(val)

    fy, grad = safe_updates_f(y)
    heap_gradient = Heap(grad, abs_val)

    f_history = [fy]
    accuracy_history = [accuracy_score(y0, y_unlabeled)]
    time_history = [0]

    step_size = 1 / estimate_lipschitz_constant(hessian)
    total_time = 0.0

    for i in tqdm(range(num_iters), desc=f"Running BCGD"):
        start_time = time.process_time()

        # Extract the feature with the largest gradient
        max_grad_val, idx = heap_gradient.get_max()

        # Perform a line search along the direction of the largest gradient
        y[idx] = y[idx] - step_size * grad[idx]

        # Update the function information
        fy, grad = fast_updates_f(fy, grad, y, step_size, idx)

        iter_time = time.process_time() - start_time
        total_time += iter_time
        time_history.append(total_time)

        #print(f'iter:{i}, norm:{np.linalg.norm(grad)}')

        f_history.append(fy)
        y_pred = np.sign(y)
        accuracy_history.append(accuracy_score(y_pred, y_unlabeled))

        # Push the updated gradient components to the heap
        heap_gradient.update_priority(idx, grad[idx])

        # Check the convergence criterion
        if abs(f_history[i+1] - f_history[i]) / abs(f_history[i]) < tol:
            print(f"\nConverged after {i} iterations")
            break

    return y, f_history, accuracy_history, time_history

def coord_min_jacobi(W_unlabeled, W_labeled, y_labeled, y, y_unlabeled, loss, num_iters=100, tol=1e-6):

    W_unlabeled_nodiag = W_unlabeled - np.identity(len(y))

    y_history = [y]
    accuracy_history = [accuracy_score(y, y_unlabeled)]
    time_history = [0]
    f_history = [loss(y)]

    total_time = 0.0

    for i in tqdm(range(num_iters), desc=f"Running Coordinate Jacobi"):
        start_time = time.process_time()

        num = W_labeled.T @ y_labeled + W_unlabeled_nodiag.T @ y
        den = np.sum(W_labeled, axis=0) + np.sum(W_unlabeled_nodiag, axis=0)

        y = num / den

        iter_time = time.process_time() - start_time
        total_time += iter_time
        time_history.append(total_time)

        y_pred = np.sign(y)
        accuracy_history.append(accuracy_score(y_pred, y_unlabeled))
        f_history.append(loss(y))
        y_history.append(y)

        if abs(f_history[i+1] - f_history[i]) / abs(f_history[i]) < tol:
            print(f"\nConverged after {i} iterations")
            break

    return y, f_history, accuracy_history, time_history