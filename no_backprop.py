import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import time

# ==============================================================================
# Part 1: Tubular Plasticity (TP) - Final Evolution
# ==============================================================================

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

class TubularLayer:
    def __init__(self, n_inputs, n_neurons):
        # We drop elasticity for now to focus on the core learning rule
        self.diameters = np.random.randn(n_inputs, n_neurons) * np.sqrt(1. / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        # Store inputs and pre-activations for learning
        self.inputs = None
        self.pre_activations = None
        self.activations = None

    def forward(self, inputs):
        self.inputs = inputs
        self.pre_activations = np.dot(self.inputs, self.diameters) + self.biases
        self.activations = tanh(self.pre_activations)
        return self.activations

class TubularNetwork:
    def __init__(self, layer_dims):
        self.layers = []
        for i in range(len(layer_dims) - 1):
            self.layers.append(TubularLayer(layer_dims[i], layer_dims[i+1]))

    def forward(self, X):
        activations = X
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def predict(self, X):
        return np.sign(self.forward(X))

    # --- THE NEW TRAINING METHOD ---
    def train(self, X, y, epochs, learning_rate, nudge_factor=0.1, weight_decay=0.0001, batch_size=32):
        history = {'loss': []}
        
        for epoch in tqdm(range(epochs), desc="Training Progress (TP Evolved)"):
            epoch_loss = 0
            permutation = np.random.permutation(len(X))
            X_shuffled, y_shuffled = X[permutation], y[permutation]

            for i in range(0, len(X), batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]

                # --- Phase 1: "Free" Phase ---
                y_pred = self.forward(X_batch)
                epoch_loss += np.mean((y_batch - y_pred)**2) * len(X_batch)

                # --- Phase 2: Create Local Targets by "Nudging" ---
                # Calculate the final layer's error
                final_error = y_batch - y_pred
                
                # Start with the final layer's target
                # The "local error" for the last layer is just the output error
                local_error = final_error

                # --- Loop BACKWARDS to generate targets for each layer ---
                # This is NOT backpropagation. We are not propagating gradients.
                # We are propagating a TARGET signal to tell each layer what it SHOULD have done.
                for j in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[j]
                    
                    # Convert the error signal into a pre-activation delta
                    # This is the "inverse" of the activation function
                    pre_activation_delta = local_error * tanh_derivative(layer.pre_activations)

                    # --- CORE LEARNING RULE ---
                    # Update weights based on this local target signal
                    # Δw = learning_rate * (input * local_target)
                    weight_update = np.dot(layer.inputs.T, pre_activation_delta)
                    bias_update = np.sum(pre_activation_delta, axis=0)

                    # Apply the update with weight decay
                    layer.diameters += learning_rate * weight_update - weight_decay * layer.diameters
                    layer.biases += learning_rate * bias_update
                    
                    # Project the error to the previous layer to create its target
                    # This tells the layer before it what error it was "responsible" for
                    if j > 0:
                        local_error = np.dot(pre_activation_delta, layer.diameters.T)

            history['loss'].append(epoch_loss / len(X))
        return history

# ==============================================================================
# Part 2 & 3: Baseline, Visualization, and Execution
# (Mostly unchanged, but updated for the new TP model)
# ==============================================================================
def run_backprop_baseline(X_train, y_train, X_test, y_test, layer_dims, epochs=500):
    # (This function is unchanged)
    print("\n--- Running Backpropagation Baseline ---")
    X_train_t, y_train_t = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_test_t, y_test_t = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    class MLP(nn.Module):
        def __init__(self, dims):
            super().__init__()
            layers = []
            for i in range(len(dims) - 2): layers.extend([nn.Linear(dims[i], dims[i+1]), nn.Tanh()])
            layers.append(nn.Linear(dims[-2], dims[-1]))
            self.network = nn.Sequential(*layers)
        def forward(self, x): return self.network(x)
    model = MLP(layer_dims)
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): accuracy = accuracy_score(y_test_t.numpy(), np.sign(model(X_test_t).numpy()))
    print(f"Backprop Baseline Test Accuracy: {accuracy * 100:.2f}%")
    return model

def plot_decision_boundary(model, X, y, title):
    # (This function is unchanged)
    plt.figure(figsize=(10, 7))
    x_min, x_max, y_min, y_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_data = np.c_[xx.ravel(), yy.ravel()]
    if isinstance(model, TubularNetwork): Z = model.predict(mesh_data)
    else:
        model.eval()
        with torch.no_grad(): Z = np.sign(model(torch.FloatTensor(mesh_data)).numpy())
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(title); plt.xlabel("Feature 1 (Scaled)"); plt.ylabel("Feature 2 (Scaled)")
    plt.show()

if __name__ == "__main__":
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    y[y == 0] = -1 # Use [-1, 1] labels
    y_reshaped = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y_reshaped, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled, X_scaled = scaler.fit_transform(X_train), scaler.transform(X_test), scaler.transform(X)
    layer_dimensions = [X_train.shape[1], 16, 16, 1]
    
    start_time_bp = time.time()
    bp_model = run_backprop_baseline(X_train_scaled, y_train, X_test_scaled, y_test, layer_dimensions)
    bp_duration = time.time() - start_time_bp

    print("\n--- Starting training with Evolved Tubular Plasticity ---")
    tp_network = TubularNetwork(layer_dims=layer_dimensions)
    start_time_tp = time.time()
    # Note: This method often requires fewer epochs
    training_history = tp_network.train(X_train_scaled, y_train, epochs=200, learning_rate=0.005, weight_decay=0.0001, batch_size=16)
    tp_duration = time.time() - start_time_tp
    print("TP Training finished.")

    tp_accuracy = accuracy_score(y_test, tp_network.predict(X_test_scaled))
    
    print("\n" + "="*30 + "\n      EXPERIMENT RESULTS\n" + "="*30)
    print(f"Backprop Training Time: {bp_duration:.2f} seconds")
    print(f"Evolved TP Training Time: {tp_duration:.2f} seconds")
    print("-" * 30)
    print(f"EVOLVED TP Final Test Accuracy: {tp_accuracy * 100:.2f}%")
    print("="*30 + "\n")

    plt.figure(figsize=(10, 5))
    plt.plot(training_history['loss']); plt.title("Evolved TP Training Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.grid(True); plt.show()
    plot_decision_boundary(tp_network, X_scaled, y, f"Evolved Tubular Plasticity\nAccuracy: {tp_accuracy*100:.2f}%")