from dataclasses import dataclass, field
import numpy as np
import json

@dataclass
class LogisticRegression:
    _mean: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    _std: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    _is_fitted: bool = field(default=False, init=False)
    _weights: dict = field(default_factory=dict, init=False)
    _classes: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    
    # Hyperparameters
    learning_rate: float = field(init=False)
    max_iterations: int = field(init=False)

    def __post_init__(self):
        """Load hyperparameters from config file"""
        with open('hyperparams.json', 'r') as f:
            hyperparams = json.load(f)
        
        self.learning_rate = hyperparams['learning_rate']
        self.max_iterations = hyperparams['max_iterations']

    def fit_normalize(self, X):
        """Learn normalization stats from training data ONCE"""
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        self._std = np.where(self._std == 0, 1, self._std)
        self._is_fitted = True
        return (X - self._mean) / self._std

    def transform(self, X):
        """Apply SAME normalization to any data"""
        if not self._is_fitted:
            raise ValueError("Must call fit_normalize first!")
        return (X - self._mean) / self._std

    def _sigmoid(self, z):
        """Sigmoid activation function with overflow protection"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _fit_binary_classifier(self, X, y, class_name):
        """Fit binary classifier for one class vs all others"""
        # Add bias term (intercept)
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Initialize weights randomly
        np.random.seed(42)
        weights = np.random.normal(0, 0.01, X_with_bias.shape[1])
        
        print(f"Training {class_name}...")
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X_with_bias @ weights
            predictions = self._sigmoid(z)
            
            # Calculate cost (log-loss)
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Calculate gradients
            gradient = X_with_bias.T @ (predictions - y) / len(y)
            
            # Update weights
            new_weights = weights - self.learning_rate * gradient
            
            # Check for convergence
            tolerance = 1e-6
            if np.linalg.norm(new_weights - weights) < tolerance:
                print(f"  Converged after {iteration + 1} iterations")
                break
                
            weights = new_weights
            
            # Print progress every 200 iterations
            if iteration % 200 == 0:
                print(f"  Iteration {iteration}, Cost: {cost:.6f}")
        
        return weights

    def fit(self, X, y):
        """Fit multiclass logistic regression using One-vs-Rest approach"""
        if not self._is_fitted:
            raise ValueError("Must call fit_normalize first!")
        
        # Get unique classes
        self._classes = np.unique(y)
        print(f"Training classifiers for: {list(self._classes)}")
        
        # Train one binary classifier for each class
        for class_name in self._classes:
            # Create binary target (1 for current class, 0 for others)
            y_binary = (y == class_name).astype(int)
            
            # Train binary classifier
            weights = self._fit_binary_classifier(X, y_binary, class_name)
            self._weights[class_name] = weights
        
        print("Training completed!")

    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self._weights:
            raise ValueError("Must call fit first!")
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Get probabilities for each class
        probabilities = {}
        for class_name in self._classes:
            z = X_with_bias @ self._weights[class_name]
            probabilities[class_name] = self._sigmoid(z)
        
        return probabilities

    def predict(self, X):
        """Predict classes using One-vs-Rest strategy"""
        probabilities = self.predict_proba(X)
        
        # Convert to array format for easier manipulation
        prob_array = np.column_stack([probabilities[class_name] for class_name in self._classes])
        
        # Get class with highest probability for each sample
        predicted_indices = np.argmax(prob_array, axis=1)
        predictions = self._classes[predicted_indices]
        
        return predictions

    def save_model(self):
        pass

