from dataclasses import dataclass, field
import numpy as np
import json
from core.operations import mean, std

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
        # Calculate mean for each column using custom mean function
        self._mean = np.array([mean(X[:, i]) for i in range(X.shape[1])])
        # Calculate std for each column using custom std function
        self._std = np.array([std(X[:, i]) for i in range(X.shape[1])])
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

    def _fit_binary_classifier(self, X, y, random_state=42):
        """Fit binary classifier for one class vs all others"""
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Set seed for reproducible weight initialization
        np.random.seed(random_state)
        weights = np.random.normal(0, 0.01, X_with_bias.shape[1]) # preventing form symmetry issues
        
        for _ in range(self.max_iterations):
            z = X_with_bias @ weights
            predictions = self._sigmoid(z)

            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)

            gradient = X_with_bias.T @ (predictions - y) / len(y)
            weights -= self.learning_rate * gradient
            # cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            # if _ % 200 == 0:
            #     print(f"  Iteration {_}, Cost: {cost:.6f}")
        
        return weights

    def fit(self, X, y):
        """Fit multiclass logistic regression using One-vs-Rest approach"""
        if not self._is_fitted:
            raise ValueError("Must call fit_normalize first")
        
        self._classes = np.unique(y)

        for class_name in self._classes:
            y_binary = (y == class_name).astype(int)

            weights = self._fit_binary_classifier(X, y_binary, random_state=42)
            self._weights[class_name] = weights


    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self._weights:
            raise ValueError("Must call fit first!")
        
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        probabilities = {}
        for class_name in self._classes:
            z = X_with_bias @ self._weights[class_name]
            probabilities[class_name] = self._sigmoid(z)
        
        return probabilities

    def predict(self, X):
        """Predict classes using One-vs-Rest strategy"""
        probabilities = self.predict_proba(X)

        prob_array = np.column_stack([probabilities[class_name] for class_name in self._classes])
        
        predicted_indices = np.argmax(prob_array, axis=1)
        predictions = self._classes[predicted_indices]
        return predictions

    def save_model(self):
        """Save the trained model weights and normalization parameters to a file"""
        if not self._weights:
            raise ValueError("Model must be trained before saving!")
        
        model_data = {
            'weights': {class_name: weights.tolist() for class_name, weights in self._weights.items()},
            'classes': self._classes.tolist(),
            'mean': self._mean.tolist(),
            'std': self._std.tolist(),
            'is_fitted': self._is_fitted
        }
        
        import json
        with open('weights.json', 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath='weights.json'):
        """Load a previously saved model"""
        import json
        import numpy as np
        
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self._weights = {class_name: np.array(weights) 
                        for class_name, weights in model_data['weights'].items()}
        self._classes = np.array(model_data['classes'])
        self._mean = np.array(model_data['mean'])
        self._std = np.array(model_data['std'])
        self._is_fitted = model_data['is_fitted']
        
        print(f"Model loaded from {filepath}")