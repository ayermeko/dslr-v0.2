# Magic Hat - Logistic Regression Multi-Classifier

A Hogwarts House sorting system using one-vs-all logistic regression with gradient descent.

## 📋 Project Overview

This project implements a multi-class classifier to sort students into Hogwarts houses based on their academic performance. The system uses logistic regression with a one-vs-all (one-vs-rest) approach to predict which house each student belongs to.

## 🎯 Implementation Requirements

### Phase 1: Training (`logreg_train.py`)
Trains logistic regression models using gradient descent and saves the learned parameters.

### Phase 2: Prediction (`logreg_predict.py`) 
Loads trained models and generates house predictions for new students.

## 📋 Complete TODO List

### 🎯 **Phase 1: logreg_train.py**

#### **Data Handling:**
- [ ] Parse command line argument for `dataset_train.csv`
- [ ] Load and validate the training dataset
- [ ] Extract manually selected features from pair plot analysis:
  - 'Defense Against the Dark Arts'
  - 'Charms' 
  - 'Herbology'
  - 'Divination'
  - 'Muggle Studies'
- [ ] Handle missing values (mean imputation)
- [ ] Normalize/standardize features (mean=0, std=1)
- [ ] Encode house labels (Gryffindor=0, Slytherin=1, Ravenclaw=2, Hufflepuff=3)

#### **Logistic Regression Implementation:**
- [ ] Implement sigmoid function: `σ(z) = 1/(1 + e^(-z))`
- [ ] Implement cost function (log-likelihood)
- [ ] Implement gradient calculation
- [ ] Implement gradient descent algorithm
- [ ] Add regularization (optional but recommended)

#### **One-vs-All Training:**
- [ ] Create 4 binary classifiers (one for each house)
- [ ] Train each classifier:
  - [ ] Gryffindor vs Others
  - [ ] Slytherin vs Others  
  - [ ] Ravenclaw vs Others
  - [ ] Hufflepuff vs Others
- [ ] Store learned weights for each classifier

#### **Model Persistence:**
- [ ] Save trained weights to file (JSON/CSV format)
- [ ] Include feature names and normalization parameters
- [ ] Add metadata (training accuracy, convergence info)

### 🔮 **Phase 2: logreg_predict.py**

#### **Data Handling:**
- [ ] Parse command line arguments for `dataset_test.csv` and weights file
- [ ] Load test dataset
- [ ] Extract same features used in training
- [ ] Handle missing values (same strategy as training)
- [ ] Apply same normalization (using training parameters)

#### **Prediction Implementation:**
- [ ] Load trained weights from file
- [ ] Implement prediction function using sigmoid
- [ ] For each test sample:
  - [ ] Calculate probability for each house (4 classifiers)
  - [ ] Select house with highest probability
- [ ] Generate predictions for all test samples

#### **Output Generation:**
- [ ] Create `houses.csv` with exact format:
  ```
  Index,Hogwarts House
  0,Gryffindor
  1,Slytherin
  ...
  ```
- [ ] Ensure proper indexing matches test dataset

### ⚙️ **Phase 3: Testing & Validation**

#### **Performance Evaluation:**
- [ ] Calculate training accuracy
- [ ] Implement cross-validation (optional)
- [ ] Test on validation split
- [ ] Generate confusion matrix
- [ ] Calculate precision, recall, F1-score per house

#### **Error Handling:**
- [ ] Validate file paths exist
- [ ] Handle malformed CSV data
- [ ] Check for feature mismatches between train/test
- [ ] Add informative error messages

## 🛠️ **Implementation Guidelines**

### **Key Functions to Implement:**

#### `logreg_train.py`
```python
def sigmoid(z):
    """Sigmoid activation function"""
    
def cost_function(X, y, theta):
    """Calculate logistic regression cost"""
    
def gradient_descent(X, y, theta, learning_rate, iterations):
    """Optimize weights using gradient descent"""
    
def train_one_vs_all(X, y, num_classes):
    """Train multiple binary classifiers"""
    
def save_model(weights, filename):
    """Save trained model to file"""
```

#### `logreg_predict.py`
```python
def load_model(filename):
    """Load trained model from file"""
    
def predict_probabilities(X, weights):
    """Calculate prediction probabilities"""
    
def predict_classes(X, weights):
    """Make final class predictions"""
    
def save_predictions(predictions, filename):
    """Save predictions to houses.csv"""
```

### **Recommended Parameters:**
- **Learning rate**: 0.01 - 0.1
- **Iterations**: 1000 - 5000  
- **Convergence threshold**: 1e-6
- **Features**: 5 (manually selected from pair plot analysis)

### **Model File Format:**
```json
{
    "features": [
        "Defense Against the Dark Arts",
        "Charms", 
        "Herbology",
        "Divination",
        "Muggle Studies"
    ],
    "normalization": {
        "mean": [mean1, mean2, mean3, mean4, mean5],
        "std": [std1, std2, std3, std4, std5]
    },
    "weights": {
        "Gryffindor": [w0, w1, w2, w3, w4, w5],
        "Slytherin": [w0, w1, w2, w3, w4, w5],
        "Ravenclaw": [w0, w1, w2, w3, w4, w5], 
        "Hufflepuff": [w0, w1, w2, w3, w4, w5]
    }
}
```

## 🧮 **Mathematical Foundation**

### **Sigmoid Function:**
```
σ(z) = 1 / (1 + e^(-z))
```

### **Cost Function (Log-Likelihood):**
```
J(θ) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

### **Gradient:**
```
∂J/∂θ = (1/m) * X^T * (h(x) - y)
```

### **One-vs-All Strategy:**
Train 4 separate binary classifiers:
- House A vs {House B, House C, House D}
- Prediction = argmax(P(House A), P(House B), P(House C), P(House D))

## 🎯 **Feature Selection Rationale**

Features selected through **visual pair plot analysis**:

1. **Defense Against the Dark Arts** - Strong discriminative power
2. **Charms** - Excellent class separation  
3. **Herbology** - Good Hufflepuff indicator
4. **Divination** - Strong Ravenclaw/Gryffindor separator
5. **Muggle Studies** - Unique patterns per house

These 5 features were chosen over algorithmic selection because:
- ✅ Visual pair plot analysis revealed clear separation patterns
- ✅ Human pattern recognition detected non-linear relationships
- ✅ Domain knowledge guided meaningful feature selection
- ✅ Reduced overfitting risk compared to using all features

## 🚀 **Usage**

### Training:
```bash
python logreg_train.py datasets/dataset_train.csv
```

### Prediction:
```bash
python logreg_predict.py datasets/dataset_test.csv model_weights.json
```

### Output:
- `model_weights.json` - Trained model parameters
- `houses.csv` - House predictions for test data

## 🏆 **Success Criteria**

- [ ] Training converges with low cost
- [ ] Predictions match required CSV format
- [ ] High accuracy on validation data
- [ ] Robust handling of edge cases
- [ ] Clean, readable code structure

---

**Note**: This implementation prioritizes mathematical understanding and clean gradient descent implementation over using external ML libraries. The focus is on learning the fundamentals of logistic regression and