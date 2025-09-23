import sys

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: > python3 logreg_train.py [path to dataset]")
        # Understand this before imprementing:
        # - Form pair plot visualization, which features to use for logistic regression?
        # - What does that mean to find a minimum error using a gradient descent
        # - How does a pair plot visualization can relate to this classification problem?
        # - How does predicted values can linearly seperate different classes?
        # - ...
    except Exception as e:
        print(f"{type(e).__name__}: {e}")