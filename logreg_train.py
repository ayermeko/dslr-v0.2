import sys
from core.operations import validate
from core.model.preprocessing import clear_data, split_randomize

def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: missing .csv or arg issue.")        
        X, y = clear_data(filepath=sys.argv[1])
        # loading selected features
        X_train, X_test, y_train, y_test = split_randomize(X, y)

        print(X_train)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()