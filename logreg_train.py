import sys
from core.model.preprocessing import clear_data, split_randomize
from core.model.classify import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: missing .csv or arg issue.")        
        X, y = clear_data(filepath=sys.argv[1])
        # loading selected features
        X_train, X_test, y_train, y_test = split_randomize(X, y, test_size=0.3, random_state=42)

        model = LogisticRegression()

        model.fit_normalize(X_train)

        X_train_norm = model.transform(X_train)
        X_test_norm = model.transform(X_test)


        # model.fit(X_train_norm, y_train)

        # y_pred = model.predict(X_test_norm)

        # print(f'Misclasified samples: {sum(y_test != y_pred)}')
        # print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

        # model.save_model()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()