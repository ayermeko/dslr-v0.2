import sys
from core.operations import describe, validate


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Useage: python3 describe.py [dataset].csv")
        
        csv_path = sys.argv[1]

        dataset = validate(csv_path)
        describe(dataset)

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()