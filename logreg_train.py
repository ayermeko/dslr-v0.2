from core.operations import validate
import sys


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: missing .csv or arg issue.")        
        dataset = validate(sys.argv[1])
        
    except Exception as e:
        print(f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    main()