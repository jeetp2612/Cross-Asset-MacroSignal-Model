import importlib.util
import subprocess
import sys

REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "yfinance": "yfinance",
}


def missing_packages():
    return [package for module, package in REQUIRED.items() if importlib.util.find_spec(module) is None]


def main():
    if sys.version_info < (3, 10):
        print(f"This app needs Python 3.10 or newer. You have {sys.version.split()[0]}.")
        sys.exit(1)
    if importlib.util.find_spec("tkinter") is None:
        print("Python's window toolkit (tkinter) is missing.")
        print("Windows and Mac: reinstall Python from python.org and keep 'tcl/tk' ticked.")
        print("Linux: run  sudo apt install python3-tk")
        sys.exit(1)
    missing = missing_packages()
    if missing:
        print(f"These packages are missing: {', '.join(missing)}")
        answer = input("Install them now? (y/n): ").strip().lower()
        if answer != "y":
            print("Please run:  python -m pip install -r requirements.txt")
            sys.exit(1)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except subprocess.CalledProcessError:
            print("The installation failed. Check your internet connection and try again.")
            sys.exit(1)
    import gui

    gui.main()


if __name__ == "__main__":
    main()
