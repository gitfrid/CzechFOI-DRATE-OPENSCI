import sys
import importlib.metadata

def check_env():
    # List of requirements from your preprint scripts
    packages = [
        "numpy", 
        "pandas", 
        "statsmodels", 
        "scipy", 
        "joblib", 
        "tqdm", 
        "plotly",
        "tqdm-joblib"
    ]
    
    print(f"{'Package':<20} | {'Version':<15}")
    print("-" * 40)
    print(f"{'Python':<20} | {sys.version.split()[0]:<15}")

    for package in packages:
        try:
            version = importlib.metadata.version(package)
            print(f"{package:<20} | {version:<15}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package:<20} | NOT INSTALLED")
    print("-" * 40)

if __name__ == "__main__":
    check_env()