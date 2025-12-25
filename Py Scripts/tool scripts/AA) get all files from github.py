import requests

# Repository details
USER = "gitfrid"
REPO = "CzechFOI-DRATE-OPENSCI"
BRANCH = "main"
START_PATH = "Plot Results"

def print_all_entries(path):
    """Recursively prints every single entry found in the repository path."""
    api_url = f"https://api.github.com/repos/{USER}/{REPO}/contents/{path}?ref={BRANCH}"
    response = requests.get(api_url)
    
    if response.status_code != 200:
        print(f"[ERROR] Could not access {path} - Status: {response.status_code}")
        return

    items = response.json()

    for item in items:
        # Check type: 'dir' for folders, 'file' for files
        item_type = item['type'].upper()
        item_path = item['path']

        if item['type'] == 'dir':
            print(f"[DIR]  {item_path}")
            # Recurse into the folder
            print_all_entries(item_path)
        else:
            print(f"[FILE] {item_path}")

if __name__ == "__main__":
    print(f"--- Printing ALL entries in {REPO}/{START_PATH} ---")
    print_all_entries(START_PATH)
    print("--- End of Tree ---")