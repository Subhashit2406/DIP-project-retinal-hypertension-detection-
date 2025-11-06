import os
import requests
import zipfile
import tarfile
from pathlib import Path

# Dataset URLs and information (Working Links - Manual Download Required)
datasets = {
    'DRIVE': {
        'url': 'https://drive.google.com/uc?export=download&id=1Rz-kGz9BbD6nqHvWd6M49vRZsw4N2pq9',
        'filename': 'DRIVE.zip',
        'extract_type': 'zip',
        'manual_url': 'https://drive.grand-challenge.org/site/competitions/drive/'
    },
    'STARE': {
        'url': 'https://www.clemson.edu/cecas/research/labs/vipl/downloads/STARE.zip',
        'filename': 'STARE.zip',
        'extract_type': 'zip',
        'manual_url': 'https://cecas.clemson.edu/~ahoover/stare/'
    },
    'MESSIDOR': {
        'url': 'https://www.adcis.net/wp-content/uploads/2020/07/Messidor.zip',
        'filename': 'Messidor.zip',
        'extract_type': 'zip',
        'manual_url': 'https://www.adcis.net/en/third-party/messidor/'
    },
    'CHASE_DB1': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/andrewmvd/chasedb1',
        'filename': 'CHASE_DB1.zip',
        'extract_type': 'zip',
        'manual_url': 'https://blogs.kingston.ac.uk/retinal/chasedb1/'
    }
}

def download_file(url, filename, chunk_size=8192):
    """Download file from URL with progress"""
    print(f"Downloading {filename} from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(".1f", end='', flush=True)
                else:
                    print(".", end='', flush=True)

    print("Download complete!")

def extract_file(filename, extract_type, extract_to):
    """Extract zip or tar files"""
    print(f"Extracting {filename} to {extract_to}")

    if extract_type == 'zip':
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif extract_type == 'tar':
        with tarfile.open(filename, 'r') as tar_ref:
            tar_ref.extractall(extract_to)

    print("Extraction complete!")

def main():
    # Create dataset directory if it doesn't exist
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)

    for name, info in datasets.items():
        try:
            print(f"\n=== Downloading {name} Dataset ===")

            # Download the file
            download_file(info['url'], info['filename'])

            # Extract the file
            extract_to = dataset_dir / name.lower()
            extract_to.mkdir(exist_ok=True)
            extract_file(info['filename'], info['extract_type'], str(extract_to))

            # Clean up downloaded file
            os.remove(info['filename'])
            print(f"Cleaned up {info['filename']}")

        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")
            print(f"📥 Please download manually from: {info.get('manual_url', 'Search online for ' + name + ' dataset')}")
            print(f"💾 Save as: {info['filename']}")
            print(f"📂 Extract to: dataset/{name.lower()}/")
            print()

    print("\nDataset download process completed!")
    print("Check the 'dataset' directory for downloaded datasets")

if __name__ == "__main__":
    main()
