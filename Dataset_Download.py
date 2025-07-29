#Human Scar Detection (Industry Application Project)
#Step 1A - Dataset Downloader
#This script downloads and categorizes the Fitzpatrick17k dataset from its metadata file.
#Link to github https://github.com/mattgroh/fitzpatrick17k

import os
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict
import time
import random

#Relative Folders.

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "RawDataset"

METADATA_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"

#Sets maximum download size (if the dataset is too large for example).
MAX_DOWNLOAD_SIZE_GB = 25
MAX_DOWNLOAD_SIZE_BYTES = MAX_DOWNLOAD_SIZE_GB * (1024**3)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

### Scraper Logic ###

#Replaces spaces with underscores and remove characters that are invalid in folder names
def sanitize_label(label):
    return "".join(c for c in label.lower().replace(' ', '_') if c.isalnum() or c in ('_')).rstrip()

def download_image(session, url, save_path):
    try:
        img_response = session.get(url, headers=HEADERS, stream=True, timeout=30)
        img_response.raise_for_status() 

        image_size = 0
        with open(save_path, 'wb') as f:
            for chunk in img_response.iter_content(chunk_size=8192):
                f.write(chunk)
                image_size += len(chunk)
        return image_size

    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not download {url}. Error: {e}")
        return 0
    except IOError as e:
        print(f"Warning: Could not save file to {save_path}. Error: {e}")
        return 0

def main():
    print("### Dataset Downloader ###")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Images will be saved in categorized subfolders inside: {SAVE_DIR}\n")

    session = requests.Session()

    #Downloads and reads the CSV file.
    print(f"Getting data from: {METADATA_URL}...")
    try:
        df = pd.read_csv(METADATA_URL)
        #Drops rows where the URL is missing, as they cannot be downloaded
        df.dropna(subset=['url'], inplace=True)
        print(f"Data loaded, found {len(df)} images with valid URLs to download.")
    except Exception as e:
        print(f"Could NOT download or read the CSV metadata file. Error: {e}")
        return

    #Iterates through the data and downloads images.
    print("\nDownloading...\n")
    
    total_images = len(df)
    image_counters = defaultdict(int)
    total_downloaded_size = 0

    #Shuffles the dataframe to download images from different categories.
    df = df.sample(frac=1).reset_index(drop=True)

    for index, row in df.iterrows():
        if total_downloaded_size >= MAX_DOWNLOAD_SIZE_BYTES:
            print(f"\nStopping: Download limit of {MAX_DOWNLOAD_SIZE_GB} GB has been reached.")
            break

        label = row['label']
        url = row['url']
        
        category_folder_name = sanitize_label(label)
        category_folder_path = SAVE_DIR / category_folder_name
        category_folder_path.mkdir(parents=True, exist_ok=True)
        
        file_ext = Path(url).suffix.lower() or '.jpg'
        image_index = image_counters[category_folder_name]
        filename = f"{category_folder_name}_{image_index}{file_ext}"
        save_path = category_folder_path / filename

        print(f"Processing image {index + 1}/{total_images} | Category: {label}")
        
        downloaded_bytes = download_image(session, url, save_path)
        
        if downloaded_bytes > 0:
            total_downloaded_size += downloaded_bytes
            image_counters[category_folder_name] += 1
            total_gb = total_downloaded_size / (1024**3)
            print(f"Downloaded correctly, ({downloaded_bytes / 1024:.1f} KB). Total size: {total_gb:.3f} GB / {MAX_DOWNLOAD_SIZE_GB:.1f} GB")
            time.sleep(random.uniform(0.05, 0.1)) # Small polite pause

    print("\n### Download Complete ###")
    print(f"Total downloaded: {total_downloaded_size / (1024**3):.3f} GB")
    print("Final image counts for downloaded categories:")
    if not image_counters:
        print("No images were downloaded.")
    else:
        sorted_categories = sorted(image_counters.items(), key=lambda item: (-item[1], item[0]))
        for label, count in sorted_categories:
            print(f"  - {label}: {count} images")

if __name__ == '__main__':
    main()