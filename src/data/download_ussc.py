import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file(url, target_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    with open(target_path, 'wb') as file, tqdm(
        desc=os.path.basename(target_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def find_and_download_ussc_data(target_dir='../../data/raw'):
    base_url = 'https://www.ussc.gov/research/datafiles/commission-datafiles'
    logging.info(f"Fetching page {base_url}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links on the page that point to zip files
    zip_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.zip') and ('spss' in href.lower() or 'sas' in href.lower() or 'csv' in href.lower() or 'fy' in href.lower()):
            full_url = urljoin(base_url, href)
            zip_links.append(full_url)
            
    target_url = "https://www.ussc.gov/sites/default/files/zip/opafy24nid_csv.zip"
    logging.info(f"Targeting USSC FY24 CSV dataset: {target_url}")

    filename = os.path.basename(target_url)
    target_path = os.path.join(target_dir, filename)
    
    if not os.path.exists(target_path):
        logging.info(f"Downloading {target_url} to {target_path}...")
        try:
            download_file(target_url, target_path)
            logging.info("Download completed.")
        except Exception as e:
            logging.error(f"Failed to download: {e}")
    else:
        logging.info(f"File {filename} already exists in {target_dir}. Skipping download.")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(script_dir, '..', '..', 'data', 'raw')
    find_and_download_ussc_data(target_dir=raw_data_dir)
