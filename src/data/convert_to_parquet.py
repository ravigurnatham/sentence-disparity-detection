import os
import zipfile
import pandas as pd
import logging
from glob import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_all_zips(raw_dir):
    zips = glob(os.path.join(raw_dir, '*.zip'))
    for zip_file in zips:
        extract_dir = zip_file.replace('.zip', '')
        if not os.path.exists(extract_dir):
            logging.info(f"Extracting {zip_file} to {extract_dir}")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(extract_dir)
        else:
            logging.info(f"{extract_dir} already exists. Skipping extraction.")

def convert_to_parquet(raw_dir, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    
    # After extraction, search for CSV or SAS files in raw subdirectories
    data_files = []
    data_files.extend(glob(os.path.join(raw_dir, '**', '*.csv'), recursive=True))
    data_files.extend(glob(os.path.join(raw_dir, '**', '*.sas7bdat'), recursive=True))
    data_files.extend(glob(os.path.join(raw_dir, '**', '*.txt'), recursive=True)) # sometimes delimited text
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        base_name, ext = os.path.splitext(filename)
        out_path = os.path.join(processed_dir, f"{base_name}.parquet")
        
        if os.path.exists(out_path):
            logging.info(f"{out_path} already exists. Skipping conversion.")
            continue
            
        logging.info(f"Processing {file_path}...")
        try:
            if ext.lower() == '.csv' or ext.lower() == '.txt':
                # Attempt to read as csv
                # try low_memory=False to avoid DtypeWarnings
                df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
                # Some text files might be tab or space separated, but CSV is standard fallback.
            elif ext.lower() == '.sas7bdat':
                df = pd.read_sas(file_path, encoding='latin1')
            else:
                logging.warning(f"Unsupported file extension {ext} for {file_path}")
                continue
                
            logging.info(f"Loaded {file_path} with shape {df.shape}. Converting to Parquet...")
            df.columns = [str(c).upper().strip() for c in df.columns] # normalize columns
            df.to_parquet(out_path, engine='pyarrow', index=False)
            logging.info(f"Saved to {out_path}")
        except Exception as e:
            logging.error(f"Failed to convert {file_path}: {e}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'raw'))
    processed_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed'))
    
    extract_all_zips(raw_dir)
    convert_to_parquet(raw_dir, processed_dir)
