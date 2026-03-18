import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(raw_parquet_path, output_parquet_path):
    if not os.path.exists(raw_parquet_path):
        logging.error(f"Input file not found: {raw_parquet_path}")
        return
        
    logging.info(f"Loading data from {raw_parquet_path}...")
    df = pd.read_parquet(raw_parquet_path)
    
    # 1. Target Variables
    # SENTTOT is total sentence in months. We will filter out missing/life sentences for baseline regression.
    # Often, courts use 9999 or similar codes for 'life' or 'missing'. 
    # For a basic prototype, we drop rows where SENTTOT is null.
    initial_len = len(df)
    df = df.dropna(subset=['SENTTOT'])
    df = df[df['SENTTOT'] < 9990]  # Filter out extreme dummy codes if any
    logging.info(f"Filtered out null/extreme sentence lengths. Retained {len(df)} from {initial_len} records.")
    
    # 2. Key Predictive Features (The "Legally Blind" model subset)
    # OFFTYPE2: primary offense type
    # XCRHISSR: Criminal History Category (usually 1 through 6)
    # DISTRICT: Judicial district
    # TOTCHPTS: Total criminal history points
    
    predictive_cols = ['OFFTYPE2', 'XCRHISSR', 'DISTRICT', 'TOTCHPTS']
    
    # 3. Protected Attributes (For evaluation only, not training)
    # NEWRACE: Race/Ethnicity (e.g., 1=White, 2=Black, 3=Hispanic, etc.)
    # MONSEX: Sex (0=Male, 1=Female)
    eval_cols = ['NEWRACE', 'MONSEX']
    
    # Combine target, predictors, and eval columns
    target_col = 'SENTTOT'
    keep_cols = [target_col] + predictive_cols + eval_cols
    
    # Filter dataset to just these columns to save memory
    # Ensure they exist (column names can vary slightly by year)
    available_cols = [col for col in keep_cols if col in df.columns]
    
    if target_col not in available_cols:
        logging.error(f"Target column {target_col} not found in dataset!")
        return
        
    df_clean = df[available_cols].copy()
    
    # Basic Cleaning: drop rows with missing essential predictors
    df_clean = df_clean.dropna(subset=[col for col in predictive_cols if col in available_cols])
    
    logging.info(f"Final dataset shape for modeling: {df_clean.shape}")
    
    # Save the prepared feature subset
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, engine='pyarrow', index=False)
    logging.info(f"Saved engineered features to {output_parquet_path}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed', 'opafy24nid.parquet'))
    output_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed', 'features_fy24.parquet'))
    
    engineer_features(input_path, output_path)
