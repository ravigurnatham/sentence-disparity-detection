# Real-Time Federal Sentencing Disparity System - Prototype

This document outlines the implementation plan for the local prototype of the sentencing disparity prediction system, focusing initially on U.S. Sentencing Commission (USSC) public data.

## Proposed Changes
### Project Infrastructure
- Set up a standard data science Python project structure:
  - `data/raw/`: Raw downloaded USSC data.
  - `data/processed/`: Parquet files.
  - `notebooks/`: Jupyter notebooks for EDA.
  - `src/data/`: Data ingestion and cleaning scripts.
  - `src/features/`: Feature engineering scripts.
  - `src/models/`: Training and evaluation scripts.
- Use `requirements.txt` to manage dependencies (e.g., `pandas`, `pyarrow`, `scikit-learn`, `jupyter`, `requests`).

### Data Ingestion Pipeline
- **USSC Downloader**: A Python script to autonomously fetch USSC datafiles (SAS/CSV) for recent fiscal years.
- **Parquet Converter**: A script utilizing `pandas` and `pyarrow` to convert large tabular data into Parquet format for fast, memory-efficient querying.

## Phase 2: Modeling and Fairness Metrics
### Feature Engineering
- Process target variables: Total Sentence length (`SENTTOT`) and prison months (`TOTPRISN`).
- Process predictive features:
  - Offense Characteristics (e.g., offense level `OFFTYPE2`, mandatory minimums).
  - Criminal History (e.g., criminal history category `XCRHISSR`, points).
  - Judicial Context: District (`DISTRICT`).
- Identify protected attributes for fairness evaluation: Race/Ethnicity (`NEWRACE`) and Sex (`MONSEX`).

### Model Development
1. **Baseline Model (XGBoost/LightGBM)**: Train a non-linear tree ensemble to predict sentencing lengths based on legal guidelines and case facts, explicitly omitting protected attributes during training to establish a "blind" baseline.
2. **Fairness Evaluation**: 
   - Apply the baseline model to held-out test data.
   - Calculate Disparate Impact Ratio and Demographic Parity to identify variations across protected groups or specific districts.
3. **Draft GLMM Framework**: Scaffold the structure for a Hierarchical Bayesian Generalized Linear Mixed Model to explicitly model district and judge-level random effects (though Judge ID is often masked in public data, District operates as a proxy for regional variation).


