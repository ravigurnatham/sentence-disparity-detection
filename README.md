<div align="center">
  <h1> EB-2 NIW: Predictive Justice & Sentencing Disparity Prototype</h1>
  <p><strong>A Real-Time Machine Learning Pipeline to Detect and Mitigate Systemic Bias in the U.S. Federal Judiciary</strong></p>
  <p><em>Work of: Ravi Gurnatham</em></p>
</div>

---

## 📖 Project Overview
This repository contains the foundational data ingestion, feature engineering, and predictive modeling pipeline designed to analyze U.S. Sentencing Commission (USSC) historical data. 

The primary objective of this **National Interest Waiver (EB-2 NIW) Endeavor** is to develop a real-time, "Legally Blind" AI system capable of integrating with the PACER/CM-ECF federal court network. By establishing a strictly objective mathematical baseline for criminal sentences, this system flags anomalous deviations caused by geographic jurisdiction or protected demographic attributes *before* a judge successfully finalizes a sentence.

## 📂 Repository Structure

```text
antigravity-niw/
│
├── data/
│   ├── raw/                 # Immutably stored USSC ZIP/CSV/SAS datasets
│   └── processed/           # High-speed Apache Parquet columnar databases
│
├── notebooks/
│   └── 02_Disparity_Analysis.ipynb # Beautifully generated EDA & Disparity Visualizations
│
├── src/
│   ├── data/
│   │   ├── download_ussc.py        # Webscraper to ingest massive federal datasets
│   │   └── convert_to_parquet.py   # Transformation engine (CSV -> PyArrow/Parquet)
│   │
│   ├── features/
│   │   ├── build_features.py       # Drops demographics to create a 'Legally Blind' tensor
│   │   └── expand_notebook.py      # Utility for injecting Markdown interpretations
│   │
│   └── models/
│       └── train_xgboost.py        # eXtreme Gradient Boosting objective baseline trainer
│
├── mathematical_foundation.md      # LaTeX representations of the loss & fairness formulations
├── phase_1_architecture.md         # Data ingestion Mermaid architectures
├── phase_1_research.md             # Detailed scenarios on "Messy Data" & mapping dummy codes
├── phase_2_research.md             # Detailed notes on XGBoost grid modeling & bias laundering
├── requirements.txt                # Python environment dependencies
└── README.md                       # This document
```

---

## 📚 Detailed Research Documentation
As part of this prototype, we have thoroughly documented the theoretical and mathematical scenarios involved in predictive justice. Please refer to these specific artifacts (slides/phases) for deep-dives into the project's foundation:

1. **[Mathematical Foundation](mathematical_foundation.md)**: Details the separation of objective law from subjective discretion using XGBoost loss functions and Hierarchical GLMM math structures.
2. **[Phase 1 Architecture](phase_1_architecture.md)**: Contains mermaid flowchart diagrams detailing the web scraping, extraction, and automated Parquet transformation pipeline.
3. **[Phase 1 Research & Data Profiling](phase_1_research.md)**: Breaks down the legacy constraints of SAS to CSV transformations, schema volatility, and how we handle extreme dummy codes like "Life in Prison".
4. **[Phase 2 Research methodologies](phase_2_research.md)**: Explains the "Legally Blind" constraint to avoid bias laundering, handling categorical severity, and isolating the "Trial Penalty".

---

##  Research & Implementation Phases

### Phase 1: Data Ingestion & Transformation 
To train an accurate model, we ingested over 60,000 individual federal offender records from the FY2024 dataset.
*   **The "Messy Data" Problem:** The pipeline autonomously identifies and strips arbitrary government dummy codes (e.g., USSC uses `9999` to represent "Life in Prison"). We mathematically clip these values to prevent extreme bias in regression curves.
*   **Performance Optimization:** Reading raw 1.5GB government CSVs causes local memory failures (OOM). We architected a PyArrow/Pandas engine (`convert_to_parquet.py`) that compresses legacy text files into high-speed **Apache Parquet** formats, allowing the system to query massive judicial datasets in milliseconds.

### Phase 2: "Legally Blind" Modeling & Disparity Detection
Models will simply memorize historical racism/sexism if given the opportunity. We prevented "bias laundering" by designing an incredibly strict feature engineering pipeline.
*   **Feature Selection:** The training tensor (`X_train`) is explicitly stripped of Location (`DISTRICT`), Race (`NEWRACE`), and Sex (`MONSEX`). The model learns the law using only Offense Severity and Criminal History Points.
*   **Why XGBoost?** Federal guidelines are not continuous; they are rigid matrices (Grids). Decision Trees perfectly mimic the logical splits a judge uses to calculate a sentence, vastly outperforming standard linear OLS regressions.
*   **Residual Disparity Analysis:** We quantify judicial bias mathematically by calculating the Residual ($\epsilon_{bias}$):
  
  ```math
  \text{Residual} = \text{Actual Sentence Imposed} - \text{Objective AI Baseline Prediction}
  ```

### Key Statistical Findings (From EDA Notebook)
Our Phase 2 model mathematically proved the necessity of this NIW endeavor:
1.  **Geographic Lottery:** A defendant in Federal District 11 received sentences averaging **+35.3 months** harsher than the baseline, while defendants in District 94 received sentences **-31.4 months** lower for the exact same crime and criminal history. A disparity of 5.5 years purely based on geography.
2.  **Demographic Disparities:** Female defendants averaged sentences nearly **19 months lighter** than male defendants strictly controlling for legal facts.
3.  **The Trial Penalty:** Trend analysis proves that judges deviate from the guidelines most aggressively on high-severity crimes, resulting in drastically varied applications of justice depending on the presiding judge.

---

##  Quick Start Guide

**1. Clone and Setup Environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Execute the Data Pipeline (Phase 1)**
```bash
# Downloads the FY24 USSC Individual Offender dataset
python src/data/download_ussc.py

# Converts the massive raw CSV to compressed Parquet format
python src/data/convert_to_parquet.py
```

**3. Train the Baseline and Extract Residuals (Phase 2)**
```bash
# Isolates the objective legal facts from protected demographic classes
python src/features/build_features.py

# Trains the XGBoost Regressor and outputs demographic Disparate Impact Metrics
python src/models/train_xgboost.py
```

## 🔮 Future Roadmap (Phase 3)
The next evolution of this prototype involves migrating the XGBoost predicting model into a localized **FastAPI** backend. This will simulate a real-world integration where a federal judge typing a sentence into the PACER terminal immediately receives a ping if their planned sentence breaks the statistical boundaries of geographic/racial fairness.
