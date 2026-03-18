import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_fairness(df_test, y_true, y_pred):
    """
    Very simple disparity analysis comparing baseline 'legally blind' predictions 
    to actual sentencing outcomes, broken down by race and sex.
    """
    df_test['Actual'] = y_true
    df_test['Predicted'] = y_pred
    df_test['Residual'] = df_test['Actual'] - df_test['Predicted']  # Positive = harsher than expected
    
    logging.info("--- Fairness Evaluation (Residuals) ---")
    
    # Sex differences (0 usually Male, 1 usually Female in USSC codebook, but we just group)
    if 'MONSEX' in df_test.columns:
        sex_grouped = df_test.groupby('MONSEX')['Residual'].mean()
        logging.info(f"Mean Residual by Sex:\n{sex_grouped}")
    
    # Race/Ethnicity differences 
    # (Typical USSC: 1=White, 2=Black, 3=Hispanic, 4=AmInd/Alaskan, 6=Asian/Pacific)
    if 'NEWRACE' in df_test.columns:
        race_grouped = df_test.groupby('NEWRACE')['Residual'].mean()
        logging.info(f"Mean Residual by Race:\n{race_grouped}")
        
    # District Level Disparity (Are some districts consistently harsher/lenient?)
    if 'DISTRICT' in df_test.columns:
        district_grouped = df_test.groupby('DISTRICT')['Residual'].mean().sort_values()
        logging.info(f"Most Lenient Districts (Mean Residual):\n{district_grouped.head(3)}")
        logging.info(f"Harshest Districts (Mean Residual):\n{district_grouped.tail(3)}")


def train_baseline_model(features_path):
    logging.info(f"Loading features from {features_path}")
    df = pd.read_parquet(features_path)
    
    # 1. Separate Predictors and Target
    target_col = 'SENTTOT'
    # Exclude protected attributes from training!
    exclude_cols = [target_col, 'NEWRACE', 'MONSEX']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Standardize data / handle categoricals for baseline prototype
    # XGBoost can handle NaNs natively
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode categorical features if any exist as strings
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
    # Keep evaluation columns aside
    eval_df = df[exclude_cols + ['DISTRICT']].copy() if 'DISTRICT' in df.columns else df[exclude_cols].copy()
    
    # 2. Train Test Split
    X_train, X_test, y_train, y_test, eval_train, eval_test = train_test_split(
        X, y, eval_df, test_size=0.2, random_state=42
    )
    
    logging.info(f"Training XGBoost Regressor on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # 3. Predict and Evaluate Accuracy
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Baseline accuracy metrics
    # Note: R^2 or similar can be calculated, but absolute error in months is practical
    logging.info(f"Baseline Legal Accuracy -> MSE: {mse:.2f}, MAE: {mae:.2f} months")
    
    # 4. Evaluate Fairness / Disparities
    evaluate_fairness(eval_test, y_test, y_pred)

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'data', 'processed', 'features_fy24.parquet'))
    train_baseline_model(features_path)
