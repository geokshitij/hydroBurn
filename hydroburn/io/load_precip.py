"""
Module for loading precipitation data.
"""
import pandas as pd
from pathlib import Path

def load_precipitation(file_path: str) -> pd.DataFrame:
    """
    Load precipitation data from CSV.
    
    Expected format:
    date,precipitation_mm
    YYYY-MM-DD,value
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and 'precipitation' column.
    """
    df = pd.read_csv(file_path)
    
    # Standardize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Find date column
    date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
    if not date_col:
        raise ValueError("Could not identify date column in precipitation file")
        
    # Find precip column
    precip_col = next((c for c in df.columns if 'precip' in c or 'prcp' in c), None)
    if not precip_col:
        raise ValueError("Could not identify precipitation column in precipitation file")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.rename(columns={precip_col: 'precipitation'})
    
    # Ensure numeric
    df['precipitation'] = pd.to_numeric(df['precipitation'], errors='coerce').fillna(0)
    
    return df[['precipitation']]
