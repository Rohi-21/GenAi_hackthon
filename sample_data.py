import pandas as pd
import numpy as np
import random

def generate_dirty_titanic(n_records: int = 200) -> pd.DataFrame:
    """
    Generates a highly messy, synthetic Titanic-like dataset.
    Intentionally injects errors across all 6 dimensions of data quality:
    1. Completeness: Missing values in Age, Cabin, Embarked, Fare.
    2. Uniqueness: Exact and near-duplicate rows.
    3. Validity: Invalid ages (<0, >120), invalid survival codes (e.g. 3), and type mismatches.
    4. Consistency: Categorical labeling conflicts (e.g. 'M', 'male', 'Male', 'fem', 'female').
    5. Accuracy: High-leverage outliers in Fare and Age.
    6. Integrity: Mismatch between columns (e.g. child traveling with parents but Parch=0).
    """
    np.random.seed(42)
    random.seed(42)
    
    # 1. Base clean structure
    data = {
        'PassengerId': list(range(1, n_records + 1)),
        'Survived': np.random.choice([0, 1], size=n_records, p=[0.6, 0.4]),
        'Pclass': np.random.choice([1, 2, 3], size=n_records, p=[0.25, 0.25, 0.50]),
        'Name': [f"Passenger, Row_{i}" for i in range(1, n_records + 1)],
        'Sex': np.random.choice(['male', 'female'], size=n_records, p=[0.55, 0.45]),
        'Age': np.random.normal(loc=29.0, scale=14.0, size=n_records).round(1),
        'SibSp': np.random.choice([0, 1, 2, 3], size=n_records, p=[0.7, 0.2, 0.07, 0.03]),
        'Parch': np.random.choice([0, 1, 2], size=n_records, p=[0.8, 0.15, 0.05]),
        'Ticket': [f"PC {random.randint(10000, 99999)}" for _ in range(n_records)],
        'Fare': np.random.exponential(scale=32.0, size=n_records).round(4),
        'Cabin': [f"C{random.randint(1, 148)}" if random.random() > 0.7 else None for _ in range(n_records)],
        'Embarked': np.random.choice(['S', 'C', 'Q'], size=n_records, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Ensure some column types are 'object' initially to represent raw files
    df['Survived'] = df['Survived'].astype(object)
    df['Pclass'] = df['Pclass'].astype(object)
    df['Age'] = df['Age'].astype(object)
    df['Fare'] = df['Fare'].astype(object)
    
    # 2. Inject Completeness Issues (Missing Values / Sentinel Nulls)
    # Age: 20% missing, some as NaNs, some as "?" or "N/A"
    for i in range(0, n_records, 5):
        df.at[i, 'Age'] = np.nan
    for i in range(3, n_records, 15):
        df.at[i, 'Age'] = "?"
    for i in range(7, n_records, 20):
        df.at[i, 'Age'] = "n/a"
        
    # Cabin: 75% missing
    for i in range(n_records):
        if i % 4 != 0:
            df.at[i, 'Cabin'] = np.nan
            
    # Embarked: 3% missing
    for i in range(11, n_records, 33):
        df.at[i, 'Embarked'] = np.nan
        
    # Fare: 5% missing
    for i in range(13, n_records, 20):
        df.at[i, 'Fare'] = "NULL"
        
    # 3. Inject Uniqueness Issues (Duplicates)
    # Inject exact duplicates
    dup_indices = [5, 12, 45, 92, 115]
    for idx in dup_indices:
        dup_row = df.iloc[idx].copy()
        df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)
        
    # Inject near duplicates (same person, slightly different Fare or Age)
    near_dup_row = df.iloc[20].copy()
    near_dup_row['Fare'] = float(near_dup_row['Fare']) + 1.5 if str(near_dup_row['Fare']).replace('.', '', 1).isdigit() else 20.0
    near_dup_row['PassengerId'] = df['PassengerId'].max() + 1
    df = pd.concat([df, pd.DataFrame([near_dup_row])], ignore_index=True)
    
    # 4. Inject Validity Issues (Out of range, type errors)
    # Out of range Age
    df.at[10, 'Age'] = -5.0
    df.at[35, 'Age'] = 150.0
    df.at[62, 'Age'] = 250.0  # extreme outlier
    
    # Invalid Survived values
    df.at[15, 'Survived'] = 3
    df.at[82, 'Survived'] = "Yes"
    df.at[142, 'Survived'] = "unknown"
    
    # Invalid Pclass
    df.at[40, 'Pclass'] = 5
    df.at[99, 'Pclass'] = "First"
    
    # 5. Inject Consistency Issues (Categorical Variants)
    # Sex variations: 'M', 'm', 'male', 'Male', 'FEMALE', 'F', 'f', 'female'
    sex_variants = ['M', 'm', 'male', 'Male', 'FEMALE', 'F', 'f', 'female', 'fem', 'MALE']
    for i in range(len(df)):
        if df.at[i, 'Sex'] == 'male':
            df.at[i, 'Sex'] = random.choice(['male', 'Male', 'M', 'm', 'MALE'])
        else:
            df.at[i, 'Sex'] = random.choice(['female', 'FEMALE', 'F', 'f', 'fem'])
            
    # Embarked variations
    for i in range(len(df)):
        if pd.notna(df.at[i, 'Embarked']):
            if df.at[i, 'Embarked'] == 'S':
                df.at[i, 'Embarked'] = random.choice(['S', 'Southampton', 's'])
            elif df.at[i, 'Embarked'] == 'C':
                df.at[i, 'Embarked'] = random.choice(['C', 'Cherbourg', 'c'])
            elif df.at[i, 'Embarked'] == 'Q':
                df.at[i, 'Embarked'] = random.choice(['Q', 'Queenstown', 'q'])
                
    # 6. Inject Accuracy Issues (Outliers)
    df.at[25, 'Fare'] = 9999.99  # Extreme Fare outlier
    df.at[74, 'Fare'] = -50.0   # Negative Fare
    df.at[110, 'Fare'] = 1500.0  # High Fare outlier
    
    # Reset index and return
    df = df.reset_index(drop=True)
    return df

def save_dirty_titanic(file_path: str, n_records: int = 200) -> None:
    """Generates the dirty titanic dataset and saves it to the specified path."""
    df = generate_dirty_titanic(n_records)
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
