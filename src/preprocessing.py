import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Fill missing numeric columns with mean
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 2. Fill missing categorical columns with 'missing'
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].fillna('missing')

    # 3. Smoker binary mapping
    if 'smoker' in df.columns:
        df['smoker'] = df['smoker'].fillna('no')
        df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # 4. BMI category
    if 'bmi' in df.columns:
        def bmi_category(bmi):
            if bmi < 18.5:
                return 'underweight'
            elif bmi < 25:
                return 'normal'
            elif bmi < 30:
                return 'overweight'
            else:
                return 'obese'
        df['bmi_category'] = df['bmi'].apply(bmi_category)

    # 5. Scale numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df



'''
def main():
    # Sample data
    data = {
        'age': [25, 30, None, 40],
        'bmi': [27.5, None, 31.2, 22.0],
        'smoker': ['no', 'yes', 'no', None],
        'region': ['northwest', None, 'southeast', 'southwest']
    }
    df = pd.DataFrame(data)

    print("Original Data:")
    print(df, "\n")

    # Preprocess
    df_processed = preprocess_data(df)

    print("Processed Data:")
    print(df_processed)

if __name__ == "__main__":
    main()
'''