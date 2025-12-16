import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocessing import preprocess_data


def test_preprocess_data():
    # Sample data
    data = {
        'bmi': [27.5, None, 31.2, 22.0],
        'smoker': ['no', 'yes', 'no', 'yes'],
        'age': [25, 30, None, 40]
    }
    df = pd.DataFrame(data)

    # Run preprocessing
    processed = preprocess_data(df)

    # ---- Test 1: BMI missing values filled ----
    assert processed['bmi'].isnull().sum() == 0, "BMI missing values were not filled."

    # ---- Test 2: Age missing values filled ----
    assert processed['age'].isnull().sum() == 0, "Age missing values were not filled."

    # ---- Test 3: Smoker column is mapped correctly ----
    assert processed['smoker'].tolist() == [0, 1, 0, 1], "Smoker column was not mapped correctly."

    # ---- Test 4: BMI category created ----
    expected_categories = ['overweight', 'overweight', 'obese', 'normal']
    assert processed['bmi_category'].tolist() == expected_categories, "BMI categories are incorrect."

'''
    # ---- Test 5: Numeric scaling (0-1) ----
    for col in ['bmi', 'age']:
        assert processed[col].min() >= 0 and processed[col].max() <= 1, f"{col} is not scaled correctly."
'''