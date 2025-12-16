import pandas as pd

def create_customer_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        std_amount=('Amount', 'std'),
        transaction_count=('Amount', 'count'),
        last_transaction=('TransactionStartTime', 'max')
    ).reset_index()

    return agg
