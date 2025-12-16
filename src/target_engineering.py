# src/target_engineering.py

import pandas as pd
from sklearn.cluster import KMeans
from src.config import RANDOM_STATE


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary (RFM) features per customer.
    """
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    snapshot_date = df["TransactionStartTime"].max()

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    ).reset_index()

    return rfm


def assign_proxy_target(rfm: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Assign a proxy credit risk target using KMeans clustering.
    
    Customers in the cluster with the lowest average monetary value
    are labeled as high risk.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    rfm["cluster"] = kmeans.fit_predict(rfm[["recency", "frequency", "monetary"]])

    high_risk_cluster = rfm.groupby("cluster")["monetary"].mean().idxmin()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm
