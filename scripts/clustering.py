import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_rfm(file_path):
    """
    Loads the RFM metrics from a CSV file and returns a dataframe.
    """
    try:
        rfm = pd.read_csv(file_path)
        print(f"RFM metrics loaded successfully from {file_path}.")
        return rfm
    except Exception as e:
        print(f"Error loading RFM metrics: {e}")
        raise

def scale_features(rfm):
    """
    Scales the RFM features using StandardScaler.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    print("RFM features scaled successfully.")
    return rfm_scaled, scaler

def determine_optimal_k(rfm_scaled, max_k=10):
    """
    Determines the optimal number of clusters using the Elbow Method and Silhouette Scores.
    """
    wcss = []
    silhouette_scores = []
    K = range(2, max_k+1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(rfm_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k}: WCSS={kmeans.inertia_}, Silhouette Score={score}")

    return wcss, silhouette_scores

def apply_kmeans(rfm_scaled, n_clusters):
    """
    Applies K-Means clustering to the scaled RFM data.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    labels = kmeans.labels_
    print(f"K-Means clustering applied with K={n_clusters}.")
    return kmeans, labels

def save_clusters(rfm, labels, output_path):
    """
    Saves the RFM data with cluster labels to a CSV file.
    """
    rfm['Cluster'] = labels
    try:
        rfm.to_csv(output_path, index=False)
        print(f"Cluster assignments saved to {output_path}.")
    except Exception as e:
        print(f"Error saving cluster assignments: {e}")
        raise

def save_metrics(wcss, silhouette_scores, output_path_wcss, output_path_silhouette):
    """
    Saves WCSS and Silhouette Scores to CSV files.
    """
    try:
        # Save WCSS
        wcss_df = pd.DataFrame({
            'K': range(2, 2 + len(wcss)),
            'WCSS': wcss
        })
        wcss_df.to_csv(output_path_wcss, index=False)
        print(f"WCSS saved to {output_path_wcss}.")

        # Save Silhouette Scores
        silhouette_df = pd.DataFrame({
            'K': range(2, 2 + len(silhouette_scores)),
            'SilhouetteScore': silhouette_scores
        })
        silhouette_df.to_csv(output_path_silhouette, index=False)
        print(f"Silhouette Scores saved to {output_path_silhouette}.")
    except Exception as e:
        print(f"Error saving clustering metrics: {e}")
        raise

def main():
    # Define file paths
    rfm_input_path = os.path.join('..', 'raw_data', 'rfm_with_clusters.csv')
    rfm_output_path = os.path.join('..', 'raw_data', 'rfm_with_clusters.csv')
    wcss_output_path = os.path.join('..', 'raw_data', 'wcss.csv')
    silhouette_output_path = os.path.join('..', 'raw_data', 'silhouette_scores.csv')

    # Load RFM data
    rfm = load_rfm(rfm_input_path)

    # Scale features
    rfm_scaled, scaler = scale_features(rfm)

    # Determine optimal K
    wcss, silhouette_scores = determine_optimal_k(rfm_scaled, max_k=10)

    # Choose optimal K based on Silhouette Score (max score) (+2 bc K starts at 2)
    optimal_k = silhouette_scores.index(max(silhouette_scores))
    print(f"Optimal number of clusters determined: K={optimal_k}")

    # Apply K-Means with optimal K
    kmeans, labels = apply_kmeans(rfm_scaled, optimal_k)

    # Save cluster assignments
    save_clusters(rfm, labels, rfm_output_path)

    # Save clustering metrics
    save_metrics(wcss, silhouette_scores, wcss_output_path, silhouette_output_path)

if __name__ == "__main__":
    main()
