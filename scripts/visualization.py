import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that plots use seaborn's style
sns.set_theme(style="whitegrid")

def plot_elbow_method(wcss, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(wcss)+1), wcss, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal K')
    plt.savefig(os.path.join(output_path, 'elbow_method.png'))
    plt.close()

def plot_silhouette_scores(k_values, silhouette_scores, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Various K')
    plt.savefig(os.path.join(output_path, 'silhouette_scores.png'))
    plt.close()

def plot_rfm_distributions(rfm, output_path):
    metrics = ['Recency', 'Frequency', 'Monetary']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Cluster', y=metric, data=rfm, palette='Set2')
        plt.title(f'{metric} Distribution by Cluster')
        plt.savefig(os.path.join(output_path, f'{metric}_distribution.png'))
        plt.close()

def main():
    # Define paths
    data_path = os.path.join('..', 'raw_data', 'rfm_with_clusters.csv')
    output_path = os.path.join('..', 'images')

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the RFM data with cluster assignments
    rfm = pd.read_csv(data_path)

    # Load WCSS and Silhouette Score
    wcss_path = os.path.join('..', 'raw_data', 'wcss.csv')
    silhouette_path = os.path.join('..', 'raw_data', 'silhouette_scores.csv')

    if os.path.exists(wcss_path):
        wcss_df = pd.read_csv(wcss_path)
        wcss = wcss_df['WCSS'].tolist()
        plot_elbow_method(wcss, output_path)

    if os.path.exists(silhouette_path):
        silhouette_df = pd.read_csv(silhouette_path)
        k_values = silhouette_df['K'].tolist()
        silhouette_scores = silhouette_df['SilhouetteScore'].tolist()
        plot_silhouette_scores(k_values, silhouette_scores, output_path)

    # Generate RFM Distribution Box Plots
    plot_rfm_distributions(rfm, output_path)

    print("All visualizations have been saved to the 'images/' directory.")

if __name__ == "__main__":
    main()
