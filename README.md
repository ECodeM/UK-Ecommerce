## Customer Segmentation Project

**Overview** \
This project aims to analyze customer purchase data from an online retail store in the UK to segment customers into distinct groups. By using K-Means clustering, the goal is to identify different customer behaviors and preferences, enabling targeted marketing strategies.

**Findings**

The segmentation revealed two distinct customer groups:

**High-Value Customers:** This group tends to purchase larger quantities of products and may be more responsive to targeted marketing efforts. They contribute significantly to overall sales and should be prioritized for loyalty programs.

**Low-Value Customers:** This group generally makes fewer purchases and may require different marketing strategies, such as promotions or personalized offers, to increase engagement and sales.

![image](https://github.com/user-attachments/assets/317f2f4e-7d68-4262-a412-a1bf60f02216)

![image](https://github.com/user-attachments/assets/db637eed-d4ee-48c7-92ef-850edc034552)

![image](https://github.com/user-attachments/assets/fa06aebd-72c0-4307-8ceb-859b3991b38a)


**Dataset** \
The dataset used in this analysis is the Online Retail dataset from the UCI Machine Learning Repository. It contains transactions from a UK-based online retail store, with key features including:

Invoice: Unique identifier for each transaction \
Customer ID: Unique identifier for each customer \
Quantity: The number of items purchased \
Price: Price per item

**Data Preparation** \
The data preparation involves the following steps:

- Data Cleaning: Handling missing values and removing duplicates.
- Aggregation: Grouping the data by CustomerID to calculate:
- Total quantity purchased
- Average unit price
- Number of purchases
- Clustering
- K-Means clustering is applied to segment customers based on their purchasing behavior. The optimal number of clusters determined through the elbow method and the silhouette score was 2. The resulting clusters provide insights into different customer segments.

**Files** \
data_preparation.py: Loads, cleans, and prepares the data for clustering. \
clustering.py: Performs K-Means clustering and assigns cluster labels to customers. \
visualization.py: Generates visualizations to illustrate the clustered data.
