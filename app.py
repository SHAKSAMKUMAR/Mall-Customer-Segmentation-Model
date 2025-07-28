import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("🧠 Mall Customer Segmentation")

st.markdown("Enter details for each customer:")

# Number of customers
num_customers = st.number_input("Number of customers to input", min_value=1, max_value=20, value=5)

data = {'Customer ID': [], 'Annual Income (k$)': [], 'Spending Score (1-100)': []}

for i in range(num_customers):
    st.subheader(f"Customer {i + 1}")
    customer_id = st.text_input(f"Customer {i + 1} ID", key=f"id_{i}")
    income = st.number_input(f"Annual Income (k$) for Customer {i + 1}", key=f"income_{i}")
    score = st.slider(f"Spending Score (1-100) for Customer {i + 1}", min_value=1, max_value=100, key=f"score_{i}")

    data['Customer ID'].append(customer_id)
    data['Annual Income (k$)'].append(income)
    data['Spending Score (1-100)'].append(score)

# When data is entered, run clustering
if st.button("Run Clustering"):
    df = pd.DataFrame(data)

    st.subheader("Entered Data")
    st.write(df)

    # Features
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Fit KMeans
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    df['Cluster'] = kmeans.labels_

    st.subheader("Clustered Customers")
    st.write(df)

    # Plot the clusters
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'],
                         c=df['Cluster'], cmap='plasma')
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")
    st.pyplot(fig)
