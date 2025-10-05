# cluster_students.py
import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ DB CONNECTION ------------------
def get_connection():
    return mysql.connector.connect(
        host="localhost",      # change if needed
        user="root",           # your MySQL username
        password="",           # your MySQL password
        database="icat_db"     # your database
    )

# ------------------ LOAD STUDENTS ------------------
def load_students():
    conn = get_connection()
    query = """
        SELECT id, application_number, family_name, first_name, sex, strand, course,
               general_ability, verbal_aptitude, numerical_aptitude,
               spatial_aptitude, perceptual_aptitude, manual_dexterity,
               cluster_group
        FROM students
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ------------------ UPDATE CLUSTERS ------------------
def update_clusters(df):
    conn = get_connection()
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute(
            "UPDATE students SET cluster_group=%s WHERE id=%s",
            (int(row["cluster_group"]), int(row["id"]))
        )
    conn.commit()
    conn.close()

# ------------------ STREAMLIT APP ------------------
st.title("🎓 ICAT Student Clustering Dashboard")

# Load data
df = load_students()
st.subheader("📋 Student Records")
st.write(f"Total Students: {len(df)}")
st.dataframe(df)

if len(df) > 0:
    # Select features
    features = [
        "general_ability", "verbal_aptitude", "numerical_aptitude",
        "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"
    ]
    X = df[features]

    # Choose k
    k = st.slider("Select number of clusters (k)", 2, 6, 3)

    if st.button("🔄 Run KMeans Clustering"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        df["cluster_group"] = kmeans.fit_predict(X)

        # Update DB
        update_clusters(df)
        st.success("✅ Clustering complete! Database updated.")

        # Show updated data
        st.subheader("📊 Clustered Student Data")
        st.dataframe(df)

        # Visualization
        st.subheader("📈 Clustering Visualization")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.scatterplot(
            x="numerical_aptitude", y="verbal_aptitude",
            hue="cluster_group", palette="Set1", data=df, ax=ax
        )
        plt.title("K-Means Clustering of Students")
        st.pyplot(fig)
