import streamlit as st
import pandas as pd
import mysql.connector
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from db_connection import get_connection

# ------------------ LOAD DATA FROM MYSQL ------------------
def load_data():
    conn = get_connection()
    query = "SELECT * FROM students"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ------------------ SAVE STUDENT RECORD ------------------
def save_student(record):
    conn = get_connection()
    cursor = conn.cursor()

    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""

    cursor.execute(sql, record)
    conn.commit()
    conn.close()

# ------------------ STREAMLIT APP ------------------
st.title("ICAT Student Clustering Dashboard")

# Upload CSV/Excel
st.subheader("📂 Upload Student Data")
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    st.write("Preview of Uploaded File:")
    st.dataframe(df_upload.head())

    # Save to DB
    if st.button("Save Uploaded Data to Database"):
        for _, row in df_upload.iterrows():
            save_student(tuple(row.values))
        st.success("Uploaded data saved to database successfully!")

# Manual Input
st.subheader("✍️ Manual Input of Student Data")
with st.form("student_form"):
    application_number = st.text_input("Application Number")
    family_name = st.text_input("Family Name")
    first_name = st.text_input("First Name")
    middle_name = st.text_input("Middle Name")
    sex = st.selectbox("Sex", ["Male", "Female"])
    strand = st.text_input("Strand")
    course = st.text_input("Course")
    general_ability = st.number_input("General Ability", min_value=0.0)
    verbal_aptitude = st.number_input("Verbal Aptitude", min_value=0.0)
    numerical_aptitude = st.number_input("Numerical Aptitude", min_value=0.0)
    spatial_aptitude = st.number_input("Spatial Aptitude", min_value=0.0)
    perceptual_aptitude = st.number_input("Perceptual Aptitude", min_value=0.0)
    manual_dexterity = st.number_input("Manual Dexterity", min_value=0.0)

    submitted = st.form_submit_button("Save Student")
    if submitted:
        record = (application_number, family_name, first_name, middle_name, sex, strand, course,
                  general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
                  perceptual_aptitude, manual_dexterity)
        save_student(record)
        st.success("Student record saved successfully!")

# Show Data + Clustering
st.subheader("📊 Student Data & Clustering")

df = load_data()
st.write(f"Total Students: {len(df)}")
st.dataframe(df)

if len(df) > 0:
    # Select features for clustering
    features = ["general_ability", "verbal_aptitude", "numerical_aptitude",
                "spatial_aptitude", "perceptual_aptitude", "manual_dexterity"]

    X = df[features]

    # Run KMeans
    k = st.slider("Select number of clusters (k)", 2, 6, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["cluster_group"] = kmeans.fit_predict(X)

    # Visualization
    st.subheader("Clustering Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x="numerical_aptitude", y="verbal_aptitude",
                    hue="cluster_group", palette="Set1", data=df, ax=ax)
    plt.title("K-Means Clustering of Students")
    st.pyplot(fig)

    # Save cluster groups back to DB
    if st.button("Update Cluster Groups in Database"):
        conn = get_connection()
        cursor = conn.cursor()
        for idx, row in df.iterrows():
            cursor.execute("UPDATE students SET cluster_group=%s WHERE id=%s", (int(row["cluster_group"]), int(row["id"])))
        conn.commit()
        conn.close()
        st.success("Cluster groups updated in database!")
