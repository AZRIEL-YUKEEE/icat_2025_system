import streamlit as st
import pandas as pd
import mysql.connector
import io
from datetime import date

# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",   # change if needed
        password="",   # change if needed
        database="icat_db"
    )

# ----------------- FETCH STUDENTS -----------------
def fetch_students(year=None):
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    base_query = """SELECT application_number, family_name, first_name, middle_name, sex, strand, course,
                           gwa, date_taken, general_ability, verbal_aptitude, numerical_aptitude,
                           spatial_aptitude, perceptual_aptitude, manual_dexterity
                    FROM students WHERE 1=1"""
    params = []

    # Filter by year if chosen
    if year:
        base_query += " AND YEAR(date_taken) = %s"
        params.append(year)

    cur.execute(base_query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows)

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="ICAT Report Generator", layout="wide")
st.title("📊 ICAT Student Report Generator")

# Sidebar Year Filter
st.sidebar.header("Filter by Year")

# Build list of years dynamically from DB
def get_years():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT YEAR(date_taken) FROM students ORDER BY YEAR(date_taken) DESC")
    years = [row[0] for row in cur.fetchall() if row[0] is not None]
    cur.close()
    conn.close()
    return years

years = get_years()
year_selected = st.sidebar.selectbox("Select Year", options=["All"] + [str(y) for y in years])

# Generate Report
if st.button("🔍 Generate Report"):
    year_filter = None if year_selected == "All" else int(year_selected)
    df = fetch_students(year_filter)

    if df.empty:
        st.warning("No records found for the selected year.")
    else:
        st.success(f"✅ Found {len(df)} records for year {year_selected if year_filter else 'All Years'}")
        st.dataframe(df)

        # Convert to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Report")

            # Summary sheet
            summary = {
                "Total Records": [len(df)],
                "Average GWA": [round(df["gwa"].mean(), 2) if "gwa" in df else None]
            }
            pd.DataFrame(summary).to_excel(writer, index=False, sheet_name="Summary")

        excel_data = output.getvalue()

        st.download_button(
            label="📥 Download Excel Report",
            data=excel_data,
            file_name=f"icat_report_{year_selected if year_filter else 'all'}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
