from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mysql.connector
from io import BytesIO

app = FastAPI()

# ✅ Enable CORS for Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Laravel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",          # change if needed
        password="",          # change if needed
        database="icat_db"    # change to your DB
    )

# Save to database
def save_students(records):
    conn = get_connection()
    cursor = conn.cursor()
    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    try:
        cursor.executemany(sql, records)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()

        # ✅ Accept both CSV and Excel
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(await file.read()))
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(BytesIO(await file.read()))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload CSV or Excel only.")

        # ✅ Ensure date_taken exists
        if "date_taken" not in df.columns:
            df["date_taken"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # ✅ Replace NaN and empty strings with None so MySQL stores them as NULL
        df = df.where(pd.notnull(df), None)  # replace NaN → None
        df = df.applymap(
            lambda x: None if (
                x is None or
                (isinstance(x, str) and x.strip().lower() in ["", "nan", "none"])
            ) else x
        )

        # ✅ Convert to list of tuples for DB insert with NaN → None
        records = []
        for _, row in df.iterrows():
            records.append((
                None if pd.isna(row.get("Application_Number")) else row.get("Application_Number"),
                None if pd.isna(row.get("Family_Name")) else row.get("Family_Name"),
                None if pd.isna(row.get("First_Name")) else row.get("First_Name"),
                None if pd.isna(row.get("Middle_Name")) else row.get("Middle_Name"),
                None if pd.isna(row.get("Sex")) else row.get("Sex"),
                None if pd.isna(row.get("Strand")) else row.get("Strand"),
                None if pd.isna(row.get("Course")) else row.get("Course"),
                None if pd.isna(row.get("General_Ability")) else row.get("General_Ability"),
                None if pd.isna(row.get("Verbal_Aptitude")) else row.get("Verbal_Aptitude"),
                None if pd.isna(row.get("Numerical_Aptitude")) else row.get("Numerical_Aptitude"),
                None if pd.isna(row.get("Spatial_Aptitude")) else row.get("Spatial_Aptitude"),
                None if pd.isna(row.get("Perceptual_Aptitude")) else row.get("Perceptual_Aptitude"),
                None if pd.isna(row.get("Manual_Dexterity")) else row.get("Manual_Dexterity"),
                None if pd.isna(row.get("date_taken")) else row.get("date_taken"),
            ))

        # ✅ Save to DB
        save_students(records)

        return {"message": f"✅ Saved {len(records)} records successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
