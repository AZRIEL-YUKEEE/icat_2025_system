from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import mysql.connector
from io import BytesIO

app = FastAPI(title="ICAT Student Data Uploader API")

# ----------------- ENABLE CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------- DATABASE CONNECTION -----------------
def get_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",   # change if needed
        password="",   # change if needed
        database="icat_db"
    )

# ----------------- SAVE STUDENT RECORD -----------------
def save_students(records):
    conn = get_connection()
    cur = conn.cursor()
    sql = """INSERT INTO students 
             (application_number, family_name, first_name, middle_name, sex, strand, course,
              general_ability, verbal_aptitude, numerical_aptitude, spatial_aptitude,
              perceptual_aptitude, manual_dexterity, date_taken)
             VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    cur.executemany(sql, records)  # bulk insert
    conn.commit()
    cur.close()
    conn.close()

# ----------------- API ENDPOINTS -----------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df_upload = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df_upload = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Use CSV or Excel.")

        # Ensure date_taken column exists
        if "date_taken" not in df_upload.columns:
            df_upload["date_taken"] = "2025-01-01"
        else:
            df_upload["date_taken"] = pd.to_datetime(
                df_upload["date_taken"], errors="coerce"
            ).dt.strftime("%Y-%m-%d").fillna("2025-01-01")

        # Replace NaN values
        score_columns = [
            "General_Ability", "Verbal_Aptitude", "Numerical_Aptitude",
            "Spatial_Aptitude", "Perceptual_Aptitude", "Manual_Dexterity"
        ]
        for col in score_columns:
            if col in df_upload.columns:
                df_upload[col] = df_upload[col].fillna(0)

        df_upload = df_upload.where(pd.notnull(df_upload), None)

        # Return preview only
        preview = df_upload.head().to_dict(orient="records")
        return {"message": "File processed successfully", "preview": preview}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df_upload = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith(".xlsx") or file.filename.endswith(".xls"):
            df_upload = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Use CSV or Excel.")

        if "date_taken" not in df_upload.columns:
            df_upload["date_taken"] = "2025-01-01"
        else:
            df_upload["date_taken"] = pd.to_datetime(
                df_upload["date_taken"], errors="coerce"
            ).dt.strftime("%Y-%m-%d").fillna("2025-01-01")

        score_columns = [
            "General_Ability", "Verbal_Aptitude", "Numerical_Aptitude",
            "Spatial_Aptitude", "Perceptual_Aptitude", "Manual_Dexterity"
        ]
        for col in score_columns:
            if col in df_upload.columns:
                df_upload[col] = df_upload[col].fillna(0)

        df_upload = df_upload.where(pd.notnull(df_upload), None)

        # Build records
        records = []
        for _, row in df_upload.iterrows():
            records.append((
                row['Application Number'], row['Family_Name'], row['First_Name'],
                row['Middle_Name'], row['Sex'], row['Strand'], row['Course'],
                row['General_Ability'], row['Verbal_Aptitude'], row['Numerical_Aptitude'],
                row['Spatial_Aptitude'], row['Perceptual_Aptitude'], row['Manual_Dexterity'],
                row['date_taken']
            ))

        save_students(records)
        return {"message": "All uploaded data saved successfully to the database!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
