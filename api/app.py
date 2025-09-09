from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, text
import pandas as pd
import redis
import json
import hashlib
import mlflow
import os

# ------------------ CONFIG ------------------
BASE_DIR = "/home/bishnu-upadhyay/projects/MLops"

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_URI = "models:/Bank_XGB_Model@Production"

# Load model
try:
    print("Loading model from MLflow...")
    model = mlflow.xgboost.load_model(MODEL_URI)
    print("Model loaded.")
except Exception as e:
    raise RuntimeError(f"Failed to load model from MLflow: {e}")

# Redis
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_TTL_SECONDS = 24 * 3600
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# Database
DB_URL = "mysql+pymysql://bishnu:bishnu%40pass123@localhost:3306/bank"
TABLE_NAME = "new_bank_data"
engine = create_engine(DB_URL)
metadata = MetaData()

# Table schema (fixed)
new_bank_data_table = Table(
    TABLE_NAME,
    metadata,
    Column('age', Integer),
    Column('job', Float),
    Column('marital', Float),
    Column('education', Float),
    Column('campaign', Float),
    Column('previous', Float),
    Column('emp_var_rate', Float),
    Column('cons_price_idx', Float),
    Column('cons_conf_idx', Float),
    Column('euribor3m', Float),
    Column('nr_employed', Float),
    Column('contact_flag', Float),
    Column('default_flag', Float),
    Column('housing_flag', Float),
    Column('loan_flag', Float),
    Column('y', Integer),
)
metadata.create_all(engine)

# ------------------ FASTAPI ------------------
app = FastAPI(title="Bank Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Input schema ------------------
class InputData(BaseModel):
    age: int = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    campaign: int = Field(ge=0, le=100)
    previous: int
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float

# ------------------ UTILS ------------------
def make_cache_key(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True)
    return "pred:" + hashlib.sha256(s.encode()).hexdigest()

def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Map binary columns
    binary_mappings = {
        'contact': lambda x: 1 if x == 'cellular' else 0,
        'default': lambda x: 1 if x == 'yes' else 0,
        'housing': lambda x: 1 if x == 'yes' else 0,
        'loan': lambda x: 1 if x == 'yes' else 0
    }
    for col, func in binary_mappings.items():
        df[f"{col}_flag"] = df[col].apply(func)
    df.drop(columns=binary_mappings.keys(), inplace=True)

    # Simple categorical encoding (ordinal)
    categorical_cols = ['job', 'marital', 'education']
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # Ensure numeric types
    numeric_cols = ['age', 'campaign', 'previous', 'emp_var_rate',
                    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
    df[numeric_cols] = df[numeric_cols].astype(float)
    df['age'] = df['age'].astype(int)

    return df

# ------------------ ROUTES ------------------
@app.get("/")
def root():
    return {"message": "Bank Prediction API running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        payload = data.dict()
        key = make_cache_key(payload)

        # Check Redis cache
        cached = r.get(key)
        if cached:
            result = json.loads(cached)
            result["cached"] = True
            # Map 0/1 to no/yes for response
            result["prediction"] = "yes" if result["prediction"] == 1 else "no"
            return result

        # Preprocess input
        df = preprocess_input(payload)

        # Predict
        prob = float(model.predict_proba(df)[:, 1][0])
        pred = int(prob >= 0.5)
        result = {
            "prediction": "yes" if pred == 1 else "no",  # Map here
            "probability": prob,
            "cached": False
        }

        # Cache in Redis (store 0/1 internally)
        r.setex(key, CACHE_TTL_SECONDS, json.dumps({
            **df.iloc[0].to_dict(),
            "prediction": pred,
            "probability": prob
        }))

        # Prepare data to save in DB
        db_data = df.iloc[0].to_dict()
        db_data['y'] = pred  # store 0/1 in DB
        with engine.begin() as conn:
            conn.execute(new_bank_data_table.insert().values(db_data))

        return result

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
