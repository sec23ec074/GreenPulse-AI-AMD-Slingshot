from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import numpy as np
import io

app = FastAPI(title="GreenPulse AI - AMD Slingshot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMISSION_FACTOR = 0.82

@app.get("/")
def home():
    return {"message": "GreenPulse AI Running"}

@app.post("/upload-energy/")
async def upload_energy(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df["carbon"] = df["kwh"] * EMISSION_FACTOR
    return {
        "total_carbon": round(df["carbon"].sum(),2),
        "average_carbon": round(df["carbon"].mean(),2)
    }

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"date":"ds","kwh":"y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[["ds","yhat"]].tail(30).to_dict(orient="records")

@app.post("/ml-recommend/")
async def ml_recommend(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    X = np.arange(len(df)).reshape(-1,1)
    y = df["kwh"].values
    model = LinearRegression()
    model.fit(X,y)
    trend = model.coef_[0]
    predicted_next = model.predict([[len(df)]])[0]
    carbon_next = predicted_next * EMISSION_FACTOR
    if trend > 15:
        reduction = 15
        message = "Strong upward energy trend detected."
    elif trend > 5:
        reduction = 8
        message = "Moderate increase detected."
    else:
        reduction = 3
        message = "Energy stable."
    savings = (predicted_next * reduction/100) * EMISSION_FACTOR
    return {
        "trend": round(float(trend),2),
        "predicted_next_kwh": round(float(predicted_next),2),
        "predicted_carbon": round(float(carbon_next),2),
        "recommended_reduction_percent": reduction,
        "estimated_savings": round(float(savings),2),
        "message": message
    }

@app.get("/leaderboard/")
def leaderboard():
    return [
        {"department":"CSE","score":88},
        {"department":"ECE","score":75},
        {"department":"Mechanical","score":63}
    ]
