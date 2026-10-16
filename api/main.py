import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API de Previsão de Churn")

modelo = joblib.load("models/modelo_churn.joblib")
LIMIAR = 0.3


class Cliente(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def inicio():
    return {"mensagem": "API de churn no ar"}


@app.post("/prever")
def prever(cliente: Cliente):
    dados = pd.DataFrame([cliente.model_dump()])
    probabilidade = float(modelo.predict_proba(dados)[0, 1])
    return {
        "probabilidade_cancelar": round(probabilidade, 3),
        "vai_cancelar": probabilidade >= LIMIAR,
        "limiar": LIMIAR,
    }