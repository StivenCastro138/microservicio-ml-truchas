"""
MICROSERVICIO ML TRUCHAS
========================
Features: days, temperature, ph, dissolved_oxygen
Targets:  lengths (cm), weights (g)

Deploy en Render.com (gratis):
  Build: pip install -r requirements.txt
  Start: uvicorn main:app --host 0.0.0.0 --port $PORT

Estructura:
  /
  ├── main.py
  ├── requirements.txt          ← fastapi uvicorn scikit-learn joblib numpy
  └── models/
      ├── modelo_ml_truchas.pkl
      ├── scaler_ml_truchas.pkl
      └── metadata_modelo.json
"""

import json, joblib, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="ML Truchas API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS_DIR = Path(__file__).parent / "models"

try:
    modelo   = joblib.load(MODELS_DIR / "modelo_ml_truchas.pkl")
    scaler   = joblib.load(MODELS_DIR / "scaler_ml_truchas.pkl")
    with open(MODELS_DIR / "metadata_modelo.json") as f:
        metadata = json.load(f)
    DISPONIBLE = True
    print("✅ Modelo ML cargado")
except FileNotFoundError as e:
    modelo = scaler = metadata = None
    DISPONIBLE = False
    print(f"⚠️ Modelo no encontrado: {e}")


class Entrada(BaseModel):
    dias_prediccion:   int   = Field(..., ge=1, le=1000)
    temperatura:       float = Field(..., ge=0,  le=40)
    ph:                float = Field(..., ge=0,  le=14)
    oxigeno_disuelto:  float = Field(..., ge=0,  le=20,
                                     description="Oxígeno disuelto (mg/L)")
    class Config:
        json_schema_extra = {"example": {
            "dias_prediccion": 80,
            "temperatura": 16.5,
            "ph": 7.2,
            "oxigeno_disuelto": 8.5
        }}


@app.post("/predecir")
async def predecir(e: Entrada):
    if not DISPONIBLE:
        raise HTTPException(503, "Modelo no disponible")
    try:
        gb_l, rf_l = modelo["gb_length"], modelo["rf_length"]
        gb_w, rf_w = modelo["gb_weight"], modelo["rf_weight"]
        p_gb, p_rf = modelo["peso_gb"],   modelo["peso_rf"]

        # Features: [days, temperature, ph, dissolved_oxygen]
        def inferir(dias):
            X  = np.array([[dias, e.temperatura, e.ph, e.oxigeno_disuelto]])
            Xs = scaler.transform(X)
            l  = float(p_gb * gb_l.predict(Xs)[0] + p_rf * rf_l.predict(Xs)[0])
            w  = float(p_gb * gb_w.predict(Xs)[0] + p_rf * rf_w.predict(Xs)[0])
            return round(l, 2), round(w, 2)

        # Día actual = día_prediccion - 1
        l_act, w_act = inferir(max(1, e.dias_prediccion - 1))
        l_pred, w_pred = inferir(e.dias_prediccion)

        # Importancia
        imp_l = p_gb * gb_l.feature_importances_ + p_rf * rf_l.feature_importances_
        imp_w = p_gb * gb_w.feature_importances_ + p_rf * rf_w.feature_importances_
        features = modelo["features"]
        importancia = {
            feat: {"longitud": round(float(il)*100, 2), "peso": round(float(iw)*100, 2)}
            for feat, il, iw in zip(features, imp_l, imp_w)
        }

        r2_l = metadata["metricas_test"]["longitud"]["r2"]
        r2_w = metadata["metricas_test"]["peso"]["r2"]

        return {
            "dias_prediccion":      e.dias_prediccion,
            "longitud_actual_cm":   l_act,
            "longitud_predicha_cm": l_pred,
            "crecimiento_longitud": round(l_pred - l_act, 3),
            "peso_actual_g":        w_act,
            "peso_predicho_g":      w_pred,
            "crecimiento_peso":     round(w_pred - w_act, 3),
            "confianza_longitud":   round(r2_l, 6),
            "confianza_peso":       round(r2_w, 6),
            "importancia_variables": importancia,
            "modelo_tipo":          f"Híbrido {int(p_gb*100)}% GB + {int(p_rf*100)}% RF",
            "variables_entrada": {
                "dias": e.dias_prediccion,
                "temperatura": e.temperatura,
                "ph": e.ph,
                "oxigeno_disuelto": e.oxigeno_disuelto,
            }
        }
    except Exception as ex:
        raise HTTPException(500, str(ex))


@app.get("/")
def root():
    return {"servicio": "ML Truchas", "disponible": DISPONIBLE, "docs": "/docs"}

@app.get("/status")
def status():
    if not DISPONIBLE:
        return {"disponible": False}
    return {
        "disponible": True,
        "features": metadata.get("features"),
        "metricas": metadata.get("metricas_test"),
    }