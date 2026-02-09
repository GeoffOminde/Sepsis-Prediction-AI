import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path
import logging
import time
import json
import pandas as pd
import io

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from models.sepsis_vae import SepsisVAE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinicalAPI")

app = FastAPI(
    title="🏥 Sepsis VAE Clinical Microservice",
    description="Generative AI for batch analysis of external EHR data (CSV/JSON/Excel)",
    version="1.1.0"
)

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "vae_output" / "deployment" / "sepsis_vae_model.pth"
FEATURE_NAMES = [
    'age', 'gender', 'heart_rate', 'respiratory_rate', 'temperature', 
    'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'wbc_count', 
    'lactate', 'creatinine', 'platelet_count', 'bilirubin',
    'map', 'pulse_pressure', 'shock_index', 'age_norm'
]

# Core features needed from external sources
REQUIRED_CORE_FEATURES = [
    'age', 'gender', 'heart_rate', 'respiratory_rate', 'temperature', 
    'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'wbc_count', 
    'lactate', 'creatinine', 'platelet_count', 'bilirubin'
]

# Feature synonyms for automatic mapping
SYNONYMS = {
    'heart_rate': ['hr', 'heartrate', 'pulse', 'beats'],
    'respiratory_rate': ['rr', 'resprate', 'breaths', 'resp'],
    'temperature': ['temp', 't', 'bodytemp'],
    'systolic_bp': ['sbp', 'bp_sys', 'systolic'],
    'diastolic_bp': ['dbp', 'bp_dia', 'diastolic'],
    'oxygen_saturation': ['spo2', 'o2sat', 'saturation', 'o2'],
    'wbc_count': ['wbc', 'whitecells', 'leukocytes'],
    'lactate': ['lactic_acid', 'lac'],
    'creatinine': ['crea', 'cr'],
    'platelet_count': ['plt', 'platelets'],
    'bilirubin': ['bili', 'total_bilirubin']
}

# Global model instance
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if not MODEL_PATH.exists():
            logger.error(f"Model not found at {MODEL_PATH}")
            return
            
        checkpoint = torch.load(MODEL_PATH)
        config = checkpoint['model_config']
        
        model = SepsisVAE(
            input_dim=config['input_dim'],
            latent_dim=config['latent_dim'],
            hidden_dims=config['hidden_dims']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✓ Sepsis VAE Model loaded successfully for clinical use")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

class PatientRecord(BaseModel):
    patient_id: str
    data: Dict[str, Any]

class BatchAnalysisRequest(BaseModel):
    records: List[PatientRecord]

class IndividualResult(BaseModel):
    patient_id: str
    sepsis_risk: float
    uncertainty: float
    risk_level: str
    recommendation: str
    top_factors: List[Dict[str, float]]

class BatchResponse(BaseModel):
    results: List[IndividualResult]
    summary: Dict[str, Any]
    status: str

def get_risk_level(risk: float) -> str:
    if risk < 0.3: return "Low"
    if risk < 0.6: return "Moderate"
    if risk < 0.8: return "High"
    return "Critical"

def get_clinical_recommendation(risk_level: str, uncertainty: float) -> str:
    if uncertainty > 0.3:
        return "⚠ HIGH UNCERTAINTY: Model confidence is low. Prioritize clinical assessment."
    if risk_level == "Critical":
        return "❗ IMMEDIATE ACTION: Initiate sepsis bundle."
    if risk_level == "High":
        return "🟡 URGENT: Consider blood cultures."
    return "🟢 STABLE: Low risk."

def map_features(external_data: Dict[str, Any]) -> List[float]:
    """Map external dictionary keys to ordered model features"""
    mapped = {}
    
    # Normalize keys to lowercase for matching
    external_data = {k.lower(): v for k, v in external_data.items()}
    
    for core_feat in REQUIRED_CORE_FEATURES:
        # 1. Direct match
        if core_feat in external_data:
            mapped[core_feat] = external_data[core_feat]
            continue
            
        # 2. Synonym match
        found = False
        for syn in SYNONYMS.get(core_feat, []):
            if syn in external_data:
                mapped[core_feat] = external_data[syn]
                found = True
                break
        
        if not found:
            # Impute or default (using training means)
            defaults = {'gender': 0.5, 'heart_rate': 80, 'temperature': 37, 'lactate': 1.5}
            mapped[core_feat] = defaults.get(core_feat, 0.0)
            
    # Calculate derived features
    features = [float(mapped[f]) for f in REQUIRED_CORE_FEATURES]
    sbp, dbp, hr, age = features[5], features[6], features[2], features[0]
    
    map_val = (sbp + 2 * dbp) / 3
    pp = sbp - dbp
    si = hr / sbp if sbp > 0 else 0
    age_norm = age / 100
    
    features.extend([map_val, pp, si, age_norm])
    return features

def explain_prediction(model: SepsisVAE, features: torch.Tensor) -> List[Dict[str, float]]:
    features.requires_grad = True
    _, _, _, sepsis_risk = model(features)
    sepsis_risk.backward()
    importances = features.grad.abs().squeeze().numpy()
    factors = []
    for i, name in enumerate(FEATURE_NAMES):
        factors.append({"factor": name, "importance": float(importances[i])})
    return sorted(factors, key=lambda x: x['importance'], reverse=True)[:3]

async def analyze_records(records: List[Dict[str, Any]], ids: List[str]) -> BatchResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    results = []
    risk_stats = []
    
    for record, pid in zip(records, ids):
        try:
            features = map_features(record)
            features_tensor = torch.FloatTensor([features])
            
            # Predict
            pred_dict = model.predict_sepsis(features_tensor, n_samples=10)
            risk = float(pred_dict['risk_mean'].item())
            uncertainty = float(pred_dict['risk_std'].item())
            risk_level = get_risk_level(risk)
            
            # Explain
            top_factors = explain_prediction(model, features_tensor)
            
            results.append(IndividualResult(
                patient_id=pid,
                sepsis_risk=risk,
                uncertainty=uncertainty,
                risk_level=risk_level,
                recommendation=get_clinical_recommendation(risk_level, uncertainty),
                top_factors=top_factors
            ))
            risk_stats.append(risk)
            
        except Exception as e:
            logger.error(f"Error analyzing record for {pid}: {str(e)}")

    summary = {
        "total_analyzed": len(results),
        "mean_risk": float(sum(risk_stats)/len(risk_stats)) if risk_stats else 0,
        "high_risk_count": len([r for r in results if r.risk_level in ["High", "Critical"]]),
        "mean_uncertainty": float(sum([r.uncertainty for r in results])/len(results)) if results else 0
    }
    
    return BatchResponse(results=results, summary=summary, status="SUCCESS")

@app.post("/v1/analyze/upload", response_model=BatchResponse)
async def upload_file(file: UploadFile = File(...)):
    """Accepts CSV or Excel files for batch analysis"""
    contents = await file.read()
    
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(contents))
    elif file.filename.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")

    # Expecting 'patient_id' or 'id' column
    id_col = next((c for c in df.columns if c.lower() in ['patient_id', 'id', 'pid']), None)
    if not id_col:
        df['temp_id'] = [f"Patient_{i}" for i in range(len(df))]
        id_col = 'temp_id'
    
    records = df.to_dict(orient='records')
    pids = [str(r[id_col]) for r in records]
    
    return await analyze_records(records, pids)

@app.post("/v1/analyze/json", response_model=BatchResponse)
async def analyze_json(request: BatchAnalysisRequest):
    """Accepts bulk JSON data for analysis"""
    records = [r.data for r in request.records]
    pids = [r.patient_id for r in request.records]
    return await analyze_records(records, pids)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_ready": model is not None, "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
