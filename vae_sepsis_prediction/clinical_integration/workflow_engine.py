
import time
import json
import httpx
import sys
from pathlib import Path
from fhir_adapter import FHIRAdapter
from audit_logger import ClinicalAuditLogger

class SepsisWorkflowEngine:
    """
    Simulates the clinical workflow:
    EHR Event -> FHIR Transform -> AI Prediction -> Audit -> Alerting
    """
    def __init__(self, api_url: str = "http://localhost:8000/v1/predict/sepsis"):
        self.api_url = api_url
        self.adapter = FHIRAdapter()
        self.audit = ClinicalAuditLogger()
        print("🚀 Sepsis Workflow Engine Initialized")

    async def process_new_ehr_data(self, fhir_bundle: dict):
        """
        Main entry point for new clinical data
        """
        patient_id = next(
            (e['resource']['id'] for e in fhir_bundle['entry'] 
             if e['resource']['resourceType'] == 'Patient'), 
            "Unknown"
        )
        
        print(f"\n[1/4] 📥 Received HL7 FHIR Bundle for Patient: {patient_id}")
        
        # 1. Transform FHIR to Features
        features = self.adapter.transform_bundle(fhir_bundle)
        print(f"[2/4] 🔄 Transformed FHIR to 17 model features")

        # 2. Call AI Microservice
        print(f"[3/4] 🧠 Calling Sepsis VAE Prediction API...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    json={"patient_id": patient_id, "features": features},
                    timeout=10.0
                )
                
            if response.status_code == 200:
                result = response.json()
                
                # 3. Audit the Decision
                self.audit.log_prediction(patient_id, features, result)
                
                # 4. Handle Alerting
                self._trigger_clinical_alerts(result)
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Workflow Error: {str(e)}")

    def _trigger_clinical_alerts(self, prediction: dict):
        """Dispatches alerts based on risk level"""
        risk = prediction['sepsis_risk']
        level = prediction['risk_level']
        factors = [f['factor'] for f in prediction['top_contributing_factors']]
        
        print(f"\n[4/4] 📢 CLINICAL ALERT TRIGGERED")
        print(f"      DRIVE: {prediction['recommendation']}")
        print(f"      RISK: {risk:.1%} ({level})")
        print(f"      TOP FACTORS: {', '.join(factors)}")
        print(f"      UNCERTAINTY: ±{prediction['uncertainty']:.2%}")
        
        if level in ["High", "Critical"]:
            print("      🚨 DISPATCHING ALERT TO RAPID RESPONSE TEAM...")
        elif level == "Moderate":
            print("      🔔 ADDING TO CHARGE NURSE WATCHLIST...")
        else:
            print("      ✅ NO ALERT REQUIRED.")

async def run_simulation():
    engine = SepsisWorkflowEngine()
    adapter = FHIRAdapter()
    
    # Simulate a high-risk patient
    print("\n--- SIMULATING HIGH-RISK CLINICAL EVENT ---")
    high_risk_bundle = adapter.create_mock_fhir_bundle("PAT-667")
    # Manually tweak bundle for high risk in simulation
    for entry in high_risk_bundle['entry']:
        if entry['resource'].get('code', {}).get('coding', [{}])[0].get('code') == '2524-7': # Lactate
            entry['resource']['valueQuantity']['value'] = 6.8
        if entry['resource'].get('code', {}).get('coding', [{}])[0].get('code') == '8867-4': # HR
            entry['resource']['valueQuantity']['value'] = 135

    await engine.process_new_ehr_data(high_risk_bundle)

if __name__ == "__main__":
    import asyncio
    print("NOTE: Ensure api.py is running on localhost:8000")
    try:
        asyncio.run(run_simulation())
    except Exception as e:
        print(f"Could not connect to API. Run 'python api.py' in a separate terminal.")
