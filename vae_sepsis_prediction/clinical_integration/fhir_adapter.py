
import json
from typing import List, Dict, Any
import numpy as np

class FHIRAdapter:
    """
    Transforms HL7 FHIR Resources into VAE model input features
    """
    
    FEATURE_MAP = {
        'age': 'Patient.birthDate',
        'gender': 'Patient.gender',
        'heart_rate': '8867-4',
        'respiratory_rate': '9279-1',
        'temperature': '8310-5',
        'systolic_bp': '8480-6',
        'diastolic_bp': '8462-4',
        'oxygen_saturation': '2708-6',
        'wbc_count': '6690-2',
        'lactate': '2524-7',
        'creatinine': '2160-0',
        'platelet_count': '777-3',
        'bilirubin': '1975-2'
    }

    def __init__(self):
        # Default means for imputation (in production, use robust population statistics)
        self.defaults = {
            'heart_rate': 80.0,
            'respiratory_rate': 16.0,
            'temperature': 37.0,
            'systolic_bp': 120.0,
            'diastolic_bp': 80.0,
            'oxygen_saturation': 98.0,
            'wbc_count': 7.0,
            'lactate': 1.0,
            'creatinine': 1.0,
            'platelet_count': 250.0,
            'bilirubin': 0.5
        }

    def transform_bundle(self, fhir_bundle: Dict[str, Any]) -> List[float]:
        """
        Parses a FHIR Bundle (Patient + Observations) into 17 features
        """
        extracted = {}
        
        # 1. Extract Patient Demographics
        patient = next((e['resource'] for e in fhir_bundle.get('entry', []) if e['resource']['resourceType'] == 'Patient'), None)
        if patient:
            # Simple age calculation
            from datetime import date
            birth_year = int(patient['birthDate'].split('-')[0])
            extracted['age'] = date.today().year - birth_year
            extracted['gender'] = 1 if patient['gender'] == 'male' else 0
        else:
            extracted['age'] = 65.0
            extracted['gender'] = 0.5

        # 2. Extract Observations (LOINC codes)
        observations = [e['resource'] for e in fhir_bundle.get('entry', []) if e['resource']['resourceType'] == 'Observation']
        
        for obs in observations:
            loinc = obs.get('code', {}).get('coding', [{}])[0].get('code')
            val = obs.get('valueQuantity', {}).get('value')
            
            for feat, code in self.FEATURE_MAP.items():
                if loinc == code:
                    extracted[feat] = val

        # 3. Impute and Normalize
        features = []
        # Main clinical features
        for feat in ['age', 'gender', 'heart_rate', 'respiratory_rate', 'temperature', 
                    'systolic_bp', 'diastolic_bp', 'oxygen_saturation', 'wbc_count', 
                    'lactate', 'creatinine', 'platelet_count', 'bilirubin']:
            features.append(float(extracted.get(feat, self.defaults.get(feat, 0))))

        # 4. Computed Features (consistent with training)
        sbp = features[5]
        dbp = features[6]
        hr = features[2]
        
        map_val = (sbp + 2 * dbp) / 3
        pp = sbp - dbp
        si = hr / sbp if sbp > 0 else 0
        age_norm = features[0] / 100
        
        features.extend([map_val, pp, si, age_norm])
        
        return features

    def create_mock_fhir_bundle(self, patient_id: str) -> Dict[str, Any]:
        """Creates a mock FHIR Bundle for testing integration"""
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "gender": "male",
                        "birthDate": "1955-05-15"
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4"}]}, # HR
                        "valueQuantity": {"value": 112, "unit": "bpm"}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "2524-7"}]}, # Lactate
                        "valueQuantity": {"value": 4.2, "unit": "mmol/L"}
                    }
                }
            ]
        }

if __name__ == "__main__":
    adapter = FHIRAdapter()
    bundle = adapter.create_mock_fhir_bundle("PAT-001")
    features = adapter.transform_bundle(bundle)
    print(f"✓ FHIR Bundle transformed to {len(features)} features:")
    print(features)
