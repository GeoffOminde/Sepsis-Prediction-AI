
import logging
import hashlib
import json
import time
from pathlib import Path

class ClinicalAuditLogger:
    """
    HIPAA-aware audit logger.
    - Hashes Patient IDs
    - Records inputs, outputs, timestamps, and model versions
    """
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.audit_file = self.log_dir / "clinical_decisions.jsonl"
        
        # Internal logger for system events
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("Audit")

    def _hash_id(self, patient_id: str) -> str:
        """Protect PHI by hashing the patient ID"""
        return hashlib.sha256(patient_id.encode()).hexdigest()

    def log_prediction(self, patient_id: str, input_features: list, response: dict):
        """Log the complete clinical decision context"""
        audit_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "patient_hash": self._hash_id(patient_id),
            "model_version": "VAE-Sepsis-v1.0.0",
            "prediction": response,
            "input_hash": hashlib.sha256(str(input_features).encode()).hexdigest()
        }
        
        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
        
        self.logger.info(f"✓ Audited prediction for patient {patient_id[:4]}... (hash: {audit_entry['patient_hash'][:8]})")

if __name__ == "__main__":
    logger = ClinicalAuditLogger()
    logger.log_prediction("PAT-999", [1.0]*17, {"risk": 0.85})
    print(f"✓ Audit log created at {logger.audit_file}")
