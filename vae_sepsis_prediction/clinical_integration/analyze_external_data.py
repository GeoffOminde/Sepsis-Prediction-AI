
import pandas as pd
import httpx
import asyncio
import io
import json

async def simulate_external_data_analysis():
    print("🏥 EXERNAL DATA ANALYSIS SIMULATION")
    print("------------------------------------")
    
    # 1. Create a "real" external CSV file with messy column names
    data = {
        'PID': ['P-101', 'P-102', 'P-103'],
        'Age': [72, 45, 81],
        'HR': [115, 72, 128],  # Synonym for heart_rate
        'Temp': [38.5, 36.6, 39.1], # Synonym for temperature
        'SBP': [95, 120, 88], # Synonym for systolic_bp
        'DBP': [60, 80, 55], # Synonym for diastolic_bp
        'Lac': [3.5, 1.1, 5.2], # Synonym for lactate
        'O2': [92, 98, 89], # Synonym for oxygen_saturation
        'Gender': ['male', 'female', 'male']
    }
    df = pd.DataFrame(data)
    
    # Convert 'female'/'male' to 0/1 for the simulation (though api handles it)
    df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
    
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    print(f"📦 Created external CSV with {len(df)} patient records.")
    print(f"📋 Columns: {list(df.columns)}")

    # 2. Upload to the AI Microservice
    url = "http://localhost:8000/v1/analyze/upload"
    
    print(f"\n🚀 Uploading to {url}...")
    
    files = {'file': ('external_data.csv', csv_buffer, 'text/csv')}
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, timeout=30.0)
            
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ ANALYSIS COMPLETE")
            print(f"📊 Summary:")
            print(f"   - Total Analyzed: {result['summary']['total_analyzed']}")
            print(f"   - High Risk Cases: {result['summary']['high_risk_count']}")
            print(f"   - Mean Sepsis Risk: {result['summary']['mean_risk']:.1%}")
            
            print(f"\n📋 Individual Patient Results:")
            for patient in result['results']:
                risk = patient['sepsis_risk']
                level = patient['risk_level']
                factors = [f['factor'] for f in patient['top_factors']]
                
                print(f"   ID: {patient['patient_id']}")
                print(f"   Risk: {risk:.1%} ({level})")
                print(f"   Top Factors: {', '.join(factors)}")
                print(f"   Rec: {patient['recommendation']}")
                print("-" * 30)
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {str(e)}")
        print("💡 Make sure the API is running: 'python clinical_integration/api.py'")

if __name__ == "__main__":
    asyncio.run(simulate_external_data_analysis())
