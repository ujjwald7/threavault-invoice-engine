from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from fraud_detection_system import InvoiceFraudDetectionSystem

app = FastAPI()

# Initialize the fraud detection system
HISTORICAL_INDEX_PATH = "path_to_historical_index.faiss"
VENDOR_INDEX_PATH = "path_to_vendor_index.faiss"
fraud_detection_system = InvoiceFraudDetectionSystem(HISTORICAL_INDEX_PATH, VENDOR_INDEX_PATH)

# Pydantic model for text-based analysis
class TextAnalysisRequest(BaseModel):
    analysis_text: str

@app.post("/analyze-invoice")
async def analyze_invoice(file: UploadFile = File(...)):
    """Analyze an uploaded invoice image for anomalies."""
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
        
        analysis, anomalies = fraud_detection_system.analyze_invoice(temp_file_path)
        os.remove(temp_file_path)  # Clean up the temporary file

        return {
            "analysis": analysis,
            "anomalies": anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing invoice: {str(e)}")

@app.post("/detect-altered-text")
async def detect_altered_text(request: TextAnalysisRequest):
    """Detect altered text anomalies in the provided analysis."""
    try:
        is_altered = fraud_detection_system.detect_altered_text(request.analysis_text)
        return {"altered_text_detected": is_altered}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting altered text: {str(e)}")

@app.post("/detect-inconsistent-formatting")
async def detect_inconsistent_formatting(request: TextAnalysisRequest):
    """Detect inconsistent formatting anomalies in the provided analysis."""
    try:
        inconsistent_formatting = fraud_detection_system.detect_inconsistent_formatting(request.analysis_text)
        return {"inconsistent_formatting_detected": inconsistent_formatting}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting inconsistent formatting: {str(e)}")

@app.post("/detect-suspicious-vendor")
async def detect_suspicious_vendor(request: TextAnalysisRequest):
    """Detect suspicious vendor information in the provided analysis."""
    try:
        is_suspicious = fraud_detection_system.detect_suspicious_vendor(request.analysis_text)
        return {"suspicious_vendor_detected": is_suspicious}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting suspicious vendor: {str(e)}")

@app.post("/compare-historical-patterns")
async def compare_historical_patterns(request: TextAnalysisRequest):
    """Compare the provided analysis with historical patterns."""
    try:
        is_deviating = fraud_detection_system.compare_historical_patterns(request.analysis_text)
        return {"historical_deviation_detected": is_deviating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing historical patterns: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
