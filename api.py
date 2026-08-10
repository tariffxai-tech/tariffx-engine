from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS

app = FastAPI(title="TariffX AI Engine")

# <-- ADD THIS CORS BLOCK BELOW app = FastAPI() -->
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from parser import extract_invoice_text
from engine import generate_defense_brief

app = FastAPI(
    title="TariffX AI Engine API",
    description="API for processing commercial invoices and generating Customs Defense Briefs.",
    version="1.0.0"
)

# Enable CORS so Framer or web frontends can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "TariffX AI Engine API is running."}

@app.post("/analyze-invoice")
async def analyze_invoice(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_pdf_path = f"temp_{file.filename}"
    
    try:
        # Save uploaded PDF temporarily
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract text using parser.py
        invoice_text = extract_invoice_text(temp_pdf_path)
        
        if not invoice_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            
        # Run TariffX AI RAG engine
        brief_output = generate_defense_brief(invoice_text)
        
        return {
            "success": True,
            "filename": file.filename,
            "defense_brief": brief_output
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)