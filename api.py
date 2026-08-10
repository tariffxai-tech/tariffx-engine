import os
import io
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader
import resend

app = FastAPI(title="TariffX AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "tariffx-htsus")
resend_api_key = os.getenv("RESEND_API_KEY")
notification_email = os.getenv("NOTIFICATION_EMAIL") # Your email address

if resend_api_key:
    resend.api_key = resend_api_key

client = OpenAI(api_key=openai_api_key) if openai_api_key else None
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None


@app.get("/")
def read_root():
    return {"status": "online", "service": "TariffX AI Engine API"}


@app.post("/analyze-invoice")
async def analyze_invoice(
    file: UploadFile = File(...),
    name: str = Form(None),
    email: str = Form(None),
    company: str = Form(None),
    import_volume: str = Form(None)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""

        invoice_sample = extracted_text[:3000] if extracted_text.strip() else "PDF text extraction empty."

        # Vector Search
        precedent_matches = []
        if client and pc and pinecone_index_name:
            try:
                emb_res = client.embeddings.create(input=invoice_sample[:1000], model="text-embedding-3-small")
                vector = emb_res.data[0].embedding
                index = pc.Index(pinecone_index_name)
                query_res = index.query(vector=vector, top_k=3, include_metadata=True)
                for match in query_res.get("matches", []):
                    metadata = match.get("metadata", {})
                    code = metadata.get("htsus_code", "HTSUS Match")
                    title = metadata.get("title", match.get("id", ""))
                    precedent_matches.append(f"{code} - {title}")
            except Exception as e:
                print(f"Pinecone Warning: {e}")

        # Defense Brief
        if client:
            system_prompt = "You are TariffX AI. Generate a concise Defense Brief evaluating classification strategies."
            user_prompt = f"Invoice: {file.filename}\nCompany: {company}\nExtracted: {invoice_sample}"
            gpt_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            defense_brief = gpt_res.choices[0].message.content
        else:
            defense_brief = "Defense brief placeholder."

        # Send Lead Alert Email via Resend
        if resend_api_key and notification_email:
            try:
                resend.Emails.send({
                    "from": "TariffX AI <onboarding@resend.dev>",
                    "to": [notification_email],
                    "subject": f"🚨 New TariffX Lead: {company or name or 'New Submission'}",
                    "html": f"""
                        <h3>New Lead Captured on TariffX AI</h3>
                        <p><strong>Name:</strong> {name}</p>
                        <p><strong>Email:</strong> {email}</p>
                        <p><strong>Company:</strong> {company}</p>
                        <p><strong>Import Volume:</strong> {import_volume}</p>
                        <p><strong>Filename:</strong> {file.filename}</p>
                        <hr />
                        <h4>AI Defense Brief Preview:</h4>
                        <pre>{defense_brief}</pre>
                    """
                })
            except Exception as email_err:
                print(f"Email Dispatch Failed: {email_err}")

        return {
            "status": "success",
            "filename": file.filename,
            "lead_captured": bool(email),
            "defense_brief": defense_brief,
            "precedent_matches": precedent_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")