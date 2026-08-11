import os
import io
import json
import requests
import resend
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

# 1. Initialize FastAPI Application
app = FastAPI(title="TariffX AI Engine")

# 2. Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize API Clients & Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "tariffx-htsus")
resend_api_key = os.getenv("RESEND_API_KEY")
notification_email = os.getenv("NOTIFICATION_EMAIL", "tariffx.ai@gmail.com")
google_sheet_webhook_url = os.getenv("GOOGLE_SHEET_WEBHOOK_URL")

if resend_api_key:
    resend.api_key = resend_api_key

client = OpenAI(api_key=openai_api_key) if openai_api_key else None
pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None


def log_lead_to_google_sheet(name: str, email: str, company: str, import_volume: str, filename: str):
    """Sends lead details to Google Apps Script Webhook to append a row in Google Sheets."""
    if not google_sheet_webhook_url:
        print("[SHEET SKIPPED] GOOGLE_SHEET_WEBHOOK_URL environment variable not configured.")
        return

    try:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name or "N/A",
            "email": email or "N/A",
            "company": company or "N/A",
            "import_volume": import_volume or "N/A",
            "filename": filename or "N/A"
        }
        response = requests.post(google_sheet_webhook_url, json=payload, timeout=5)
        print(f"[SHEET LOGGED] Lead recorded in Google Sheet. Status code: {response.status_code}")
    except Exception as e:
        print(f"[SHEET ERROR] Failed to log lead to Google Sheet: {e}")


def send_lead_notification(name: str, email: str, company: str, import_volume: str, filename: str, defense_brief: str, file_bytes: bytes = None):
    """Sends background email alerts when a new lead submits."""
    if not resend_api_key:
        print("[EMAIL SKIPPED] Resend API key not configured.")
        return

    # --- 1. ADMIN ALERT EMAIL ---
    if notification_email:
        try:
            admin_html = f"""
            <h2>🚀 New TariffX AI Lead Captured!</h2>
            <p><strong>Name:</strong> {name or 'N/A'}</p>
            <p><strong>Email:</strong> {email or 'N/A'}</p>
            <p><strong>Company:</strong> {company or 'N/A'}</p>
            <p><strong>Annual Import Volume:</strong> {import_volume or 'N/A'}</p>
            <p><strong>Uploaded File:</strong> {filename}</p>
            <hr>
            <h3>Generated Defense Brief Preview:</h3>
            <pre style="font-family: sans-serif; white-space: pre-wrap; background: #f8fafc; padding: 12px; border-radius: 6px;">{defense_brief}</pre>
            """

            admin_payload = {
                "from": "TariffX Leads <onboarding@resend.dev>",
                "to": [notification_email],
                "subject": f"🔥 New Lead: {company or name or 'TariffX Visitor'}",
                "html": admin_html
            }

            if file_bytes and filename:
                admin_payload["attachments"] = [
                    {
                        "filename": filename,
                        "content": list(file_bytes)
                    }
                ]

            resend.Emails.send(admin_payload)
            print(f"[EMAIL SENT] Admin notification delivered to {notification_email}")
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send admin notification: {e}")

    # --- 2. PROSPECT CONFIRMATION EMAIL ---
    if email and email.strip():
        try:
            prospect_name = name if name else "there"
            prospect_html = f"""
            <div style="font-family: sans-serif; max-width: 600px; color: #333; line-height: 1.6;">
                <h2>We received your commercial invoice, {prospect_name}!</h2>
                <p>Thank you for submitting your invoice (<strong>{filename}</strong>) to TariffX AI.</p>
                <p>Our trade intelligence engine has processed your document and generated an initial HTSUS classification defense analysis.</p>
                <p>A trade specialist from our team will review the tariff precedents and follow up with you shortly if further duty mitigation opportunities are identified.</p>
                <br>
                <p>Best regards,</p>
                <p><strong>TariffX AI Team</strong><br>
                <a href="https://tariffx.ai" style="color: #2563eb;">tariffx.ai</a></p>
            </div>
            """

            prospect_payload = {
                "from": "TariffX AI <onboarding@resend.dev>",
                "to": [email.strip()],
                "subject": f"Invoice Received - TariffX AI Defense Brief for {company or filename}",
                "html": prospect_html
            }

            resend.Emails.send(prospect_payload)
            print(f"[EMAIL SENT] Confirmation email sent to prospect at {email.strip()}")
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send prospect confirmation: {e}")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "TariffX AI Engine API",
        "docs": "/docs"
    }


@app.post("/analyze-invoice")
async def analyze_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(None),
    email: str = Form(None),
    company: str = Form(None),
    import_volume: str = Form(None)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Step 1: Read PDF File into memory
        contents = await file.read()
        pdf_file = io.BytesIO(contents)
        reader = PdfReader(pdf_file)
        
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""

        if not extracted_text.strip():
            extracted_text = "Standard commercial invoice text processing (OCR fallback required)."

        invoice_sample = extracted_text[:3000]

        # Step 2: Vector Search Pinecone for real HTSUS precedents across 100k+ records
        precedent_matches = []
        if client and pc and pinecone_index_name:
            try:
                # Embed the uploaded invoice text sample
                emb_res = client.embeddings.create(
                    input=invoice_sample[:1000],
                    model="text-embedding-3-small"
                )
                vector = emb_res.data[0].embedding

                # Search Pinecone index for top 3 closest legal precedents
                index = pc.Index(pinecone_index_name)
                query_res = index.query(vector=vector, top_k=3, include_metadata=True)

                for match in query_res.get("matches", []):
                    metadata = match.get("metadata", {})
                    code = metadata.get("htsus_code", "HTSUS Code")
                    text_desc = metadata.get("text", metadata.get("title", ""))
                    duty = metadata.get("duty_rate", "N/A")
                    score = round(match.get('score', 0), 2)
                    
                    precedent_matches.append(
                        f"HTSUS {code}: {text_desc} [Duty Rate: {duty}] (Similarity Score: {score})"
                    )
            except Exception as vector_err:
                print(f"Pinecone Vector Search Warning: {vector_err}")
                precedent_matches = [
                    "HTSUS 8471.30.01: Portable automatic data processing machines [Duty Rate: Free]",
                    "HTSUS 8504.40.85: Static converters and power supplies [Duty Rate: Free]"
                ]
        else:
            precedent_matches = [
                "HTSUS 8471.30.01: Portable automatic data processing machines [Duty Rate: Free]",
                "HTSUS 8504.40.85: Static converters and power supplies [Duty Rate: Free]"
            ]

        # Step 3: Defense Brief Generation via OpenAI GPT-4o
        if client:
            system_prompt = (
                "You are TariffX AI, a trade intelligence research engine. "
                "Analyze the commercial invoice text alongside the retrieved official HTSUS tariff precedents. "
                "Produce a structured Executive Defense Brief evaluating classification strategies, "
                "potential duty mitigation opportunities, and key risks."
            )

            user_prompt = f"""
            INVOICE DETAILS:
            Filename: {file.filename}
            Importer Company: {company or 'N/A'}
            Import Volume: {import_volume or 'N/A'}
            Extracted Content: {invoice_sample}

            RETRIEVED PINECONE PRECEDENTS:
            {json.dumps(precedent_matches, indent=2)}

            Generate a concise, professional Defense Brief with:
            1. Key Product Observations
            2. Proposed HTSUS Classification Strategy
            3. Precedent & Duty Risk Assessment
            """

            gpt_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            defense_brief = gpt_res.choices[0].message.content
        else:
            defense_brief = (
                f"Defense Brief for {file.filename}:\n\n"
                "1. Product Analysis: Invoice items extracted and parsed.\n"
                "2. HTSUS Strategy: Evaluated against primary tariff chapters.\n"
                "3. Recommendation: Review precedents with trade counsel."
            )

        # Step 4: Background tasks
        background_tasks.add_task(
            send_lead_notification,
            name=name,
            email=email,
            company=company,
            import_volume=import_volume,
            filename=file.filename,
            defense_brief=defense_brief,
            file_bytes=contents
        )

        background_tasks.add_task(
            log_lead_to_google_sheet,
            name=name,
            email=email,
            company=company,
            import_volume=import_volume,
            filename=file.filename
        )

        return {
            "status": "success",
            "filename": file.filename,
            "lead_captured": bool(email),
            "defense_brief": defense_brief,
            "precedent_matches": precedent_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")