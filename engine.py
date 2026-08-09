import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("tariffx-customs")

def generate_text_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def query_pinecone_precedents(invoice_text, top_k=2):
    """Queries Pinecone for the most relevant Customs legal precedents."""
    vector = generate_text_embedding(invoice_text)
    
    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )
    
    precedents_str = ""
    for match in results.matches:
        meta = match.metadata
        precedents_str += f"\n- PRECEDENT MATCH [{meta['title']}] (HTS {meta['hts_code']}) (Confidence Score: {round(match.score, 3)}):\n  \"{meta['text']}\"\n"
        
    return precedents_str if precedents_str else "No direct rulings index matches found."

def generate_defense_brief(invoice_text):
    """Retrieves relevant precedents from Pinecone and generates a Defense Brief using GPT-4o."""
    print("Searching Pinecone for relevant U.S. Customs rulings...")
    precedents = query_pinecone_precedents(invoice_text)
    print("Found precedent matches!")

    prompt = f"""
    You are an elite U.S. Customs and Trade Compliance Attorney working for TariffX AI.
    Analyze the following commercial invoice data and precedent matches to generate a Customs Defense Brief.

    COMMERCIAL INVOICE DATA:
    {invoice_text}

    PRECEDENT RULINGS & MATCHES FROM PINECONE:
    {precedents}

    Generate a structured, professional 1-page Customs Defense Brief including:
    1. Executive Summary & Recommended HTS Code Classification
    2. Exact Legal Precedent Citations (cite the specific HQ/NY ruling numbers found in the precedents)
    3. Estimated Duty Rate vs Potential Section 301 Tariff Exposure
    4. Legal Defense Narrative under 19 U.S.C. § 1484 (Reasonable Care)
    5. Action Items for Customs Broker Filing
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a trade compliance research specialist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    print("Connecting to TariffX AI Engine...")
    
    sample_invoice = """
    COMMERCIAL INVOICE #10492
    Exporter: Shenzhen Industrial Components Ltd.
    Importer: USA Electronics Logistics Inc.
    Item: 5,000 units - Custom Molded Plastic Enclosures for Industrial Circuit Boards.
    Declared Value: $45,000 USD.
    Declared HTS: 3926.90.9988
    """
    
    brief = generate_defense_brief(sample_invoice)
    
    print("\n" + "=" * 60)
    print("          TARIFFX AI - CUSTOMS DEFENSE BRIEF OUTPUT          ")
    print("=" * 60)
    print(brief)
    print("=" * 60)