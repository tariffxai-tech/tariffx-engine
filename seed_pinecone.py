import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 1. Load keys from .env
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("Missing API keys in .env file.")

openai_client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)

INDEX_NAME = "tariffx-customs"

# 2. Ensure Pinecone Index Exists
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, # Matches text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# 3. Precedent Rulings Dataset (CROSS Legal Precedents)
sample_rulings = [
    {
        "id": "HQ_H302149",
        "hts_code": "3926.90.9988",
        "title": "HQ H302149: Classification of Molded Plastic Enclosures",
        "text": "HQ H302149 (2019): Molded plastic housings designed exclusively to protect printed circuit boards in industrial automation units are classified under 3926.90.9988. Heading 8538 does not apply as the item lacks electrical contacts or connector pins."
    },
    {
        "id": "NY_N310452",
        "hts_code": "8504.40.9580",
        "title": "NY N310452: Classification of Switching Power Supplies",
        "text": "NY N310452 (2020): Static converters and switching power supplies imported for use with telecommunication apparatus are classified under HTS sub-heading 8504.40.9580. Section 301 Exclusion List applies if imported under secondary subheading 9903.88.03."
    },
    {
        "id": "HQ_H295831",
        "hts_code": "8544.42.9000",
        "title": "HQ H295831: Classification of Custom Wiring Harnesses",
        "text": "HQ H295831 (2018): Insulated electric conductors fitted with modular connectors designed for DC voltage under 1,000V are classified under 8544.42.9000. Subject to Section 301 duties unless origin can be established outside of China under Substantial Transformation rules."
    }
]

def generate_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def seed_database():
    print(f"Seeding '{INDEX_NAME}' index with legal precedent rulings...\n")
    vectors_to_upsert = []

    for item in sample_rulings:
        print(f"-> Generating vector for Ruling {item['id']} ({item['hts_code']})...")
        embedding = generate_embedding(item["text"])
        
        vectors_to_upsert.append({
            "id": item["id"],
            "values": embedding,
            "metadata": {
                "hts_code": item["hts_code"],
                "title": item["title"],
                "text": item["text"]
            }
        })

    print("\nUploading vectors to Pinecone...")
    index.upsert(vectors=vectors_to_upsert)
    print("Success! Pinecone database populated with precedent rulings.")

if __name__ == "__main__":
    seed_database()