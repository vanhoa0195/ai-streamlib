import os
import json
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import AzureOpenAI

# Create ChromaDB client
client_chroma = chromadb.Client()

# Initialize AzureOpenAI client for embeddings
client = AzureOpenAI(
    api_version="2024-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def load_vna_data(json_path):
    """Load and flatten VNA data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    metadatas = []

    # Process airline info
    airline_info = data.get('airline_info', {})
    documents.append(f"Airline Information: {airline_info.get('description', '')}")
    metadatas.append({"category": "airline_info", "source": airline_info.get('source', '')})

    # Process routes
    for route in data.get('routes', []):
        doc = f"Route {route.get('title', '')}: {route.get('description', '')}. Frequency: {route.get('frequency', '')}. Airports: {route.get('departure_airport', '')} to {route.get('arrival_airport', '')}"
        documents.append(doc)
        metadatas.append({"category": "routes", "id": route.get('id', ''), "source": route.get('source', '')})

    # Process fares
    for fare in data.get('fares', []):
        benefits = ", ".join(fare.get('benefits', []))
        doc = f"Fare {fare.get('name', '')}: {fare.get('description', '')}. Benefits: {benefits}"
        documents.append(doc)
        metadatas.append({"category": "fares", "id": fare.get('id', ''), "source": fare.get('source', '')})

    # Process promotions
    for promo in data.get('current_promotions', []):
        doc = f"Promotion {promo.get('title', '')}: {promo.get('description', '')}. Terms: {promo.get('terms', '')}"
        documents.append(doc)
        metadatas.append({"category": "promotions", "id": promo.get('id', ''), "source": promo.get('source', '')})

    # Process ancillary products
    for product in data.get('ancillary_products', []):
        doc = f"Service {product.get('name', '')}: {product.get('description', '')}"
        if 'price_note' in product:
            doc += f". {product['price_note']}"
        if 'note' in product:
            doc += f". Note: {product['note']}"
        documents.append(doc)
        metadatas.append({"category": "products", "id": product.get('id', ''), "source": product.get('source', '')})

    # Process procedures
    for proc in data.get('procedures', []):
        doc = f"Procedure {proc.get('name', '')}: {proc.get('description', '')}"
        if 'steps' in proc:
            doc += f". Steps: {', '.join(proc['steps'])}"
        if 'details' in proc:
            details = proc['details']
            doc += f". Details: {json.dumps(details)}"
        documents.append(doc)
        metadatas.append({"category": "procedures", "id": proc.get('id', ''), "source": proc.get('source', '')})

    # Process flights
    for flight in data.get('flights', []):
        doc = (
            f"Flight {flight.get('flight_number', '')} from {flight.get('origin', '')} to {flight.get('destination', '')} "
            f"on {flight.get('departure_date', '')}. "
            f"Departure: {flight.get('departure_datetime', '')}, Arrival: {flight.get('arrival_datetime', '')}, "
            f"Duration: {flight.get('duration', '')}, Aircraft: {flight.get('aircraft', '')}, "
            f"Status: {flight.get('status', '')}. "
            f"Fares - Economy: {flight.get('fare_economy', '')}, "
            f"Premium Economy: {flight.get('fare_premium_economy', '')}, "
            f"Business: {flight.get('fare_business', '')}"
        )
        documents.append(doc)
        metadatas.append({
            "category": "flights",
            "id": flight.get('id', ''),
            "flight_number": flight.get('flight_number', ''),
            "origin": flight.get('origin', ''),
            "destination": flight.get('destination', ''),
            "departure_date": flight.get('departure_date', ''),
            "departure_datetime": flight.get('departure_datetime', ''),
            "source": flight.get('source', '')
        })

    return documents, metadatas

def load_vna_procedures(json_path):
    """Load Vietnamese procedures from procedures.json file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        procedures = json.load(f)

    documents = []
    metadatas = []

    # Process each procedure
    for proc in procedures:
        # Create a comprehensive document from the procedure
        doc = f"{proc.get('title', '')}: {proc.get('content', '')}"

        # Build metadata
        meta = {
            "category": proc.get('category', ''),
            "id": proc.get('id', ''),
            "language": proc.get('metadata', {}).get('language', 'vi'),
            "source": proc.get('metadata', {}).get('source', ''),
            "type": proc.get('metadata', {}).get('type', '')
        }

        documents.append(doc)
        metadatas.append(meta)

    return documents, metadatas

def setup_vna_collection():
    """Set up ChromaDB collection with VNA data from multiple sources."""
    # Create collection with Azure OpenAI embeddings
    collection = client_chroma.create_collection(
        name="vietnam_airlines_info",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
        ),
    )

    current_dir = os.path.dirname(__file__)

    # Load English VNA data
    vna_json_path = os.path.join(current_dir, "flights_with_dates.json")
    en_documents, en_metadatas = load_vna_data(vna_json_path)

    # Load Vietnamese procedures
    procedures_json_path = os.path.join(current_dir, "procedures.json")
    vi_documents, vi_metadatas = load_vna_procedures(procedures_json_path)

    # Combine all documents and metadatas
    all_documents = en_documents + vi_documents
    all_metadatas = en_metadatas + vi_metadatas

    # Add all documents to collection with unique IDs
    collection.add(
        documents=all_documents,
        metadatas=all_metadatas,
        ids=[f"vna_doc_{i}" for i in range(len(all_documents))]
    )

    print(f"Loaded {len(en_documents)} English documents and {len(vi_documents)} Vietnamese procedures into ChromaDB")

    return collection

def query_vna_info(collection, query, n_results=10):
    """Query the VNA collection for relevant information."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    # Format results into a coherent response
    if results and results['documents']:
        relevant_docs = results['documents'][0]  # Get the first query's results
        relevant_metadata = results['metadatas'][0]
        
        # Combine the information into a structured response
        response_parts = []
        for doc, meta in zip(relevant_docs, relevant_metadata):
            category = meta.get('category', 'information')
            response_parts.append(f"{doc}")
        
        return "\n".join(response_parts)
    
    return "I couldn't find specific information about that. Please contact Vietnam Airlines directly for more details."

if __name__ == "__main__":
    # Test the setup
    collection = setup_vna_collection()
    test_query = "What are the baggage rules?"
    result = query_vna_info(collection, test_query)
    print(f"Test Query: {test_query}")
    print(f"Result: {result}")