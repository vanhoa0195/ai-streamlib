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

def load_datasource_file(file_path):
    """Load and process a single JSON file from datasource."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    metadatas = []
    file_name = os.path.basename(file_path)

    # Handle array of fares (fare_info.json)
    if isinstance(data, list) and all('booking_class' in item for item in data):
        for fare in data:
            fare_bases = fare.get('fare_bases', [])
            for base in fare_bases:
                benefits = ", ".join(base.get('benefits', []))
                doc = (f"Fare Class {fare.get('name', '')} ({fare.get('booking_class', '')}), "
                      f"Type: {base.get('name', '')} - {base.get('description', '')}. "
                      f"Benefits: {benefits}")
                documents.append(doc)
                metadatas.append({
                    "category": "fare_info",
                    "id": f"{fare.get('id', '')}_{base.get('code', '')}",
                    "source": fare.get('source', ''),
                    "file": file_name
                })
        return documents, metadatas

    # Handle dictionary format
    if isinstance(data, dict):
        # Airline info
        if 'airline_info' in data:
            airline_info = data['airline_info']
            doc = f"Airline Information: {airline_info.get('description', '')}"
            documents.append(doc)
            metadatas.append({
                "category": "airline_info",
                "source": airline_info.get('source', ''),
                "file": file_name
            })

        # Routes
        for route in data.get('routes', []):
            doc = f"Route {route.get('title', '')}: {route.get('description', '')}. Frequency: {route.get('frequency', '')}. Airports: {route.get('departure_airport', '')} to {route.get('arrival_airport', '')}"
            documents.append(doc)
            metadatas.append({
                "category": "routes",
                "id": route.get('id', ''),
                "source": route.get('source', ''),
                "file": file_name
            })

        # Flight options from flight JSONs
        for flight_option in data.get('flightOptions', []):
            for journey in flight_option.get('journeys', []):
                airline = journey.get('airlineName', '')
                duration = journey.get('duration', '')
                for fare_option in journey.get('fareOptions', []):
                    fare_class = fare_option.get('journeyFareClass', '')
                    fare_infos = fare_option.get('fareInfos', [])
                    for fare_info in fare_infos:
                        cabin = fare_info.get('cabin', {}).get('name', '')
                        seats = fare_info.get('seatRemain', '')
                        doc = f"Flight {airline}: {duration} minutes, Class {fare_class}, Cabin {cabin}, {seats} seats available"
                        documents.append(doc)
                        metadatas.append({
                            "category": "flight_options",
                            "source": "flight_data",
                            "file": file_name
                        })

        # Promotions
        for promo in data.get('current_promotions', []):
            doc = f"Promotion {promo.get('title', '')}: {promo.get('description', '')}. Terms: {promo.get('terms', '')}"
            documents.append(doc)
            metadatas.append({
                "category": "promotions",
                "id": promo.get('id', ''),
                "source": promo.get('source', ''),
                "file": file_name
            })

        # Ancillary products
        for product in data.get('ancillary_products', []):
            doc = f"Service {product.get('name', '')}: {product.get('description', '')}"
            if 'price_note' in product:
                doc += f". {product['price_note']}"
            if 'note' in product:
                doc += f". Note: {product['note']}"
            documents.append(doc)
            metadatas.append({
                "category": "products",
                "id": product.get('id', ''),
                "source": product.get('source', ''),
                "file": file_name
            })

        # Procedures
        for proc in data.get('procedures', []):
            doc = f"Procedure {proc.get('name', '')}: {proc.get('description', '')}"
            if 'steps' in proc:
                doc += f". Steps: {', '.join(proc['steps'])}"
            if 'details' in proc:
                doc += f". Details: {json.dumps(proc['details'])}"
            documents.append(doc)
            metadatas.append({
                "category": "procedures",
                "id": proc.get('id', ''),
                "source": proc.get('source', ''),
                "file": file_name
            })

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
    datasource_dir = os.path.join(current_dir, "datasource")

    # Process all JSON files in datasource directory
    all_documents = []
    all_metadatas = []
    doc_count = 0

    for root, _, files in os.walk(datasource_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    documents, metadatas = load_datasource_file(file_path)
                    all_documents.extend(documents)
                    all_metadatas.extend(metadatas)
                    doc_count += len(documents)
                    print(f"Processed {file}: {len(documents)} documents")
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    # Add all documents to collection with unique IDs
    if all_documents:
        collection.add(
            documents=all_documents,
            metadatas=all_metadatas,
            ids=[f"vna_doc_{i}" for i in range(len(all_documents))]
        )

    print(f"Total documents loaded into ChromaDB: {doc_count}")

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