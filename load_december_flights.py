"""
Script to load December flight data into ChromaDB
"""
from vna_chromadb import setup_vna_collection, query_vna_info
import os
import json

def load_december_flights():
    """Load December flight data into ChromaDB"""
    # Set up the ChromaDB collection
    collection = setup_vna_collection()
    
    print("Loading December flight data...")
    
    # Process flight data files
    current_dir = os.path.dirname(__file__)
    flights_dir = os.path.join(current_dir, "datasource", "flights")
    
    december_files = [f for f in os.listdir(flights_dir) if f.endswith(".json") and "-2025120" in f or "-2025121" in f]
    doc_count = 0
    
    for file in december_files:
        file_path = os.path.join(flights_dir, file)
        print(f"Processing {file}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            flight_data = json.load(f)
        
        # Extract route and date from filename
        route_date = file.split('.')[0]  # e.g., HANSGN-20251201
        route = route_date[:6]  # e.g., HANSGN
        date = route_date[7:]  # e.g., 20251201
        
        # Process each flight option
        documents = []
        metadatas = []
        
        for flight_option in flight_data.get('flightOptions', []):
            for journey in flight_option.get('journeys', []):
                airline = journey.get('airlineName', '')
                duration = journey.get('duration', '')
                
                for fare_option in journey.get('fareOptions', []):
                    fare_class = fare_option.get('journeyFareClass', '')
                    fare_infos = fare_option.get('fareInfos', [])
                    
                    for fare_info in fare_infos:
                        cabin = fare_info.get('cabin', {}).get('name', '')
                        seats = fare_info.get('seatRemain', '')
                        price = fare_info.get('price', '')
                        
                        # Create a rich document text
                        doc = (f"Flight from {route[:3]} to {route[3:]} on {date}: "
                               f"{airline} flight, duration {duration} minutes, "
                               f"Class {fare_class}, Cabin {cabin}, "
                               f"{seats} seats available, Price: {price}")
                        
                        documents.append(doc)
                        metadatas.append({
                            "category": "flight_options",
                            "route": route,
                            "date": date,
                            "source": "december_flights",
                            "file": file
                        })
        
        # Add documents to collection
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=[f"dec_flight_{route}_{date}_{i}" for i in range(len(documents))]
            )
            doc_count += len(documents)
            print(f"Added {len(documents)} documents from {file}")
    
    print(f"\nTotal December flight documents loaded into ChromaDB: {doc_count}")
    
    # Test query
    test_query = "Show me flights from Hanoi to Ho Chi Minh City on December 25, 2025"
    result = query_vna_info(collection, test_query)
    print(f"\nTest Query: {test_query}")
    print(f"Result: {result}")

if __name__ == "__main__":
    load_december_flights()