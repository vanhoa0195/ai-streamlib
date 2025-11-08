"""
Script to generate flight data for December 2025
"""
import json
import os
from datetime import datetime, timedelta
from flight_data import flight_templates

def generate_december_flights():
    """Generate flight schedule for December 2025"""
    flights = []
    
    # Set start date to December 1, 2025
    start_date = datetime(2025, 12, 1)
    
    # Generate flights for all days in December
    for day_offset in range(31):  # December has 31 days
        flight_date = start_date + timedelta(days=day_offset)

        for template in flight_templates:
            # Parse departure time
            dep_hour, dep_min = map(int, template["departure_time"].split(":"))

            # Create departure datetime
            departure_dt = flight_date.replace(hour=dep_hour, minute=dep_min, second=0, microsecond=0)

            # Calculate arrival datetime
            arrival_dt = departure_dt + timedelta(minutes=template["duration_minutes"])

            # Calculate duration in hours and minutes
            duration_hours = template["duration_minutes"] // 60
            duration_mins = template["duration_minutes"] % 60
            duration_str = f"{duration_hours}h {duration_mins}m" if duration_mins > 0 else f"{duration_hours}h"

            # Create flight entry
            flight = {
                "id": f"flight_{template['flight_number']}_{flight_date.strftime('%Y%m%d')}",
                "flight_number": template["flight_number"],
                "origin": template["origin"],
                "destination": template["destination"],
                "departure_date": departure_dt.strftime("%Y-%m-%d"),
                "departure_time": departure_dt.strftime("%H:%M"),
                "departure_datetime": departure_dt.strftime("%Y-%m-%d %H:%M"),
                "arrival_date": arrival_dt.strftime("%Y-%m-%d"),
                "arrival_time": arrival_dt.strftime("%H:%M"),
                "arrival_datetime": arrival_dt.strftime("%Y-%m-%d %H:%M"),
                "duration": duration_str,
                "aircraft": template["aircraft"],
                "frequency": template["frequency"],
                "status": "Available",
                "fare_economy": template["fare_economy"],
                "fare_premium_economy": template["fare_premium_economy"],
                "fare_business": template["fare_business"],
                "source": "flight_schedule"
            }

            flights.append(flight)

    return flights

def save_flights_by_date(flights):
    """Save flights to individual JSON files by date in the datasource/flights/december_2025 directory"""
    # Group flights by origin-destination and date
    flights_by_date = {}
    for flight in flights:
        departure_date = flight["departure_date"]
        origin_code = flight["origin"].split("(")[1].strip(")")
        dest_code = flight["destination"].split("(")[1].strip(")")
        route_key = f"{origin_code}{dest_code}"
        date_key = departure_date.replace("-", "")
        
        key = f"{route_key}-{date_key}"
        if key not in flights_by_date:
            flights_by_date[key] = {
                "flightOptions": [],
                "date": departure_date
            }
        
        # Convert flight to flightOptions format
        flight_option = {
            "journeys": [{
                "airlineName": "Vietnam Airlines",
                "flightNumber": flight["flight_number"],
                "duration": str(int(flight["duration"].split("h")[0]) * 60),  # Convert to minutes
                "aircraft": flight["aircraft"],
                "fareOptions": [{
                    "journeyFareClass": "Economy",
                    "fareInfos": [{
                        "cabin": {"name": "Economy"},
                        "seatRemain": "9+",
                        "price": flight["fare_economy"]
                    }]
                }, {
                    "journeyFareClass": "Premium Economy",
                    "fareInfos": [{
                        "cabin": {"name": "Premium Economy"},
                        "seatRemain": "9+",
                        "price": flight["fare_premium_economy"]
                    }]
                }, {
                    "journeyFareClass": "Business",
                    "fareInfos": [{
                        "cabin": {"name": "Business"},
                        "seatRemain": "9+",
                        "price": flight["fare_business"]
                    }]
                }]
            }]
        }
        
        flights_by_date[key]["flightOptions"].append(flight_option)
    
    # Create december_2025 directory if it doesn't exist
    output_dir = "datasource/flights/december_2025"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each group to a separate file
    for key, data in flights_by_date.items():
        filename = os.path.join(output_dir, f"{key}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {key} to {filename}")

if __name__ == "__main__":
    # Generate December flights
    december_flights = generate_december_flights()
    print(f"Generated {len(december_flights)} flights for December 2025")
    
    # Save to individual files by date
    save_flights_by_date(december_flights)
    
    # Save all flights to a single file for reference
    with open("december_flights.json", "w", encoding="utf-8") as f:
        json.dump(december_flights, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all {len(december_flights)} flights to december_flights.json")