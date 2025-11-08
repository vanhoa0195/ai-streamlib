"""
Script to generate flight data with specific dates for the next 30 days
"""
import json
from datetime import datetime, timedelta

def generate_flights_with_dates():
    """Generate flight schedule with specific dates for next 30 days"""

    # Flight templates with routes and schedules
    flight_templates = [
        # Hanoi to Ho Chi Minh City
        {
            "flight_number": "VN210",
            "origin": "Hanoi (HAN)",
            "destination": "Ho Chi Minh City (SGN)",
            "departure_time": "06:00",
            "duration_minutes": 130,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "1,850,000 VND",
            "fare_premium_economy": "2,590,000 VND",
            "fare_business": "3,700,000 VND"
        },
        {
            "flight_number": "VN212",
            "origin": "Hanoi (HAN)",
            "destination": "Ho Chi Minh City (SGN)",
            "departure_time": "10:30",
            "duration_minutes": 135,
            "aircraft": "Airbus A321neo",
            "frequency": "Daily",
            "fare_economy": "1,900,000 VND",
            "fare_premium_economy": "2,660,000 VND",
            "fare_business": "3,800,000 VND"
        },
        {
            "flight_number": "VN220",
            "origin": "Hanoi (HAN)",
            "destination": "Ho Chi Minh City (SGN)",
            "departure_time": "14:00",
            "duration_minutes": 135,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "1,950,000 VND",
            "fare_premium_economy": "2,730,000 VND",
            "fare_business": "3,900,000 VND"
        },
        {
            "flight_number": "VN228",
            "origin": "Hanoi (HAN)",
            "destination": "Ho Chi Minh City (SGN)",
            "departure_time": "18:30",
            "duration_minutes": 135,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "2,000,000 VND",
            "fare_premium_economy": "2,800,000 VND",
            "fare_business": "4,000,000 VND"
        },
        # Ho Chi Minh City to Hanoi
        {
            "flight_number": "VN211",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Hanoi (HAN)",
            "departure_time": "06:30",
            "duration_minutes": 135,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "1,850,000 VND",
            "fare_premium_economy": "2,590,000 VND",
            "fare_business": "3,700,000 VND"
        },
        {
            "flight_number": "VN213",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Hanoi (HAN)",
            "departure_time": "11:00",
            "duration_minutes": 135,
            "aircraft": "Airbus A321neo",
            "frequency": "Daily",
            "fare_economy": "1,900,000 VND",
            "fare_premium_economy": "2,660,000 VND",
            "fare_business": "3,800,000 VND"
        },
        {
            "flight_number": "VN221",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Hanoi (HAN)",
            "departure_time": "15:00",
            "duration_minutes": 135,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "1,950,000 VND",
            "fare_premium_economy": "2,730,000 VND",
            "fare_business": "3,900,000 VND"
        },
        {
            "flight_number": "VN229",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Hanoi (HAN)",
            "departure_time": "19:00",
            "duration_minutes": 135,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "2,000,000 VND",
            "fare_premium_economy": "2,800,000 VND",
            "fare_business": "4,000,000 VND"
        },
        # Ho Chi Minh City to Singapore
        {
            "flight_number": "VN660",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Singapore (SIN)",
            "departure_time": "09:00",
            "duration_minutes": 135,
            "aircraft": "Airbus A321neo",
            "frequency": "Daily",
            "fare_economy": "4,000,000 VND",
            "fare_premium_economy": "5,600,000 VND",
            "fare_business": "8,000,000 VND"
        },
        {
            "flight_number": "VN662",
            "origin": "Ho Chi Minh City (SGN)",
            "destination": "Singapore (SIN)",
            "departure_time": "15:30",
            "duration_minutes": 135,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "4,100,000 VND",
            "fare_premium_economy": "5,740,000 VND",
            "fare_business": "8,200,000 VND"
        },
        # Hanoi to Singapore
        {
            "flight_number": "VN420",
            "origin": "Hanoi (HAN)",
            "destination": "Singapore (SIN)",
            "departure_time": "08:30",
            "duration_minutes": 210,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "5,000,000 VND",
            "fare_premium_economy": "7,000,000 VND",
            "fare_business": "10,000,000 VND"
        },
        {
            "flight_number": "VN422",
            "origin": "Hanoi (HAN)",
            "destination": "Singapore (SIN)",
            "departure_time": "16:00",
            "duration_minutes": 210,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "5,200,000 VND",
            "fare_premium_economy": "7,280,000 VND",
            "fare_business": "10,400,000 VND"
        },
        # Hanoi to Tokyo
        {
            "flight_number": "VN340",
            "origin": "Hanoi (HAN)",
            "destination": "Tokyo (NRT)",
            "departure_time": "09:30",
            "duration_minutes": 330,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "8,500,000 VND",
            "fare_premium_economy": "11,900,000 VND",
            "fare_business": "17,000,000 VND"
        },
        {
            "flight_number": "VN342",
            "origin": "Hanoi (HAN)",
            "destination": "Tokyo (NRT)",
            "departure_time": "22:00",
            "duration_minutes": 330,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "8,800,000 VND",
            "fare_premium_economy": "12,320,000 VND",
            "fare_business": "17,600,000 VND"
        },
        # Hanoi to Seoul
        {
            "flight_number": "VN530",
            "origin": "Hanoi (HAN)",
            "destination": "Seoul (ICN)",
            "departure_time": "11:00",
            "duration_minutes": 270,
            "aircraft": "Airbus A350-900",
            "frequency": "Daily",
            "fare_economy": "6,500,000 VND",
            "fare_premium_economy": "9,100,000 VND",
            "fare_business": "13,000,000 VND"
        },
        {
            "flight_number": "VN532",
            "origin": "Hanoi (HAN)",
            "destination": "Seoul (ICN)",
            "departure_time": "20:00",
            "duration_minutes": 270,
            "aircraft": "Boeing 787-9 Dreamliner",
            "frequency": "Daily",
            "fare_economy": "6,700,000 VND",
            "fare_premium_economy": "9,380,000 VND",
            "fare_business": "13,400,000 VND"
        },
    ]

    flights = []
    today = datetime.now()

    # Generate flights for next 30 days
    for day_offset in range(30):
        flight_date = today + timedelta(days=day_offset)

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

if __name__ == "__main__":
    flights = generate_flights_with_dates()
    print(f"Generated {len(flights)} flights for the next 30 days")
    print("\nSample flight:")
    print(json.dumps(flights[0], indent=2, ensure_ascii=False))

    # Save to file
    with open("flights_with_dates.json", "w", encoding="utf-8") as f:
        json.dump(flights, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(flights)} flights to flights_with_dates.json")
