"""
Common flight data templates and utilities
"""

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
    }
]