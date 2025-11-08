import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

def format_currency(amount: Union[int, float]) -> str:
    """Format currency in Vietnamese Dong with thousand separators"""
    if amount is None:
        return "N/A"
    return f"{int(amount):,}₫"

def format_duration(minutes: int) -> str:
    """Format duration in hours and minutes"""
    hours = minutes // 60
    mins = minutes % 60
    if mins == 0:
        return f"{hours}h"
    return f"{hours}h {mins}m"

def sort_flights(flights: List[Dict], sort_by: str = "departure") -> List[Dict]:
    """
    Sort flights based on different criteria
    
    Args:
        flights: List of flight dictionaries
        sort_by: Sorting criteria ('departure', 'duration', 'price', 'stops')
        
    Returns:
        Sorted list of flights
    """
    if sort_by == "departure":
        return sorted(flights, key=lambda x: x.get('departTime', ''))
    elif sort_by == "duration":
        return sorted(flights, key=lambda x: x.get('duration', 0))
    elif sort_by == "price":
        return sorted(flights, key=lambda x: min([fare.get('price', float('inf')) 
                                                for fare in x.get('fareOptions', [])], 
                                               default=float('inf')))
    elif sort_by == "stops":
        return sorted(flights, key=lambda x: len(x.get('connectingFlights', [])))
    return flights

def format_flight_info(flight_data: Dict[str, Any], language: str = "en") -> str:
    """
    Format the flight information into a readable string.
    
    Args:
        flight_data: Flight information from the API response
        
    Returns:
        Formatted string with flight details
    """
    # City names in English and Vietnamese
    airport_codes = {
        "HAN": {"en": "Hanoi (HAN)", "vi": "Hà Nội (HAN)"},
        "SGN": {"en": "Ho Chi Minh City (SGN)", "vi": "TP. Hồ Chí Minh (SGN)"},
        "DAD": {"en": "Da Nang (DAD)", "vi": "Đà Nẵng (DAD)"},
        "CXR": {"en": "Nha Trang (CXR)", "vi": "Nha Trang (CXR)"},
        "PQC": {"en": "Phu Quoc (PQC)", "vi": "Phú Quốc (PQC)"},
        "HUI": {"en": "Hue (HUI)", "vi": "Huế (HUI)"},
        "VCA": {"en": "Can Tho (VCA)", "vi": "Cần Thơ (VCA)"},
        "VCS": {"en": "Con Dao (VCS)", "vi": "Côn Đảo (VCS)"},
        "NRT": {"en": "Tokyo Narita (NRT)", "vi": "Tokyo Narita (NRT)"},
        "HND": {"en": "Tokyo Haneda (HND)", "vi": "Tokyo Haneda (HND)"},
        "ICN": {"en": "Seoul Incheon (ICN)", "vi": "Seoul Incheon (ICN)"},
        "BKK": {"en": "Bangkok (BKK)", "vi": "Bangkok (BKK)"},
        "SIN": {"en": "Singapore (SIN)", "vi": "Singapore (SIN)"},
        "KUL": {"en": "Kuala Lumpur (KUL)", "vi": "Kuala Lumpur (KUL)"},
        "MEL": {"en": "Melbourne (MEL)", "vi": "Melbourne (MEL)"},
        "SYD": {"en": "Sydney (SYD)", "vi": "Sydney (SYD)"},
        "CDG": {"en": "Paris (CDG)", "vi": "Paris (CDG)"},
        "LHR": {"en": "London (LHR)", "vi": "London (LHR)"},
        "FRA": {"en": "Frankfurt (FRA)", "vi": "Frankfurt (FRA)"}
    }
    
    # Status messages in English and Vietnamese
    status_messages = {
        "AVAILABLE": {"en": "Available", "vi": "Còn chỗ"},
        "LIMITED": {"en": "Limited Seats", "vi": "Còn ít chỗ"},
        "FULL": {"en": "Fully Booked", "vi": "Đã đầy"},
        "WAITLIST": {"en": "Waitlist", "vi": "Chờ đợi"},
        "CANCELED": {"en": "Canceled", "vi": "Đã hủy"},
        "DELAYED": {"en": "Delayed", "vi": "Bị trễ"}
    }
    
    # Fare family descriptions
    fare_families = {
        "ECO_LITE": {
            "en": "Economy Lite",
            "vi": "Phổ thông Tiết kiệm",
            "features": {
                "en": ["Non-refundable", "7kg hand baggage", "No checked baggage", "No seat selection"],
                "vi": ["Không hoàn tiền", "7kg hành lý xách tay", "Không có hành lý ký gửi", "Không chọn chỗ ngồi"]
            }
        },
        "ECO_BASIC": {
            "en": "Economy Basic",
            "vi": "Phổ thông Cơ bản",
            "features": {
                "en": ["Change fee applies", "7kg hand baggage", "20kg checked baggage", "Seat selection available"],
                "vi": ["Phí đổi vé", "7kg hành lý xách tay", "20kg hành lý ký gửi", "Được chọn chỗ ngồi"]
            }
        },
        "ECO_FLEX": {
            "en": "Economy Flex",
            "vi": "Phổ thông Linh hoạt",
            "features": {
                "en": ["Free date change", "7kg hand baggage", "30kg checked baggage", "Free seat selection", "Priority check-in"],
                "vi": ["Đổi ngày miễn phí", "7kg hành lý xách tay", "30kg hành lý ký gửi", "Chọn chỗ miễn phí", "Ưu tiên làm thủ tục"]
            }
        }
    }
    
    # Meal types
    meal_types = {
        "REGULAR": {"en": "Standard Meal", "vi": "Suất ăn tiêu chuẩn"},
        "VEGETARIAN": {"en": "Vegetarian Meal", "vi": "Suất ăn chay"},
        "HALAL": {"en": "Halal Meal", "vi": "Suất ăn Halal"},
        "KOSHER": {"en": "Kosher Meal", "vi": "Suất ăn Kosher"},
        "CHILD": {"en": "Child Meal", "vi": "Suất ăn trẻ em"},
        "DIABETIC": {"en": "Diabetic Meal", "vi": "Suất ăn tiểu đường"}
    }
    
    # Flight amenities
    amenities = {
        "WIFI": {"en": "Wi-Fi Available", "vi": "Có Wi-Fi"},
        "POWER": {"en": "Power Outlets", "vi": "Ổ cắm điện"},
        "USB": {"en": "USB Charging", "vi": "Sạc USB"},
        "IFE": {"en": "Entertainment System", "vi": "Hệ thống giải trí"},
        "FLATBED": {"en": "Flat-bed Seats", "vi": "Ghế nằm"},
        "LOUNGE": {"en": "Lounge Access", "vi": "Phòng chờ VIP"}
    }
    
    # Cabin class names in English and Vietnamese
    cabin_classes = {
        "ECONOMY": {"en": "Economy", "vi": "Phổ thông"},
        "PREMIUM_ECONOMY": {"en": "Premium Economy", "vi": "Phổ thông đặc biệt"},
        "BUSINESS": {"en": "Business", "vi": "Thương gia"},
        "FIRST": {"en": "First Class", "vi": "Hạng nhất"}
    }
    
    formatted_flights = []
    
    # Check if we have valid flight data
    if not flight_data or 'data' not in flight_data or not flight_data['data']:
        return "No flights found for the specified criteria."
        
    for flight in flight_data['data']:
        try:
            # Get origin and destination names
            origin = airport_codes.get(flight['startPoint'], f"{flight['startPoint']}")
            destination = airport_codes.get(flight['endPoint'], f"{flight['endPoint']}")
            
            # Format departure and arrival times
            depart_time = datetime.fromisoformat(flight['departTime'].replace('Z', '+00:00'))
            arrive_time = datetime.fromisoformat(flight['arriveTime'].replace('Z', '+00:00'))
            
            # Calculate flight duration
            duration = (arrive_time - depart_time).total_seconds() / 3600  # in hours
            
            # Extract prices for different classes
            prices = flight.get('fareOptions', {})
            economy_price = None
            premium_price = None
            business_price = None
            
            for fare in prices:
                if fare['cabinClass'] == 'ECONOMY':
                    economy_price = fare['price']
                elif fare['cabinClass'] == 'PREMIUM_ECONOMY':
                    premium_price = fare['price']
                elif fare['cabinClass'] == 'BUSINESS':
                    business_price = fare['price']
            
            # Get localized city names
            origin = airport_codes.get(flight['startPoint'], {"en": flight['startPoint'], "vi": flight['startPoint']})[language]
            destination = airport_codes.get(flight['endPoint'], {"en": flight['endPoint'], "vi": flight['endPoint']})[language]
            
            # Get localized status
            status = flight.get('status', 'AVAILABLE')
            status_text = status_messages.get(status, {"en": status, "vi": status})[language]
            
            # Format the flight information
            flight_info = []
            
            # Flight number and route
            flight_info.append(
                f"✈️ **{flight['flightNumber']}** — {origin} → {destination} "
                f"({flight.get('aircraft', 'Aircraft TBD')}, {format_duration(int(duration * 60))})"
            )
            
            # Time information
            time_info = [
                f"🕓 {depart_time.strftime('%Y-%m-%d %H:%M')} → {arrive_time.strftime('%Y-%m-%d %H:%M')}",
                f"📅 {'Daily' if language == 'en' else 'Hàng ngày'}"
            ]
            flight_info.append(" | ".join(time_info))
            
            # Status and availability
            flight_info.append(f"💺 {'Status' if language == 'en' else 'Tình trạng'}: {status_text}")
            
            # Connection information if available
            if flight.get('connectingFlights'):
                connects = flight['connectingFlights']
                connection_info = []
                for conn in connects:
                    conn_city = airport_codes.get(conn['airport'], {"en": conn['airport'], "vi": conn['airport']})[language]
                    conn_time = int(conn.get('connectionTime', 0))
                    connection_info.append(f"{conn_city} ({format_duration(conn_time)} layover)")
                if connection_info:
                    flight_info.append(f"� {'Connections' if language == 'en' else 'Quá cảnh'}: {' → '.join(connection_info)}")
            
            # Baggage allowance
            baggage = flight.get('baggage', {})
            if baggage:
                baggage_info = []
                for class_type, allowance in baggage.items():
                    class_name = cabin_classes.get(class_type, {"en": class_type, "vi": class_type})[language]
                    if isinstance(allowance, dict):
                        checked = allowance.get('checkedBaggage', 'N/A')
                        cabin = allowance.get('cabinBaggage', 'N/A')
                        baggage_info.append(f"{class_name}: {checked} + {cabin}")
                if baggage_info:
                    flight_info.append(f"🧳 {'Baggage' if language == 'en' else 'Hành lý'}: {' | '.join(baggage_info)}")
            
            flight_info = "\n".join(flight_info)
            
            # Add price information if available
            if any([economy_price, premium_price, business_price]):
                price_info = []
                price_label = "Prices" if language == "en" else "Giá vé"
                
                for class_type, price in [
                    ("ECONOMY", economy_price),
                    ("PREMIUM_ECONOMY", premium_price),
                    ("BUSINESS", business_price)
                ]:
                    if price:
                        class_name = cabin_classes[class_type][language]
                        price_info.append(f"{class_name}: {format_currency(price)}")
                
                if price_info:
                    flight_info += f"\n💰 {price_label}: {' | '.join(price_info)}"
            
            formatted_flights.append(flight_info)
            
        except Exception as e:
            continue
    
    if not formatted_flights:
        return "No flights found or unable to parse flight information."
    
    return "Dưới đây là các chuyến bay của Vietnam Airlines phù hợp với yêu cầu của bạn:\n\n" + "\n".join(formatted_flights)

def search_flights(
    start_point: str,
    end_point: str,
    depart_date: str,
    return_date: Optional[str] = None,
    adult_count: int = 1,
    child_count: int = 0,
    infant_count: int = 0,
    promo_code: str = "",
    cabin_class: str = "ECONOMY",
    sort_by: str = "departure",
    language: str = "en"
) -> Optional[Dict[str, Any]]:
    """
    Search for flights using the Flychills API
    
    Args:
        start_point: Origin airport code (e.g., 'HAN')
        end_point: Destination airport code (e.g., 'SGN')
        depart_date: Departure date in YYYY-MM-DD format
        adult_count: Number of adult passengers
        child_count: Number of child passengers
        infant_count: Number of infant passengers
        promo_code: Optional promotion code
        
    Returns:
        Dict containing the API response or None if request fails
    """
    url = "https://dev.flychills.com/ds/api/booking/search-flight"
    
    # Build flight requests
    flights_request = [
        {
            "airline": "VN",
            "startPoint": start_point,
            "providers": "VN",
            "supplier": "TPV",
            "leg": 0,
            "endPoint": end_point,
            "departDate": depart_date,
            "cabinClass": cabin_class
        }
    ]
    
    # Add return flight if return_date is specified
    if return_date:
        flights_request.append({
            "airline": "VN",
            "startPoint": end_point,
            "providers": "VN",
            "supplier": "TPV",
            "leg": 1,
            "endPoint": start_point,
            "departDate": return_date,
            "cabinClass": cabin_class
        })
    
    payload = {
        "adultCount": adult_count,
        "childCount": child_count,
        "infantCount": infant_count,
        "promoCode": promo_code,
        "flightsRequest": flights_request,
        "userKey": "test-user"
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Sort flights if we have valid data
        if data and 'data' in data and isinstance(data['data'], list):
            data['data'] = sort_flights(data['data'], sort_by)
            
            # Format the response with the specified language
            formatted_response = format_flight_info(data, language)
            data['formatted_response'] = formatted_response
            
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error searching flights: {e}")
        return None