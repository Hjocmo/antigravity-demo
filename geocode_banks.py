import pandas as pd
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

def geocode_with_fallback(geolocator, bank_name, country):
    capitals = {
        'AT': 'Vienna', 'BE': 'Brussels', 'CY': 'Nicosia', 'DE': 'Berlin', 
        'ES': 'Madrid', 'FI': 'Helsinki', 'FR': 'Paris', 'GR': 'Athens', 
        'IE': 'Dublin', 'LI': 'Vaduz', 'LU': 'Luxembourg', 'LV': 'Riga',
        'MT': 'Valletta', 'NO': 'Oslo', 'PL': 'Warsaw', 'PT': 'Lisbon', 
        'RO': 'Bucharest', 'SE': 'Stockholm', 'SI': 'Ljubljana', 'IT': 'Rome', 
        'NL': 'Amsterdam', 'DK': 'Copenhagen', 'CZ': 'Prague', 'HU': 'Budapest',
        'SK': 'Bratislava', 'BG': 'Sofia', 'HR': 'Zagreb', 'EE': 'Tallinn', 
        'LT': 'Vilnius'
    }
    
    # 1. Try with Bank Name and Country
    try:
        location = geolocator.geocode(f"{bank_name}, {country}", timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass
        
    time.sleep(1) # Extra sleep if failed or trying fallback
    
    # 2. Fallback to Capital City of the Country
    capital = capitals.get(country, "")
    query = f"{capital}, {country}" if capital else country
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return location.latitude, location.longitude
    except (GeocoderTimedOut, GeocoderServiceError):
        pass
        
    return None, None

def main():
    print("Loading network_input.csv...")
    df = pd.read_csv('network_input.csv')
    
    # User-agent must be provided per Nominatim usage policy
    geolocator = Nominatim(user_agent="systemic_banking_risk_simulator_bot")
    
    lats = []
    lons = []
    
    print(f"Geocoding {len(df)} banks...")
    for idx, row in df.iterrows():
        bank = row['BankName']
        country = row['Country']
        
        lat, lon = geocode_with_fallback(geolocator, bank, country)
        
        if lat is not None:
            print(f"[{idx+1}/{len(df)}] ✅ {bank} ({country}) -> {lat:.4f}, {lon:.4f}")
        else:
            print(f"[{idx+1}/{len(df)}] ❌ {bank} ({country}) -> NOT FOUND")
            # Absolute fallback to rough center of europe if all fails
            lat, lon = 48.0, 10.0
            
        lats.append(lat)
        lons.append(lon)
        time.sleep(1) # Respect Nominatim rate limit of 1 request per second
        
    df['lat'] = lats
    df['lon'] = lons
    
    out_file = 'network_input_geocoded.csv'
    df.to_csv(out_file, index=False)
    print(f"Done! Geocoded data saved to {out_file}.")

if __name__ == '__main__':
    main()
