import requests
import json

# Test the API
url = "http://localhost:5000/api/predict"

data = {
    "ticker": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31"
}

print("Testing API...")
print(f"Sending request to {url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, json=data, timeout=300)  # 5 minute timeout
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"RMSE: {result['rmse']:.2f}")
        print(f"Predictions: {len(result['predicted'])} data points")
        print(f"Dates range: {result['dates'][0]} to {result['dates'][-1]}")
    else:
        print(f"\n❌ Error: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"\n❌ Connection Error: {e}")

