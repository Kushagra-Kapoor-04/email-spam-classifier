import requests

url = "http://localhost:5000/predict"

# Test spam
response = requests.post(url, json={"text": "Congratulations you won a free iPhone click here to claim"})
print("SPAM TEST:", response.json())

# Test ham  
response = requests.post(url, json={"text": "Hey are we still meeting for lunch tomorrow?"})
print("HAM TEST:", response.json())