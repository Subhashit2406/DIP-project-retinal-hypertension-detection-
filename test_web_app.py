import requests

# Test the web app /analyze endpoint
url = 'http://localhost:5000/analyze'

# Open the image file
with open('dataset/train/retinal_demo.png', 'rb') as f:
    files = {'file': f}
    try:
        response = requests.post(url, files=files)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            data = response.json()
            if 'error' in data:
                print(f"Error in response: {data['error']}")
            else:
                print("Success! Response data:")
                print(f"AVR: {data.get('avr')}")
                print(f"Tortuosity: {data.get('tortuosity')}")
                print(f"CDR: {data.get('cdr')}")
                print(f"Overall risk: {data.get('overall_risk')}")
        else:
            print(f"HTTP error: {response.status_code}")
            print(f"Response body: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
