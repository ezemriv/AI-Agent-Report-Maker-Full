import requests
import time

def fetch_client_data(client_id, retries=5, delay=2):
    url = 'https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/clients-data'
    params = {'client_id': client_id}
    
    for attempt in range(retries):
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Attempt {attempt + 1} failed for client_id {client_id} with status {response.status_code}. Retrying in {delay} seconds.")
            time.sleep(delay)
    
    print(f"Failed to fetch client data for client_id {client_id} after {retries} attempts.")
    return None

def fetch_card_data(client_id, retries=5, delay=2):
    url = 'https://faas-lon1-917a94a7.doserverless.co/api/v1/web/fn-a1f52b59-3551-477f-b8f3-de612fbf2769/default/cards-data'
    params = {'client_id': client_id}
    
    for attempt in range(retries):
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Attempt {attempt + 1} failed for client_id {client_id} with status {response.status_code}. Retrying in {delay} seconds.")
            time.sleep(delay)
    
    print(f"Failed to fetch card data for client_id {client_id} after {retries} attempts.")
    return None

if __name__ == "__main__":
    # Test calls with example client IDs
    test_client_id = 1556
    print("Client Data:", fetch_client_data(test_client_id))
    print("Card Data:", fetch_card_data(test_client_id))
