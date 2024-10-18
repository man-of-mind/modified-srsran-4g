from flask import Flask, request, jsonify
import requests
import time
import json
import atexit

app = Flask(__name__)

subscription_id = None

@app.route('/y1_consumer/notify', methods=['POST'])
def receive_notifications():
    try:
        data = request.json
        print('****************************Received RAN analytics data************************************')
        print(json.dumps(data, indent=4))
        print('='*50)
        return jsonify({'status': 'received RAN metrics successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
def subscribe_for_rai():
    global subscription_id
    data = {
        'raiType': 'UE-centric data',
        'raiTypeVersion': '1.0',
        'targetEntities': ['entity 1'],
        'notificationTargetAddress': 'http://localhost:5001/y1_consumer/notify',
        'notificationCriteria': {
            'notificationMethod': 'PERIODIC',
            'notificationPeriod': 2,
            'notificationStartTime': time.time() + 2
        }
    }
    try:
        response = requests.post('http://localhost:5000/Y1_RAI_Subscriptions/v1/subscriptions/subscribe', json=data)
        response.raise_for_status()
        subscription_id = response.json()['subscription_id']
        print('RAN metrics subscribed with subscription_id:', subscription_id)
        # return subscription_id
    except requests.exceptions.HTTPError as e:
        print(f'HTTP Error: {e.response.status_code} {e.response.text}')
    except requests.exceptions.ConnectionError as e:
        print(f'Error: Unable to connect to the RAI producer API server')
    except requests.exceptions.RequestException as e:
        print(f'Error: {e}')
        

def unsubscribe_rai():
    global subscription_id
    if subscription_id is not None:
        data = {'subscription_id': subscription_id}
        try:
            response = requests.delete('http://localhost:5000/Y1_RAI_Subscriptions/v1/subscriptions/unsubscribe', json=data)
            response.raise_for_status()
            print('Unsubscribed from RAI Producer')
        except requests.exceptions.ConnectionError as e:
            print(f'Error: Unable to connect to the RAI producer API server')
        except requests.exceptions.HTTPError as e:
            print(f'HTTP Error {e.response.status_code} {e.response.text}')
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')
        
atexit.register(unsubscribe_rai)        

if __name__ == '__main__':
    subscribe_for_rai()
    try:
        app.run(port=5001, debug=True)
    except KeyboardInterrupt:
        unsubscribe_rai()