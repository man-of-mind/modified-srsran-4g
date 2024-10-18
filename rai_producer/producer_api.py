from flask import Flask, request, jsonify, Blueprint
from threading import Thread, Event
import time
import requests
import uuid


app = Flask(__name__)

y1_blueprint = Blueprint('y1', __name__, url_prefix='/Y1_RAI_Subscriptions/v1/')

subscriptions = {}
# next_subscription_id = 1
subscription_event = Event()


# dummy data
def generate_periodic_metrics(subscription_id, interval, start_time=None):
    if start_time:
        time.sleep(start_time - time.time())
    while subscription_id in subscriptions and not subscription_event.is_set():
        subscription = subscriptions[subscription_id]
        print('***************************', subscription_id)
        metric_data = {
            'subscription_id': subscription_id,
            'rai_type': subscription['raiType'],
            'rai_content': 'Sample RAN performance analytics',
            'timestamp': time.time(),
            'validity_period': 3600,
        }
        
        try:
            reponse = requests.post(subscription['notificationTargetAddress'], json=metric_data)
            reponse.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f'Error sending notification: {e}')
        time.sleep(interval)
        print('========================', subscription_id)

        
@y1_blueprint.route('/subscriptions/subscribe', methods=['POST'])
def subscribe():
    # global next_subscription_id
    try:
        data = request.json
        subscription_id = str(uuid.uuid4())
        # next_subscription_id += 1
        
        notification_method = data['notificationCriteria'].get('notificationMethod', 'PERIODIC')
        notification_interval = data['notificationCriteria'] .get('notificationPeriod', 1)
        notification_start_time = data['notificationCriteria'].get('notificationStartTime', time.time())
        
        subscription = {
            'subscription_id': subscription_id,
            'raiType': data['raiType'],
            'raiTypeVersion': data['raiTypeVersion'],
            'targetEntities': data['targetEntities'],
            'notificationTargetAddress': data['notificationTargetAddress'],
            'notificationCriteria': data['notificationCriteria'],
            'notificationParameters': data.get('filterParameters', None)
        } 
        subscriptions[subscription_id] = subscription
        
        if notification_method == 'PERIODIC':
            subscription_event.clear()
            thread = Thread(target=generate_periodic_metrics, args=(subscription_id, notification_interval, notification_start_time))
            thread.start()
        elif notification_method == 'EVENT_TRIGGERED':
            # To-Do: implement event-triggered notifications
            pass
        
        return jsonify({'subscription_id': subscription_id}), 201
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@y1_blueprint.route('/subscriptions/unsubscribe', methods=['DELETE'])
def unsubscribe():
    try:
        data = request.json
        subscription_id = data['subscription_id']
        if subscription_id in subscriptions:
            del subscriptions[subscription_id]
            if not subscriptions:
                subscription_event.set()
            return jsonify({'status': 'unsubscribed'}), 200
        else:
            return jsonify({'error': 'Subscription not found'}), 404
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
@y1_blueprint.route('/subscriptions/<int:subscription_id>', methods=['PUT'])
def update_subscription(subscription_id):
    try:
        data = request.json
        subscription = subscriptions.get(subscription_id)
        if subscription:
            subscriptions.update(data)
            return jsonify(subscription), 200
        else:
            return jsonify({'error': 'Subscription not found'}), 404
    except KeyError as e:
        return jsonify({'error': f'Missing key: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.errorhandler(403)
def access_denied(error):
    return jsonify({'error': 'Access Denied'}), 403

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error():
    return jsonify({'error': 'Internal Server Error'}), 500

app.register_blueprint(y1_blueprint)


if __name__ == '__main__':
    app.run(debug=True)
    