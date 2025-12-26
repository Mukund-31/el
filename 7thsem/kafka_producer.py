"""
Kafka Producer for Real-Time Queue Simulation
Sends simulation events to Kafka topic for RL agent consumption
"""

from kafka import KafkaProducer
import json
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueEventProducer:
    """Produces queue simulation events to Kafka."""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='queue-events'):
        self.topic = topic
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info(f"✅ Kafka Producer connected to {bootstrap_servers}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            self.producer = None
    
    def send_queue_state(self, state_data):
        """Send current queue state to Kafka."""
        if self.producer is None:
            return False
            
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'QUEUE_STATE',
                'data': state_data
            }
            
            future = self.producer.send(
                self.topic,
                key='queue_state',
                value=event
            )
            
            # Wait for send to complete (with timeout)
            future.get(timeout=1)
            logger.debug(f"Sent queue state: {state_data['queue_length']} customers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    def send_arrival(self, num_arrivals, hour):
        """Send customer arrival event."""
        if self.producer is None:
            return False
            
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'ARRIVAL',
                'data': {
                    'num_arrivals': num_arrivals,
                    'hour': hour
                }
            }
            
            self.producer.send(self.topic, key='arrival', value=event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send arrival: {e}")
            return False
    
    def send_rl_decision(self, action, confidence, state_before, state_after):
        """Send RL agent decision to Kafka."""
        if self.producer is None:
            return False
            
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'RL_DECISION',
                'data': {
                    'action': action,
                    'confidence': confidence,
                    'state_before': state_before,
                    'state_after': state_after
                }
            }
            
            self.producer.send(self.topic, key='rl_decision', value=event)
            logger.info(f"📤 RL Decision: {action} (confidence: {confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send decision: {e}")
            return False
    
    def close(self):
        """Close producer connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


if __name__ == "__main__":
    # Test producer
    producer = QueueEventProducer()
    
    if producer.producer:
        # Send test event
        test_state = {
            'num_tellers': 3,
            'queue_length': 15,
            'avg_wait': 10.5,
            'renege_rate': 0.05
        }
        
        success = producer.send_queue_state(test_state)
        print(f"Test event sent: {success}")
        
        producer.close()
    else:
        print("❌ Kafka not available - start Kafka first!")
