"""
Kafka Consumer for RL Agent Decisions
Consumes queue events and logs RL agent decisions
"""

from kafka import KafkaConsumer
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueEventConsumer:
    """Consumes queue simulation events from Kafka."""
    
    def __init__(self, bootstrap_servers='localhost:9092', topic='queue-events', group_id='rl-agent-group'):
        self.topic = topic
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',  # Start from latest messages
                enable_auto_commit=True
            )
            logger.info(f"✅ Kafka Consumer connected to {bootstrap_servers}")
            logger.info(f"📥 Listening to topic: {topic}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            self.consumer = None
    
    def consume_events(self, callback=None):
        """
        Consume events from Kafka.
        
        Args:
            callback: Optional function to call for each event
        """
        if self.consumer is None:
            logger.error("Consumer not initialized")
            return
        
        logger.info("🎧 Starting to consume events...")
        
        try:
            for message in self.consumer:
                event = message.value
                event_type = event.get('event_type', 'UNKNOWN')
                timestamp = event.get('timestamp', '')
                data = event.get('data', {})
                
                # Log event
                if event_type == 'QUEUE_STATE':
                    logger.info(f"📊 Queue State: {data.get('queue_length', 0)} customers, "
                              f"{data.get('num_tellers', 0)} tellers")
                
                elif event_type == 'ARRIVAL':
                    logger.info(f"👥 Arrivals: {data.get('num_arrivals', 0)} customers at hour {data.get('hour', 0)}")
                
                elif event_type == 'RL_DECISION':
                    action = data.get('action', 'UNKNOWN')
                    confidence = data.get('confidence', 0)
                    logger.info(f"🤖 RL Decision: {action} (confidence: {confidence:.2f})")
                    
                    state_before = data.get('state_before', {})
                    state_after = data.get('state_after', {})
                    logger.info(f"   Before: {state_before.get('num_tellers', 0)} tellers, "
                              f"{state_before.get('queue_length', 0)} queue")
                    logger.info(f"   After:  {state_after.get('num_tellers', 0)} tellers, "
                              f"{state_after.get('queue_length', 0)} queue")
                
                # Call custom callback if provided
                if callback:
                    callback(event)
                    
        except KeyboardInterrupt:
            logger.info("⏹️  Consumer stopped by user")
        except Exception as e:
            logger.error(f"Error consuming events: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")


if __name__ == "__main__":
    # Run consumer
    consumer = QueueEventConsumer()
    
    if consumer.consumer:
        print("=" * 70)
        print(" KAFKA CONSUMER - Real-Time Queue Events")
        print("=" * 70)
        print("\nListening for events... (Press Ctrl+C to stop)\n")
        
        consumer.consume_events()
    else:
        print("❌ Kafka not available - start Kafka first!")
        print("\nTo start Kafka:")
        print("1. cd c:\\Users\\mukun\\Desktop\\el\\7thsem")
        print("2. docker-compose up -d")
