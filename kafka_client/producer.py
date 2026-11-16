from kafka import KafkaProducer
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')

def produce_logs(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    
    print(f"Sending {len(lines)} log messages to Kafka...")
    for line in lines:
        producer.send('app-logs', line.strip().encode('utf-8'))
        time.sleep(0.01)
    
    producer.flush()
    print("All messages sent successfully!")

if __name__ == "__main__":
    produce_logs('./logs/sample_logs.txt')