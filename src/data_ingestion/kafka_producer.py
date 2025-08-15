"""
Real-time data producer for carbon footprint tracking system.
Handles IoT sensor data, energy grid data, and external API integrations.
"""

import json
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
from kafka.errors import KafkaError
import aiohttp
import numpy as np
from prometheus_client import Counter, Histogram, Gauge

# Metrics
MESSAGES_SENT = Counter('carbon_tracker_messages_sent_total', 'Total messages sent', ['source', 'topic'])
MESSAGE_SIZE = Histogram('carbon_tracker_message_size_bytes', 'Message size in bytes', ['source'])
ACTIVE_CONNECTIONS = Gauge('carbon_tracker_active_connections', 'Active connections', ['source'])

@dataclass
class CarbonEmissionEvent:
    """Standard carbon emission event structure"""
    device_id: str
    timestamp: datetime
    location: Dict[str, float]  # lat, lng
    emission_type: str  # electricity, gas, transportation, etc.
    consumption_value: float
    consumption_unit: str
    carbon_factor: float  # kg CO2 per unit
    calculated_emissions: float  # kg CO2
    metadata: Dict
    
    def to_json(self) -> str:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)

class CarbonDataProducer:
    """High-performance Kafka producer for carbon emission data"""
    
    def __init__(self, kafka_config: Dict, topics: Dict[str, str]):
        self.kafka_config = kafka_config
        self.topics = topics
        self.producer = None
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Carbon emission factors (kg CO2 per unit)
        self.emission_factors = {
            'electricity_grid': 0.233,  # kg CO2/kWh (US average)
            'natural_gas': 0.202,       # kg CO2/kWh
            'gasoline': 2.31,           # kg CO2/liter
            'diesel': 2.68,             # kg CO2/liter
            'heating_oil': 2.52,        # kg CO2/liter
        }
    
    async def initialize(self):
        """Initialize Kafka producer with optimized settings"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config['bootstrap_servers'],
                value_serializer=lambda v: v.encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                # Performance optimizations
                batch_size=32768,  # 32KB batches
                linger_ms=10,      # Wait 10ms for batching
                compression_type='snappy',
                acks='1',          # Wait for leader acknowledgment
                retries=3,
                max_in_flight_requests_per_connection=5,
                # Reliability settings
                enable_idempotence=True,
                delivery_timeout_ms=120000,
            )
            self.running = True
            self.logger.info("Kafka producer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka producer: {e}")
            raise
    
    async def produce_iot_sensor_data(self):
        """Simulate IoT sensor data from smart meters and devices"""
        device_types = [
            'smart_meter_electricity',
            'smart_meter_gas',
            'hvac_system',
            'industrial_equipment',
            'vehicle_telematics'
        ]
        
        while self.running:
            try:
                for device_id in range(1, 101):  # 100 devices
                    device_type = np.random.choice(device_types)
                    
                    # Generate realistic consumption data
                    if device_type == 'smart_meter_electricity':
                        consumption = np.random.normal(25, 5)  # kWh
                        unit = 'kWh'
                        emission_type = 'electricity_grid'
                    elif device_type == 'smart_meter_gas':
                        consumption = np.random.normal(15, 3)  # kWh equivalent
                        unit = 'kWh'
                        emission_type = 'natural_gas'
                    elif device_type == 'hvac_system':
                        consumption = np.random.normal(45, 10)  # kWh
                        unit = 'kWh'
                        emission_type = 'electricity_grid'
                    elif device_type == 'industrial_equipment':
                        consumption = np.random.normal(150, 30)  # kWh
                        unit = 'kWh'
                        emission_type = 'electricity_grid'
                    else:  # vehicle_telematics
                        consumption = np.random.normal(8, 2)  # liters
                        unit = 'liters'
                        emission_type = 'gasoline'
                    
                    # Calculate emissions
                    carbon_factor = self.emission_factors.get(emission_type, 0.233)
                    calculated_emissions = consumption * carbon_factor
                    
                    # Create event
                    event = CarbonEmissionEvent(
                        device_id=f"{device_type}_{device_id}",
                        timestamp=datetime.now(timezone.utc),
                        location={
                            'lat': 40.7128 + np.random.normal(0, 0.1),
                            'lng': -74.0060 + np.random.normal(0, 0.1)
                        },
                        emission_type=emission_type,
                        consumption_value=round(consumption, 2),
                        consumption_unit=unit,
                        carbon_factor=carbon_factor,
                        calculated_emissions=round(calculated_emissions, 3),
                        metadata={
                            'device_type': device_type,
                            'firmware_version': '1.2.3',
                            'signal_strength': np.random.randint(70, 100),
                            'battery_level': np.random.randint(20, 100)
                        }
                    )
                    
                    # Send to Kafka
                    await self.send_event(self.topics['iot_sensors'], event)
                    
                    # Metrics
                    MESSAGES_SENT.labels(source='iot_sensor', topic=self.topics['iot_sensors']).inc()
                    MESSAGE_SIZE.labels(source='iot_sensor').observe(len(event.to_json()))
                
                await asyncio.sleep(1)  # 1 second interval
                
            except Exception as e:
                self.logger.error(f"Error producing IoT sensor data: {e}")
                await asyncio.sleep(5)
    
    async def produce_energy_grid_data(self):
        """Fetch and produce real-time energy grid carbon intensity data"""
        api_endpoints = {
            'carbon_intensity': 'https://api.carbonintensity.org.uk/intensity',
            'energy_mix': 'https://api.eia.gov/v2/electricity/rto/fuel-type-data/data'
        }
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    # Fetch carbon intensity data
                    async with session.get(api_endpoints['carbon_intensity']) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for entry in data.get('data', []):
                                event = CarbonEmissionEvent(
                                    device_id='grid_carbon_intensity',
                                    timestamp=datetime.now(timezone.utc),
                                    location={'lat': 51.5074, 'lng': -0.1278},  # London
                                    emission_type='electricity_grid',
                                    consumption_value=1.0,
                                    consumption_unit='kWh',
                                    carbon_factor=entry['intensity']['actual'] / 1000,  # Convert to kg
                                    calculated_emissions=entry['intensity']['actual'] / 1000,
                                    metadata={
                                        'source': 'carbon_intensity_api',
                                        'forecast': entry['intensity']['forecast'],
                                        'index': entry['intensity']['index']
                                    }
                                )
                                
                                await self.send_event(self.topics['energy_grid'], event)
                                MESSAGES_SENT.labels(source='energy_grid', topic=self.topics['energy_grid']).inc()
                
                await asyncio.sleep(300)  # 5 minute interval
                
            except Exception as e:
                self.logger.error(f"Error producing energy grid data: {e}")
                await asyncio.sleep(60)
    
    async def produce_transportation_data(self):
        """Generate transportation emission events"""
        transport_modes = ['car', 'bus', 'train', 'plane', 'ship']
        
        while self.running:
            try:
                for vehicle_id in range(1, 51):  # 50 vehicles
                    mode = np.random.choice(transport_modes)
                    
                    # Generate realistic trip data
                    if mode == 'car':
                        distance = np.random.normal(25, 10)  # km
                        fuel_efficiency = np.random.normal(8, 1)  # L/100km
                        fuel_consumed = (distance / 100) * fuel_efficiency
                        emission_factor = 2.31  # kg CO2/L gasoline
                    elif mode == 'bus':
                        distance = np.random.normal(15, 5)
                        fuel_efficiency = np.random.normal(35, 5)  # L/100km
                        fuel_consumed = (distance / 100) * fuel_efficiency
                        emission_factor = 2.68  # kg CO2/L diesel
                    elif mode == 'train':
                        distance = np.random.normal(50, 20)
                        # Electric train - use electricity factor
                        energy_consumed = distance * 0.05  # kWh/km
                        fuel_consumed = energy_consumed
                        emission_factor = 0.233  # kg CO2/kWh
                    elif mode == 'plane':
                        distance = np.random.normal(500, 200)
                        fuel_efficiency = np.random.normal(3.5, 0.5)  # L/km
                        fuel_consumed = distance * fuel_efficiency
                        emission_factor = 2.52  # kg CO2/L jet fuel
                    else:  # ship
                        distance = np.random.normal(100, 50)
                        fuel_efficiency = np.random.normal(50, 10)  # L/100km
                        fuel_consumed = (distance / 100) * fuel_efficiency
                        emission_factor = 2.68  # kg CO2/L marine fuel
                    
                    calculated_emissions = fuel_consumed * emission_factor
                    
                    event = CarbonEmissionEvent(
                        device_id=f"{mode}_vehicle_{vehicle_id}",
                        timestamp=datetime.now(timezone.utc),
                        location={
                            'lat': 40.7128 + np.random.normal(0, 1),
                            'lng': -74.0060 + np.random.normal(0, 1)
                        },
                        emission_type='transportation',
                        consumption_value=round(fuel_consumed, 2),
                        consumption_unit='liters' if mode != 'train' else 'kWh',
                        carbon_factor=emission_factor,
                        calculated_emissions=round(calculated_emissions, 3),
                        metadata={
                            'transport_mode': mode,
                            'distance_km': round(distance, 1),
                            'passenger_count': np.random.randint(1, 5),
                            'route_type': np.random.choice(['urban', 'highway', 'mixed'])
                        }
                    )
                    
                    await self.send_event(self.topics['transportation'], event)
                    MESSAGES_SENT.labels(source='transportation', topic=self.topics['transportation']).inc()
                
                await asyncio.sleep(30)  # 30 second interval
                
            except Exception as e:
                self.logger.error(f"Error producing transportation data: {e}")
                await asyncio.sleep(10)
    
    async def send_event(self, topic: str, event: CarbonEmissionEvent):
        """Send event to Kafka topic with error handling"""
        try:
            future = self.producer.send(
                topic,
                key=event.device_id,
                value=event.to_json()
            )
            
            # Non-blocking send with callback
            future.add_callback(self._on_send_success)
            future.add_errback(self._on_send_error)
            
        except Exception as e:
            self.logger.error(f"Failed to send event to {topic}: {e}")
    
    def _on_send_success(self, record_metadata):
        """Callback for successful message delivery"""
        self.logger.debug(f"Message sent to {record_metadata.topic}:{record_metadata.partition}:{record_metadata.offset}")
    
    def _on_send_error(self, exception):
        """Callback for failed message delivery"""
        self.logger.error(f"Failed to send message: {exception}")
    
    async def start_all_producers(self):
        """Start all data producers concurrently"""
        await self.initialize()
        
        tasks = [
            asyncio.create_task(self.produce_iot_sensor_data()),
            asyncio.create_task(self.produce_energy_grid_data()),
            asyncio.create_task(self.produce_transportation_data())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            self.logger.info("Shutting down producers...")
            self.running = False
            if self.producer:
                self.producer.close()

# Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092']
}

TOPICS = {
    'iot_sensors': 'carbon-tracker-iot-sensors',
    'energy_grid': 'carbon-tracker-energy-grid',
    'transportation': 'carbon-tracker-transportation'
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    producer = CarbonDataProducer(KAFKA_CONFIG, TOPICS)
    asyncio.run(producer.start_all_producers())