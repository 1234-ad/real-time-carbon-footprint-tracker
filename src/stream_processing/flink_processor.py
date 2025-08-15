"""
Apache Flink stream processor for real-time carbon footprint analytics.
Handles windowing, aggregations, anomaly detection, and ML inference.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction, FilterFunction, KeyedProcessFunction
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor
from pyflink.common.time import Time
import redis
import joblib

@dataclass
class CarbonEvent:
    """Carbon emission event structure"""
    device_id: str
    timestamp: datetime
    location: Dict[str, float]
    emission_type: str
    consumption_value: float
    consumption_unit: str
    carbon_factor: float
    calculated_emissions: float
    metadata: Dict

@dataclass
class AggregatedEmissions:
    """Aggregated carbon emissions for a time window"""
    window_start: datetime
    window_end: datetime
    total_emissions: float
    avg_emissions: float
    device_count: int
    emission_types: Dict[str, float]
    location_clusters: List[Dict]
    anomaly_score: float

class CarbonEventParser(MapFunction):
    """Parse JSON carbon events from Kafka"""
    
    def map(self, value: str) -> CarbonEvent:
        try:
            data = json.loads(value)
            return CarbonEvent(
                device_id=data['device_id'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                location=data['location'],
                emission_type=data['emission_type'],
                consumption_value=data['consumption_value'],
                consumption_unit=data['consumption_unit'],
                carbon_factor=data['carbon_factor'],
                calculated_emissions=data['calculated_emissions'],
                metadata=data['metadata']
            )
        except Exception as e:
            logging.error(f"Failed to parse carbon event: {e}")
            return None

class AnomalyDetectionFilter(FilterFunction):
    """Filter events that show anomalous carbon emission patterns"""
    
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.threshold_multiplier = 3.0  # Standard deviations
    
    def filter(self, event: CarbonEvent) -> bool:
        if not event:
            return False
            
        try:
            # Get historical statistics for this device
            stats_key = f"device_stats:{event.device_id}"
            stats = self.redis_client.hgetall(stats_key)
            
            if not stats:
                # First time seeing this device, store current value
                self.redis_client.hset(stats_key, mapping={
                    'mean': event.calculated_emissions,
                    'std': 0.0,
                    'count': 1
                })
                return False
            
            # Calculate if current emission is anomalous
            mean = float(stats[b'mean'])
            std = float(stats[b'std'])
            count = int(stats[b'count'])
            
            if std > 0:
                z_score = abs(event.calculated_emissions - mean) / std
                is_anomaly = z_score > self.threshold_multiplier
            else:
                is_anomaly = False
            
            # Update running statistics
            new_count = count + 1
            new_mean = (mean * count + event.calculated_emissions) / new_count
            new_variance = ((count - 1) * std**2 + (event.calculated_emissions - mean) * (event.calculated_emissions - new_mean)) / count
            new_std = np.sqrt(max(0, new_variance))
            
            self.redis_client.hset(stats_key, mapping={
                'mean': new_mean,
                'std': new_std,
                'count': new_count
            })
            
            return is_anomaly
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            return False

class CarbonAggregationFunction(KeyedProcessFunction):
    """Aggregate carbon emissions by time windows and location"""
    
    def __init__(self, window_size_minutes: int = 5):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.emissions_state = None
        self.timer_state = None
    
    def open(self, runtime_context):
        # State to store emissions for current window
        emissions_descriptor = ListStateDescriptor(
            "emissions",
            Types.PICKLED_BYTE_ARRAY()
        )
        self.emissions_state = runtime_context.get_list_state(emissions_descriptor)
        
        # State to track window timer
        timer_descriptor = ValueStateDescriptor(
            "timer",
            Types.LONG()
        )
        self.timer_state = runtime_context.get_state(timer_descriptor)
    
    def process_element(self, event: CarbonEvent, ctx, out):
        try:
            # Add event to current window
            self.emissions_state.add(event)
            
            # Calculate window boundaries
            window_start = self._get_window_start(event.timestamp)
            window_end = window_start + self.window_size
            
            # Set timer for window end if not already set
            current_timer = self.timer_state.value()
            window_end_timestamp = int(window_end.timestamp() * 1000)
            
            if current_timer is None or current_timer != window_end_timestamp:
                ctx.timer_service().register_event_time_timer(window_end_timestamp)
                self.timer_state.update(window_end_timestamp)
                
        except Exception as e:
            logging.error(f"Error processing carbon event: {e}")
    
    def on_timer(self, timestamp: int, ctx, out):
        try:
            # Get all events in current window
            events = list(self.emissions_state.get())
            
            if not events:
                return
            
            # Calculate aggregations
            total_emissions = sum(event.calculated_emissions for event in events)
            avg_emissions = total_emissions / len(events)
            device_count = len(set(event.device_id for event in events))
            
            # Group by emission type
            emission_types = {}
            for event in events:
                emission_types[event.emission_type] = emission_types.get(event.emission_type, 0) + event.calculated_emissions
            
            # Cluster locations (simplified k-means)
            locations = [(event.location['lat'], event.location['lng']) for event in events]
            location_clusters = self._cluster_locations(locations)
            
            # Calculate anomaly score for the window
            emissions_values = [event.calculated_emissions for event in events]
            anomaly_score = self._calculate_window_anomaly_score(emissions_values)
            
            # Create aggregated result
            window_start = datetime.fromtimestamp(timestamp / 1000) - self.window_size
            window_end = datetime.fromtimestamp(timestamp / 1000)
            
            aggregated = AggregatedEmissions(
                window_start=window_start,
                window_end=window_end,
                total_emissions=round(total_emissions, 3),
                avg_emissions=round(avg_emissions, 3),
                device_count=device_count,
                emission_types=emission_types,
                location_clusters=location_clusters,
                anomaly_score=round(anomaly_score, 3)
            )
            
            # Output aggregated result
            out.collect(json.dumps(aggregated.__dict__, default=str))
            
            # Clear state for next window
            self.emissions_state.clear()
            self.timer_state.clear()
            
        except Exception as e:
            logging.error(f"Error in timer processing: {e}")
    
    def _get_window_start(self, timestamp: datetime) -> datetime:
        """Get the start of the time window for given timestamp"""
        minutes = timestamp.minute
        window_minutes = minutes - (minutes % self.window_size.seconds // 60)
        return timestamp.replace(minute=window_minutes, second=0, microsecond=0)
    
    def _cluster_locations(self, locations: List[Tuple[float, float]], k: int = 3) -> List[Dict]:
        """Simple k-means clustering for location data"""
        if len(locations) <= k:
            return [{'lat': lat, 'lng': lng, 'count': 1} for lat, lng in locations]
        
        # Initialize centroids
        locations_array = np.array(locations)
        centroids = locations_array[np.random.choice(len(locations_array), k, replace=False)]
        
        # Simple k-means iteration
        for _ in range(10):  # Max 10 iterations
            # Assign points to clusters
            distances = np.sqrt(((locations_array - centroids[:, np.newaxis])**2).sum(axis=2))
            clusters = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.array([locations_array[clusters == i].mean(axis=0) for i in range(k)])
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Return cluster information
        result = []
        for i in range(k):
            cluster_points = locations_array[clusters == i]
            if len(cluster_points) > 0:
                result.append({
                    'lat': float(centroids[i][0]),
                    'lng': float(centroids[i][1]),
                    'count': len(cluster_points)
                })
        
        return result
    
    def _calculate_window_anomaly_score(self, emissions: List[float]) -> float:
        """Calculate anomaly score for the entire window"""
        if len(emissions) < 2:
            return 0.0
        
        mean_emission = np.mean(emissions)
        std_emission = np.std(emissions)
        
        if std_emission == 0:
            return 0.0
        
        # Calculate coefficient of variation as anomaly indicator
        cv = std_emission / mean_emission
        
        # Normalize to 0-1 scale
        return min(cv, 1.0)

class MLInferenceFunction(MapFunction):
    """Apply ML models for carbon emission predictions and insights"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
    
    def open(self, runtime_context):
        # Load pre-trained model
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Loaded ML model from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load ML model: {e}")
            self.model = None
    
    def map(self, aggregated_data: str) -> str:
        if not self.model:
            return aggregated_data
        
        try:
            data = json.loads(aggregated_data)
            
            # Extract features for prediction
            features = [
                data['total_emissions'],
                data['avg_emissions'],
                data['device_count'],
                len(data['emission_types']),
                data['anomaly_score']
            ]
            
            # Make prediction (e.g., next hour emissions)
            prediction = self.model.predict([features])[0]
            
            # Add prediction to data
            data['ml_predictions'] = {
                'next_hour_emissions': round(float(prediction), 3),
                'confidence': 0.85,  # Would come from model
                'model_version': '1.0.0'
            }
            
            # Store prediction in Redis for API access
            prediction_key = f"prediction:{data['window_end']}"
            self.redis_client.setex(
                prediction_key,
                3600,  # 1 hour TTL
                json.dumps(data['ml_predictions'])
            )
            
            return json.dumps(data)
            
        except Exception as e:
            logging.error(f"Error in ML inference: {e}")
            return aggregated_data

class CarbonStreamProcessor:
    """Main stream processing application"""
    
    def __init__(self):
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_parallelism(4)
        self.env.enable_checkpointing(60000)  # Checkpoint every minute
        
        # Kafka configuration
        self.kafka_props = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'carbon-stream-processor',
            'auto.offset.reset': 'latest'
        }
    
    def create_kafka_source(self, topic: str) -> FlinkKafkaConsumer:
        """Create Kafka consumer for given topic"""
        return FlinkKafkaConsumer(
            topic,
            SimpleStringSchema(),
            self.kafka_props
        )
    
    def create_kafka_sink(self, topic: str) -> FlinkKafkaProducer:
        """Create Kafka producer for given topic"""
        return FlinkKafkaProducer(
            topic,
            SimpleStringSchema(),
            self.kafka_props
        )
    
    def run(self):
        """Run the stream processing pipeline"""
        try:
            # Create data sources
            iot_source = self.create_kafka_source('carbon-tracker-iot-sensors')
            grid_source = self.create_kafka_source('carbon-tracker-energy-grid')
            transport_source = self.create_kafka_source('carbon-tracker-transportation')
            
            # Parse events
            iot_events = self.env.add_source(iot_source).map(CarbonEventParser())
            grid_events = self.env.add_source(grid_source).map(CarbonEventParser())
            transport_events = self.env.add_source(transport_source).map(CarbonEventParser())
            
            # Union all streams
            all_events = iot_events.union(grid_events).union(transport_events)
            
            # Filter out invalid events
            valid_events = all_events.filter(lambda event: event is not None)
            
            # Anomaly detection branch
            anomalies = valid_events.filter(AnomalyDetectionFilter())
            anomaly_sink = self.create_kafka_sink('carbon-tracker-anomalies')
            anomalies.map(lambda event: json.dumps(event.__dict__, default=str)).add_sink(anomaly_sink)
            
            # Main aggregation pipeline
            aggregated = (valid_events
                         .key_by(lambda event: f"{event.emission_type}_{self._get_location_key(event.location)}")
                         .process(CarbonAggregationFunction(window_size_minutes=5)))
            
            # ML inference
            ml_enhanced = aggregated.map(MLInferenceFunction('models/carbon_prediction_model.pkl'))
            
            # Output sinks
            aggregation_sink = self.create_kafka_sink('carbon-tracker-aggregated')
            ml_enhanced.add_sink(aggregation_sink)
            
            # Real-time dashboard sink
            dashboard_sink = self.create_kafka_sink('carbon-tracker-dashboard')
            ml_enhanced.add_sink(dashboard_sink)
            
            # Execute the pipeline
            self.env.execute("Carbon Footprint Stream Processor")
            
        except Exception as e:
            logging.error(f"Error in stream processing pipeline: {e}")
            raise
    
    def _get_location_key(self, location: Dict[str, float]) -> str:
        """Create location key for partitioning (grid-based)"""
        lat_grid = int(location['lat'] * 100) // 10  # ~1km grid
        lng_grid = int(location['lng'] * 100) // 10
        return f"{lat_grid}_{lng_grid}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    processor = CarbonStreamProcessor()
    processor.run()