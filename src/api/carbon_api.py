"""
High-performance FastAPI service for carbon footprint tracking system.
Provides real-time analytics, predictions, and optimization recommendations.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import redis.asyncio as redis
from kafka import KafkaProducer
import joblib
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import uvicorn

# Metrics
API_REQUESTS = Counter('carbon_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('carbon_api_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('carbon_api_active_connections', 'Active WebSocket connections')
PREDICTION_ACCURACY = Gauge('carbon_api_prediction_accuracy', 'Model prediction accuracy')

# Pydantic Models
class CarbonEmissionRequest(BaseModel):
    device_id: str
    emission_type: str = Field(..., description="Type of emission: electricity_grid, natural_gas, gasoline, etc.")
    consumption_value: float = Field(..., gt=0, description="Consumption amount")
    consumption_unit: str = Field(..., description="Unit of consumption: kWh, liters, etc.")
    location: Dict[str, float] = Field(..., description="Latitude and longitude")
    metadata: Optional[Dict] = Field(default_factory=dict)

class CarbonEmissionResponse(BaseModel):
    device_id: str
    timestamp: datetime
    calculated_emissions: float
    carbon_factor: float
    emission_type: str
    location: Dict[str, float]
    sustainability_score: float
    recommendations: List[str]

class AggregatedEmissionsResponse(BaseModel):
    time_period: str
    total_emissions: float
    avg_emissions: float
    device_count: int
    emission_breakdown: Dict[str, float]
    trend_analysis: Dict[str, Union[float, str]]
    anomalies_detected: int

class PredictionRequest(BaseModel):
    time_horizon_hours: int = Field(default=24, ge=1, le=168)
    device_ids: Optional[List[str]] = None
    emission_types: Optional[List[str]] = None
    location_bounds: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Union[float, str]]]
    confidence_interval: Dict[str, float]
    model_version: str
    generated_at: datetime

class OptimizationRequest(BaseModel):
    target_reduction_percent: float = Field(..., ge=5, le=50)
    time_horizon_days: int = Field(default=30, ge=1, le=365)
    priority_areas: Optional[List[str]] = None

class OptimizationResponse(BaseModel):
    current_emissions: float
    target_emissions: float
    reduction_strategies: List[Dict[str, Union[str, float]]]
    estimated_savings: Dict[str, float]
    implementation_timeline: List[Dict[str, str]]

# FastAPI App
app = FastAPI(
    title="Carbon Footprint Tracker API",
    description="Real-time carbon emission tracking, analytics, and optimization",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global connections
influx_client = None
redis_client = None
kafka_producer = None
ml_model = None

# Carbon emission factors (kg CO2 per unit)
EMISSION_FACTORS = {
    'electricity_grid': 0.233,  # kg CO2/kWh (US average)
    'natural_gas': 0.202,       # kg CO2/kWh
    'gasoline': 2.31,           # kg CO2/liter
    'diesel': 2.68,             # kg CO2/liter
    'heating_oil': 2.52,        # kg CO2/liter
    'propane': 1.51,            # kg CO2/liter
    'coal': 0.820,              # kg CO2/kWh
    'renewable': 0.0,           # kg CO2/kWh (solar, wind, hydro)
}

@app.on_event("startup")
async def startup_event():
    """Initialize connections and load models"""
    global influx_client, redis_client, kafka_producer, ml_model
    
    try:
        # InfluxDB connection
        influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="carbon-tracker-super-secret-token",
            org="carbon-tracker"
        )
        
        # Redis connection
        redis_client = redis.from_url("redis://localhost:6379")
        
        # Kafka producer
        kafka_producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Load ML model
        try:
            ml_model = joblib.load('models/carbon_prediction_model.pkl')
            logging.info("ML model loaded successfully")
        except Exception as e:
            logging.warning(f"Could not load ML model: {e}")
            ml_model = None
        
        logging.info("API startup completed successfully")
        
    except Exception as e:
        logging.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections"""
    global influx_client, redis_client, kafka_producer
    
    if influx_client:
        influx_client.close()
    if redis_client:
        await redis_client.close()
    if kafka_producer:
        kafka_producer.close()

# Dependency injection
async def get_redis():
    return redis_client

async def get_influx():
    return influx_client

# API Endpoints

@app.post("/emissions/calculate", response_model=CarbonEmissionResponse)
async def calculate_emissions(
    request: CarbonEmissionRequest,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Calculate carbon emissions for a given consumption"""
    API_REQUESTS.labels(method="POST", endpoint="/emissions/calculate").inc()
    
    with REQUEST_DURATION.time():
        try:
            # Get carbon factor
            carbon_factor = EMISSION_FACTORS.get(request.emission_type, 0.233)
            
            # Calculate emissions
            calculated_emissions = request.consumption_value * carbon_factor
            
            # Calculate sustainability score (0-100)
            # Lower emissions = higher score
            max_expected_emission = 100  # Baseline for scoring
            sustainability_score = max(0, 100 - (calculated_emissions / max_expected_emission * 100))
            
            # Generate recommendations
            recommendations = await generate_recommendations(
                request.emission_type, 
                calculated_emissions, 
                sustainability_score
            )
            
            # Create response
            response = CarbonEmissionResponse(
                device_id=request.device_id,
                timestamp=datetime.utcnow(),
                calculated_emissions=round(calculated_emissions, 3),
                carbon_factor=carbon_factor,
                emission_type=request.emission_type,
                location=request.location,
                sustainability_score=round(sustainability_score, 1),
                recommendations=recommendations
            )
            
            # Store in cache for quick access
            cache_key = f"emission:{request.device_id}:{datetime.utcnow().strftime('%Y%m%d%H')}"
            await redis_client.setex(cache_key, 3600, response.json())
            
            # Send to Kafka for stream processing
            background_tasks.add_task(
                send_to_kafka,
                "carbon-tracker-api-emissions",
                response.dict()
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Error calculating emissions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/emissions/aggregated", response_model=AggregatedEmissionsResponse)
async def get_aggregated_emissions(
    time_period: str = Query("1h", description="Time period: 1h, 6h, 1d, 7d, 30d"),
    emission_type: Optional[str] = Query(None, description="Filter by emission type"),
    location_bounds: Optional[str] = Query(None, description="Lat,lng,radius in km"),
    influx_client: InfluxDBClient = Depends(get_influx)
):
    """Get aggregated carbon emissions for specified time period"""
    API_REQUESTS.labels(method="GET", endpoint="/emissions/aggregated").inc()
    
    with REQUEST_DURATION.time():
        try:
            # Parse time period
            time_map = {
                "1h": "1h", "6h": "6h", "1d": "1d", 
                "7d": "7d", "30d": "30d"
            }
            
            if time_period not in time_map:
                raise HTTPException(status_code=400, detail="Invalid time period")
            
            # Build InfluxDB query
            query = f'''
            from(bucket: "carbon-emissions")
            |> range(start: -{time_map[time_period]})
            |> filter(fn: (r) => r["_measurement"] == "carbon_events")
            '''
            
            if emission_type:
                query += f'|> filter(fn: (r) => r["emission_type"] == "{emission_type}")'
            
            query += '''
            |> group(columns: ["emission_type"])
            |> aggregateWindow(every: 1h, fn: sum, createEmpty: false)
            |> yield(name: "sum")
            '''
            
            # Execute query
            result = influx_client.query_api().query_data_frame(query)
            
            if result.empty:
                return AggregatedEmissionsResponse(
                    time_period=time_period,
                    total_emissions=0.0,
                    avg_emissions=0.0,
                    device_count=0,
                    emission_breakdown={},
                    trend_analysis={"trend": "no_data", "change_percent": 0.0},
                    anomalies_detected=0
                )
            
            # Calculate aggregations
            total_emissions = result['_value'].sum()
            avg_emissions = result['_value'].mean()
            device_count = result['device_id'].nunique() if 'device_id' in result.columns else 0
            
            # Emission breakdown by type
            emission_breakdown = result.groupby('emission_type')['_value'].sum().to_dict()
            
            # Trend analysis
            trend_analysis = await calculate_trend_analysis(result)
            
            # Anomaly detection
            anomalies_detected = await detect_anomalies_in_period(result)
            
            return AggregatedEmissionsResponse(
                time_period=time_period,
                total_emissions=round(total_emissions, 3),
                avg_emissions=round(avg_emissions, 3),
                device_count=device_count,
                emission_breakdown=emission_breakdown,
                trend_analysis=trend_analysis,
                anomalies_detected=anomalies_detected
            )
            
        except Exception as e:
            logging.error(f"Error getting aggregated emissions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predictions/emissions", response_model=PredictionResponse)
async def predict_emissions(
    request: PredictionRequest,
    redis_client: redis.Redis = Depends(get_redis)
):
    """Predict future carbon emissions using ML models"""
    API_REQUESTS.labels(method="POST", endpoint="/predictions/emissions").inc()
    
    with REQUEST_DURATION.time():
        try:
            if not ml_model:
                raise HTTPException(status_code=503, detail="ML model not available")
            
            # Check cache first
            cache_key = f"prediction:{hash(str(request.dict()))}"
            cached_result = await redis_client.get(cache_key)
            
            if cached_result:
                return PredictionResponse.parse_raw(cached_result)
            
            # Generate predictions
            predictions = []
            current_time = datetime.utcnow()
            
            for hour in range(request.time_horizon_hours):
                future_time = current_time + timedelta(hours=hour + 1)
                
                # Create feature vector (simplified)
                features = [
                    hour,  # hour offset
                    future_time.hour,  # hour of day
                    future_time.weekday(),  # day of week
                    future_time.month,  # month
                    np.sin(2 * np.pi * future_time.hour / 24),  # cyclical hour
                    np.cos(2 * np.pi * future_time.hour / 24),
                ]
                
                # Make prediction (simplified - real model would use more features)
                try:
                    if hasattr(ml_model, 'predict'):
                        prediction = float(ml_model.predict([features])[0])
                    else:
                        # Fallback to simple trend-based prediction
                        base_emission = 50.0  # Base hourly emission
                        hour_factor = 1 + 0.3 * np.sin(2 * np.pi * future_time.hour / 24)
                        prediction = base_emission * hour_factor
                    
                    predictions.append({
                        "timestamp": future_time.isoformat(),
                        "predicted_emissions": round(prediction, 3),
                        "hour_offset": hour + 1
                    })
                    
                except Exception as e:
                    logging.warning(f"Prediction error for hour {hour}: {e}")
                    continue
            
            # Calculate confidence interval
            prediction_values = [p["predicted_emissions"] for p in predictions]
            confidence_interval = {
                "lower_bound": round(np.percentile(prediction_values, 25), 3),
                "upper_bound": round(np.percentile(prediction_values, 75), 3),
                "mean": round(np.mean(prediction_values), 3)
            }
            
            response = PredictionResponse(
                predictions=predictions,
                confidence_interval=confidence_interval,
                model_version="1.0.0",
                generated_at=current_time
            )
            
            # Cache result for 15 minutes
            await redis_client.setex(cache_key, 900, response.json())
            
            return response
            
        except Exception as e:
            logging.error(f"Error predicting emissions: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimization/recommendations", response_model=OptimizationResponse)
async def get_optimization_recommendations(request: OptimizationRequest):
    """Get carbon emission reduction recommendations"""
    API_REQUESTS.labels(method="POST", endpoint="/optimization/recommendations").inc()
    
    with REQUEST_DURATION.time():
        try:
            # Simulate current emissions (would come from real data)
            current_emissions = 1000.0  # kg CO2 per month
            target_emissions = current_emissions * (1 - request.target_reduction_percent / 100)
            
            # Generate reduction strategies
            strategies = [
                {
                    "strategy": "Switch to renewable energy",
                    "potential_reduction_percent": 40.0,
                    "implementation_cost": 5000.0,
                    "payback_period_months": 24,
                    "priority": "high"
                },
                {
                    "strategy": "Improve HVAC efficiency",
                    "potential_reduction_percent": 15.0,
                    "implementation_cost": 2000.0,
                    "payback_period_months": 12,
                    "priority": "medium"
                },
                {
                    "strategy": "LED lighting upgrade",
                    "potential_reduction_percent": 8.0,
                    "implementation_cost": 800.0,
                    "payback_period_months": 6,
                    "priority": "high"
                },
                {
                    "strategy": "Smart energy management system",
                    "potential_reduction_percent": 12.0,
                    "implementation_cost": 3000.0,
                    "payback_period_months": 18,
                    "priority": "medium"
                }
            ]
            
            # Calculate estimated savings
            annual_emission_cost = current_emissions * 12 * 0.05  # $0.05 per kg CO2
            estimated_savings = {
                "annual_cost_savings": round(annual_emission_cost * request.target_reduction_percent / 100, 2),
                "annual_emission_reduction": round(current_emissions * 12 * request.target_reduction_percent / 100, 2),
                "carbon_credits_potential": round(current_emissions * 12 * request.target_reduction_percent / 100 * 0.02, 2)
            }
            
            # Implementation timeline
            timeline = [
                {"phase": "Assessment and Planning", "duration": "2 weeks", "description": "Energy audit and strategy finalization"},
                {"phase": "Quick Wins Implementation", "duration": "1 month", "description": "LED upgrades and basic efficiency measures"},
                {"phase": "Major System Upgrades", "duration": "3 months", "description": "HVAC and renewable energy installation"},
                {"phase": "Smart System Integration", "duration": "1 month", "description": "Smart controls and monitoring setup"},
                {"phase": "Optimization and Monitoring", "duration": "Ongoing", "description": "Continuous improvement and tracking"}
            ]
            
            return OptimizationResponse(
                current_emissions=current_emissions,
                target_emissions=target_emissions,
                reduction_strategies=strategies,
                estimated_savings=estimated_savings,
                implementation_timeline=timeline
            )
            
        except Exception as e:
            logging.error(f"Error generating optimization recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/dashboard")
async def get_dashboard_data(
    redis_client: redis.Redis = Depends(get_redis)
):
    """Get real-time dashboard data"""
    API_REQUESTS.labels(method="GET", endpoint="/analytics/dashboard").inc()
    
    try:
        # Get cached dashboard data
        dashboard_data = await redis_client.get("dashboard:realtime")
        
        if dashboard_data:
            return json.loads(dashboard_data)
        
        # Generate dashboard data (would come from real-time processing)
        current_time = datetime.utcnow()
        
        dashboard = {
            "timestamp": current_time.isoformat(),
            "real_time_metrics": {
                "current_emission_rate": round(np.random.normal(45, 10), 2),
                "daily_total": round(np.random.normal(1200, 200), 2),
                "active_devices": np.random.randint(150, 200),
                "anomalies_detected": np.random.randint(0, 5)
            },
            "emission_breakdown": {
                "electricity": 65.2,
                "transportation": 20.1,
                "heating": 10.5,
                "industrial": 4.2
            },
            "top_emitters": [
                {"device_id": "industrial_equipment_1", "emissions": 45.2},
                {"device_id": "hvac_system_main", "emissions": 32.1},
                {"device_id": "vehicle_fleet_01", "emissions": 28.7}
            ],
            "sustainability_score": round(np.random.normal(75, 10), 1),
            "trends": {
                "hourly_change": round(np.random.normal(0, 5), 2),
                "daily_change": round(np.random.normal(0, 10), 2),
                "weekly_change": round(np.random.normal(0, 15), 2)
            }
        }
        
        # Cache for 30 seconds
        await redis_client.setex("dashboard:realtime", 30, json.dumps(dashboard))
        
        return dashboard
        
    except Exception as e:
        logging.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "services": {
            "influxdb": "connected" if influx_client else "disconnected",
            "redis": "connected" if redis_client else "disconnected",
            "kafka": "connected" if kafka_producer else "disconnected",
            "ml_model": "loaded" if ml_model else "not_loaded"
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        iter([generate_latest().decode('utf-8')]),
        media_type="text/plain"
    )

# Helper functions

async def generate_recommendations(emission_type: str, emissions: float, score: float) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    if score < 50:
        recommendations.append("⚠️ High emissions detected - immediate action recommended")
    
    if emission_type == "electricity_grid":
        if emissions > 30:
            recommendations.append("💡 Consider switching to renewable energy sources")
            recommendations.append("🏠 Upgrade to energy-efficient appliances")
        recommendations.append("📊 Install smart meters for better monitoring")
    
    elif emission_type == "transportation":
        recommendations.append("🚗 Consider electric or hybrid vehicles")
        recommendations.append("🚌 Use public transportation when possible")
        recommendations.append("🚴 Bike or walk for short distances")
    
    elif emission_type == "natural_gas":
        recommendations.append("🌡️ Optimize heating/cooling schedules")
        recommendations.append("🏠 Improve building insulation")
    
    if score > 80:
        recommendations.append("✅ Excellent sustainability performance!")
    
    return recommendations

async def calculate_trend_analysis(data: pd.DataFrame) -> Dict[str, Union[float, str]]:
    """Calculate emission trends"""
    if len(data) < 2:
        return {"trend": "insufficient_data", "change_percent": 0.0}
    
    # Simple trend calculation
    recent_avg = data.tail(len(data)//2)['_value'].mean()
    older_avg = data.head(len(data)//2)['_value'].mean()
    
    if older_avg == 0:
        return {"trend": "no_baseline", "change_percent": 0.0}
    
    change_percent = ((recent_avg - older_avg) / older_avg) * 100
    
    if change_percent > 5:
        trend = "increasing"
    elif change_percent < -5:
        trend = "decreasing"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "change_percent": round(change_percent, 2)
    }

async def detect_anomalies_in_period(data: pd.DataFrame) -> int:
    """Detect anomalies in emission data"""
    if len(data) < 10:
        return 0
    
    # Simple statistical anomaly detection
    mean_emission = data['_value'].mean()
    std_emission = data['_value'].std()
    
    if std_emission == 0:
        return 0
    
    # Count values beyond 2 standard deviations
    anomalies = data[abs(data['_value'] - mean_emission) > 2 * std_emission]
    return len(anomalies)

async def send_to_kafka(topic: str, data: dict):
    """Send data to Kafka topic"""
    try:
        if kafka_producer:
            kafka_producer.send(topic, data)
    except Exception as e:
        logging.error(f"Error sending to Kafka: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "carbon_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )