import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 }, // Ramp up to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    http_req_failed: ['rate<0.1'],    // Error rate must be below 10%
    errors: ['rate<0.1'],             // Custom error rate must be below 10%
  },
};

// Base URL configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Test data
const testDevices = [
  'device-001', 'device-002', 'device-003', 'device-004', 'device-005'
];

const sampleCarbonReading = {
  device_id: 'load-test-device',
  timestamp: new Date().toISOString(),
  carbon_emission: Math.random() * 5,
  energy_consumption: Math.random() * 20,
  emission_factor: 0.25,
  data_source: 'load_test',
  location: 'Load Test Building'
};

// Authentication token (if required)
const authToken = __ENV.AUTH_TOKEN || '';
const headers = authToken ? { 
  'Authorization': `Bearer ${authToken}`,
  'Content-Type': 'application/json'
} : { 'Content-Type': 'application/json' };

export default function () {
  // Test scenario weights
  const scenario = Math.random();
  
  if (scenario < 0.4) {
    // 40% - Read carbon readings
    testGetCarbonReadings();
  } else if (scenario < 0.7) {
    // 30% - Post new carbon readings
    testPostCarbonReading();
  } else if (scenario < 0.85) {
    // 15% - Get device information
    testGetDevices();
  } else if (scenario < 0.95) {
    // 10% - Get analytics data
    testGetAnalytics();
  } else {
    // 5% - Get predictions
    testGetPredictions();
  }
  
  sleep(1); // Wait 1 second between requests
}

function testGetCarbonReadings() {
  const deviceId = testDevices[Math.floor(Math.random() * testDevices.length)];
  const startDate = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(); // 24 hours ago
  const endDate = new Date().toISOString();
  
  const url = `${BASE_URL}/api/carbon/readings?device_id=${deviceId}&start_date=${startDate}&end_date=${endDate}`;
  
  const response = http.get(url, { headers });
  
  const success = check(response, {
    'GET /api/carbon/readings status is 200': (r) => r.status === 200,
    'GET /api/carbon/readings response time < 500ms': (r) => r.timings.duration < 500,
    'GET /api/carbon/readings has data': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.data);
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

function testPostCarbonReading() {
  // Randomize the test data
  const reading = {
    ...sampleCarbonReading,
    device_id: testDevices[Math.floor(Math.random() * testDevices.length)],
    timestamp: new Date().toISOString(),
    carbon_emission: Math.random() * 5,
    energy_consumption: Math.random() * 20
  };
  
  const response = http.post(
    `${BASE_URL}/api/carbon/readings`,
    JSON.stringify(reading),
    { headers }
  );
  
  const success = check(response, {
    'POST /api/carbon/readings status is 201': (r) => r.status === 201,
    'POST /api/carbon/readings response time < 1000ms': (r) => r.timings.duration < 1000,
    'POST /api/carbon/readings returns created reading': (r) => {
      try {
        const data = JSON.parse(r.body);
        return data.device_id === reading.device_id;
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

function testGetDevices() {
  const response = http.get(`${BASE_URL}/api/devices`, { headers });
  
  const success = check(response, {
    'GET /api/devices status is 200': (r) => r.status === 200,
    'GET /api/devices response time < 300ms': (r) => r.timings.duration < 300,
    'GET /api/devices returns devices array': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.devices);
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

function testGetAnalytics() {
  const endpoints = [
    '/api/analytics/carbon-summary',
    '/api/analytics/emissions-by-device-type',
    '/api/analytics/emissions-trend?period=7d',
    '/api/analytics/efficiency-metrics'
  ];
  
  const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
  const response = http.get(`${BASE_URL}${endpoint}`, { headers });
  
  const success = check(response, {
    [`GET ${endpoint} status is 200`]: (r) => r.status === 200,
    [`GET ${endpoint} response time < 1000ms`]: (r) => r.timings.duration < 1000,
    [`GET ${endpoint} returns valid JSON`]: (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

function testGetPredictions() {
  const deviceId = testDevices[Math.floor(Math.random() * testDevices.length)];
  const response = http.get(`${BASE_URL}/api/predictions?device_id=${deviceId}`, { headers });
  
  const success = check(response, {
    'GET /api/predictions status is 200': (r) => r.status === 200,
    'GET /api/predictions response time < 2000ms': (r) => r.timings.duration < 2000,
    'GET /api/predictions returns predictions': (r) => {
      try {
        const data = JSON.parse(r.body);
        return Array.isArray(data.predictions);
      } catch (e) {
        return false;
      }
    },
  });
  
  errorRate.add(!success);
}

// Stress test scenario
export function stressTest() {
  const response = http.get(`${BASE_URL}/health`, { headers });
  check(response, {
    'Health check status is 200': (r) => r.status === 200,
  });
}

// Spike test scenario
export function spikeTest() {
  // Simulate sudden spike in traffic
  for (let i = 0; i < 10; i++) {
    testPostCarbonReading();
    sleep(0.1); // Very short sleep to create spike
  }
}

// Volume test scenario
export function volumeTest() {
  // Test with large payloads
  const largeReading = {
    ...sampleCarbonReading,
    metadata: {
      // Add large metadata to test payload handling
      description: 'A'.repeat(1000),
      tags: Array.from({ length: 100 }, (_, i) => `tag-${i}`),
      measurements: Array.from({ length: 50 }, (_, i) => ({
        sensor_id: `sensor-${i}`,
        value: Math.random() * 100,
        unit: 'kWh'
      }))
    }
  };
  
  const response = http.post(
    `${BASE_URL}/api/carbon/readings`,
    JSON.stringify(largeReading),
    { headers }
  );
  
  check(response, {
    'Large payload POST status is 201': (r) => r.status === 201,
    'Large payload POST response time < 3000ms': (r) => r.timings.duration < 3000,
  });
}

// Setup function (runs once at the beginning)
export function setup() {
  console.log('Starting load test...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test devices: ${testDevices.join(', ')}`);
  
  // Verify API is accessible
  const response = http.get(`${BASE_URL}/health`);
  if (response.status !== 200) {
    throw new Error(`API health check failed: ${response.status}`);
  }
  
  return { startTime: new Date() };
}

// Teardown function (runs once at the end)
export function teardown(data) {
  const endTime = new Date();
  const duration = (endTime - data.startTime) / 1000;
  console.log(`Load test completed in ${duration} seconds`);
}