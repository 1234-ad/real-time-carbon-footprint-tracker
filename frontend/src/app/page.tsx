'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  ChartBarIcon, 
  GlobeAltIcon, 
  BoltIcon, 
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon
} from '@heroicons/react/24/outline';
import DashboardLayout from '@/components/layout/DashboardLayout';
import MetricCard from '@/components/ui/MetricCard';
import EmissionsChart from '@/components/charts/EmissionsChart';
import DeviceStatusGrid from '@/components/devices/DeviceStatusGrid';
import AlertsPanel from '@/components/alerts/AlertsPanel';
import { useCarbonData } from '@/hooks/useCarbonData';
import { useRealTimeUpdates } from '@/hooks/useRealTimeUpdates';

export default function Dashboard() {
  const { data: carbonData, isLoading, error } = useCarbonData();
  const { isConnected } = useRealTimeUpdates();

  if (isLoading) {
    return (
      <DashboardLayout>
        <div className=\"flex items-center justify-center h-64\">
          <div className=\"animate-spin rounded-full h-32 w-32 border-b-2 border-green-500\"></div>
        </div>
      </DashboardLayout>
    );
  }

  if (error) {
    return (
      <DashboardLayout>
        <div className=\"bg-red-50 border border-red-200 rounded-md p-4\">
          <div className=\"flex\">
            <ExclamationTriangleIcon className=\"h-5 w-5 text-red-400\" />
            <div className=\"ml-3\">
              <h3 className=\"text-sm font-medium text-red-800\">
                Error loading dashboard data
              </h3>
              <p className=\"mt-2 text-sm text-red-700\">{error.message}</p>
            </div>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const metrics = carbonData?.summary || {
    totalEmissions: 0,
    totalDevices: 0,
    activeAlerts: 0,
    energyEfficiency: 0,
    emissionsTrend: 0
  };

  return (
    <DashboardLayout>
      <div className=\"space-y-6\">
        {/* Header */}
        <div className=\"flex items-center justify-between\">
          <div>
            <h1 className=\"text-3xl font-bold text-gray-900\">
              Carbon Footprint Dashboard
            </h1>
            <p className=\"mt-2 text-gray-600\">
              Real-time monitoring and analytics for carbon emissions
            </p>
          </div>
          <div className=\"flex items-center space-x-2\">
            <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span className=\"text-sm text-gray-600\">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Key Metrics */}
        <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6\">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <MetricCard
              title=\"Total Emissions\"
              value={`${metrics.totalEmissions.toFixed(2)} kg CO₂e`}
              icon={GlobeAltIcon}
              trend={metrics.emissionsTrend}
              trendIcon={metrics.emissionsTrend > 0 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon}
              color=\"blue\"
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <MetricCard
              title=\"Active Devices\"
              value={metrics.totalDevices.toString()}
              icon={BoltIcon}
              color=\"green\"
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <MetricCard
              title=\"Active Alerts\"
              value={metrics.activeAlerts.toString()}
              icon={ExclamationTriangleIcon}
              color={metrics.activeAlerts > 0 ? \"red\" : \"gray\"}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <MetricCard
              title=\"Efficiency Score\"
              value={`${(metrics.energyEfficiency * 100).toFixed(1)}%`}
              icon={ChartBarIcon}
              color=\"purple\"
            />
          </motion.div>
        </div>

        {/* Charts and Visualizations */}
        <div className=\"grid grid-cols-1 lg:grid-cols-2 gap-6\">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
          >
            <h2 className=\"text-lg font-semibold text-gray-900 mb-4\">
              Emissions Trend (24h)
            </h2>
            <EmissionsChart data={carbonData?.timeSeries || []} />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 }}
            className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
          >
            <h2 className=\"text-lg font-semibold text-gray-900 mb-4\">
              Recent Alerts
            </h2>
            <AlertsPanel alerts={carbonData?.alerts || []} />
          </motion.div>
        </div>

        {/* Device Status Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
        >
          <h2 className=\"text-lg font-semibold text-gray-900 mb-4\">
            Device Status Overview
          </h2>
          <DeviceStatusGrid devices={carbonData?.devices || []} />
        </motion.div>

        {/* Additional Analytics */}
        <div className=\"grid grid-cols-1 lg:grid-cols-3 gap-6\">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
          >
            <h3 className=\"text-md font-semibold text-gray-900 mb-3\">
              Top Emitters
            </h3>
            <div className=\"space-y-3\">
              {carbonData?.topEmitters?.slice(0, 5).map((device, index) => (
                <div key={device.id} className=\"flex items-center justify-between\">
                  <span className=\"text-sm text-gray-600\">{device.name}</span>
                  <span className=\"text-sm font-medium text-gray-900\">
                    {device.emissions.toFixed(2)} kg CO₂e
                  </span>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
            className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
          >
            <h3 className=\"text-md font-semibold text-gray-900 mb-3\">
              Emission Sources
            </h3>
            <div className=\"space-y-3\">
              {carbonData?.emissionSources?.map((source, index) => (
                <div key={source.type} className=\"flex items-center justify-between\">
                  <span className=\"text-sm text-gray-600\">{source.type}</span>
                  <span className=\"text-sm font-medium text-gray-900\">
                    {((source.percentage || 0) * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className=\"bg-white rounded-lg shadow-sm border border-gray-200 p-6\"
          >
            <h3 className=\"text-md font-semibold text-gray-900 mb-3\">
              Predictions
            </h3>
            <div className=\"space-y-3\">
              <div className=\"flex items-center justify-between\">
                <span className=\"text-sm text-gray-600\">Next Hour</span>
                <span className=\"text-sm font-medium text-gray-900\">
                  {carbonData?.predictions?.nextHour?.toFixed(2) || '0.00'} kg CO₂e
                </span>
              </div>
              <div className=\"flex items-center justify-between\">
                <span className=\"text-sm text-gray-600\">Today</span>
                <span className=\"text-sm font-medium text-gray-900\">
                  {carbonData?.predictions?.today?.toFixed(2) || '0.00'} kg CO₂e
                </span>
              </div>
              <div className=\"flex items-center justify-between\">
                <span className=\"text-sm text-gray-600\">This Week</span>
                <span className=\"text-sm font-medium text-gray-900\">
                  {carbonData?.predictions?.thisWeek?.toFixed(2) || '0.00'} kg CO₂e
                </span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </DashboardLayout>
  );
}