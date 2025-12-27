/**
 * RE-BMS Dashboard Screen v6.1
 * Command Center with SMP Forecast and Key Metrics
 * Connected to real SMP prediction API
 * Matches web-v6.1.0 features
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Dimensions,
  Platform,
  ActivityIndicator,
} from 'react-native';

// Conditionally import charts for native only
let LineChart: any = null;
if (Platform.OS !== 'web') {
  try {
    LineChart = require('react-native-chart-kit').LineChart;
  } catch (e) {
    console.log('LineChart not available');
  }
}

// Icon map for cross-platform compatibility
const iconMap: { [key: string]: string } = {
  'flash': '⚡',
  'trending-up': '📈',
  'leaf': '🌿',
  'pulse': '📊',
  'thermometer': '🌡️',
  'cloud': '☁️',
  'sunny': '☀️',
  'rainy': '🌧️',
  'analytics': '📊',
  'document-text': '📄',
  'settings': '⚙️',
  'notifications': '🔔',
  'calendar': '📅',
  'bar-chart': '📊',
  'pie-chart': '🥧',
  'cash': '💵',
};

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';
import { apiService, SMPForecast, DashboardKPIs, MarketStatus, ModelInfo } from '../services/api';

const { width: screenWidth } = Dimensions.get('window');

// Helper to format SMP data for chart
function formatSMPDataForChart(forecast: SMPForecast | null) {
  if (!forecast || !forecast.q50) {
    return {
      labels: ['00', '04', '08', '12', '16', '20', '24'],
      datasets: [{ data: [95, 88, 102, 145, 168, 152, 112], color: () => colors.chart.smp, strokeWidth: 2 }],
    };
  }

  // Sample every 4 hours for labels
  const labels = ['00', '04', '08', '12', '16', '20', '24'];
  const sampledData = [0, 4, 8, 12, 16, 20, 23].map(i => forecast.q50[i] || 100);

  return {
    labels,
    datasets: [{ data: sampledData, color: () => colors.chart.smp, strokeWidth: 2 }],
  };
}

interface MetricCardProps {
  title: string;
  value: string;
  unit: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  icon: string;
  color: string;
}

function MetricCard({ title, value, unit, trend, trendValue, icon, color }: MetricCardProps) {
  const trendColor = trend === 'up' ? colors.status.success :
                     trend === 'down' ? colors.status.danger : colors.text.muted;
  const trendSymbol = trend === 'up' ? '↑' : trend === 'down' ? '↓' : '–';

  return (
    <View style={styles.metricCard}>
      <View style={styles.metricHeader}>
        <Text style={{ fontSize: 18 }}>{iconMap[icon] || '•'}</Text>
        <Text style={styles.metricTitle}>{title}</Text>
      </View>
      <View style={styles.metricBody}>
        <Text style={[styles.metricValue, { color }]}>{value}</Text>
        <Text style={styles.metricUnit}>{unit}</Text>
      </View>
      {trend && trendValue && (
        <View style={styles.metricTrend}>
          <Text style={[styles.trendText, { color: trendColor }]}>{trendSymbol} {trendValue}</Text>
        </View>
      )}
    </View>
  );
}

interface MarketStatusProps {
  market: 'DAM' | 'RTM';
  status: 'open' | 'closed' | 'pending';
  deadline?: string;
}

function MarketStatusBadge({ market, status, deadline }: MarketStatusProps) {
  const statusColors = {
    open: colors.status.success,
    closed: colors.status.danger,
    pending: colors.status.warning,
  };

  return (
    <View style={styles.marketBadge}>
      <View style={[styles.statusDot, { backgroundColor: statusColors[status] }]} />
      <View>
        <Text style={styles.marketName}>{market}</Text>
        <Text style={styles.marketDeadline}>{deadline || status.toUpperCase()}</Text>
      </View>
    </View>
  );
}

export default function DashboardScreen() {
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // API data states
  const [kpis, setKpis] = useState<DashboardKPIs | null>(null);
  const [smpForecast, setSmpForecast] = useState<SMPForecast | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Fetch data from API
  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [kpisData, forecastData, statusData, modelData] = await Promise.all([
        apiService.getDashboardKPIs(),
        apiService.getSMPForecast(),
        apiService.getMarketStatus(),
        apiService.getModelInfo(),
      ]);

      setKpis(kpisData);
      setSmpForecast(forecastData);
      setMarketStatus(statusData);
      setModelInfo(modelData);
    } catch (err: any) {
      console.error('API Error:', err);
      setError('Failed to load data. Using fallback values.');
      // Set fallback values
      setKpis({
        total_capacity_mw: 225,
        current_output_mw: 168.5,
        utilization_pct: 74.9,
        daily_revenue_million: 38.5,
        revenue_change_pct: 8.3,
        current_smp: 95.0,
        smp_change_pct: 2.1,
        resource_count: 4,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Refresh every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchData();
    setRefreshing(false);
  }, [fetchData]);

  // Format chart data from API response
  const chartData = formatSMPDataForChart(smpForecast);

  // Calculate min/max/avg from forecast
  const smpStats = smpForecast?.q50
    ? {
        min: Math.min(...smpForecast.q50).toFixed(1),
        max: Math.max(...smpForecast.q50).toFixed(1),
        avg: (smpForecast.q50.reduce((a, b) => a + b, 0) / smpForecast.q50.length).toFixed(1),
      }
    : { min: '88.2', max: '168.5', avg: '135.1' };

  if (loading) {
    return (
      <View style={[styles.container, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color={colors.brand.primary} />
        <Text style={{ color: colors.text.muted, marginTop: 16 }}>Loading dashboard...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor={colors.brand.primary}
        />
      }
    >
      {/* API Status Banner */}
      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {/* Model Info Banner */}
      {modelInfo && modelInfo.status === 'ready' && (
        <View style={styles.modelBanner}>
          <Text style={styles.modelText}>🤖 Model: {modelInfo.version} | MAPE: {modelInfo.mape}%</Text>
        </View>
      )}

      {/* Market Status Header */}
      <View style={styles.marketStatusContainer}>
        <MarketStatusBadge
          market="DAM"
          status={marketStatus?.dam?.status === 'open' ? 'open' : 'closed'}
          deadline={marketStatus?.dam?.deadline || 'D-1 10:00'}
        />
        <MarketStatusBadge
          market="RTM"
          status={marketStatus?.rtm?.status === 'open' ? 'open' : 'pending'}
          deadline={marketStatus?.rtm?.next_interval || '15min'}
        />
      </View>

      {/* Key Metrics Grid */}
      <View style={styles.metricsGrid}>
        <MetricCard
          title="Current SMP"
          value={(kpis?.current_smp || 95).toFixed(1)}
          unit="₩/kWh"
          trend={kpis && kpis.smp_change_pct > 0 ? 'up' : 'down'}
          trendValue={`${kpis?.smp_change_pct?.toFixed(1) || '0'}%`}
          icon="flash"
          color={colors.chart.smp}
        />
        <MetricCard
          title="Resources"
          value={(kpis?.resource_count || 4).toString()}
          unit="active"
          icon="layers"
          color={colors.status.warning}
        />
        <MetricCard
          title="Output"
          value={(kpis?.current_output_mw || 168.5).toFixed(1)}
          unit="MW"
          trend="stable"
          trendValue={`${kpis?.utilization_pct?.toFixed(0) || 75}%`}
          icon="speedometer"
          color={colors.chart.generation}
        />
        <MetricCard
          title="Daily Revenue"
          value={(kpis?.daily_revenue_million || 38.5).toFixed(1)}
          unit="M₩"
          trend={kpis && kpis.revenue_change_pct > 0 ? 'up' : 'down'}
          trendValue={`+${kpis?.revenue_change_pct?.toFixed(1) || '8.3'}%`}
          icon="wallet"
          color={colors.brand.accent}
        />
      </View>

      {/* SMP Forecast Chart */}
      <View style={styles.chartContainer}>
        <View style={styles.chartHeader}>
          <Text style={styles.chartTitle}>24H SMP Forecast</Text>
          <View style={styles.chartLegend}>
            <View style={[styles.legendDot, { backgroundColor: colors.chart.smp }]} />
            <Text style={styles.legendText}>{smpForecast?.model_used || 'v3.1'} Model</Text>
          </View>
        </View>
        {Platform.OS === 'web' ? (
          <View style={[styles.chart, { height: 220, backgroundColor: colors.background.tertiary, borderRadius: borderRadius.lg, padding: spacing.md }]}>
            <View style={{ flex: 1, flexDirection: 'row', alignItems: 'flex-end', justifyContent: 'space-around', paddingHorizontal: 10 }}>
              {chartData.datasets[0].data.map((val, i) => (
                <View key={i} style={{ alignItems: 'center', flex: 1 }}>
                  <Text style={{ color: colors.chart.smp, fontSize: 10, marginBottom: 4, fontWeight: '600' }}>{val.toFixed(0)}</Text>
                  <View style={{ height: Math.min(val * 1.2, 150), width: '70%', maxWidth: 40, backgroundColor: colors.chart.smp, borderRadius: 4 }} />
                  <Text style={{ color: colors.text.muted, fontSize: 10, marginTop: 6 }}>{chartData.labels[i]}h</Text>
                </View>
              ))}
            </View>
            <Text style={{ color: colors.text.muted, fontSize: 10, textAlign: 'center', marginTop: 8 }}>₩/kWh by Hour (Real Model)</Text>
          </View>
        ) : LineChart && (
          <LineChart
            data={chartData}
            width={screenWidth - spacing.lg * 2}
            height={200}
            chartConfig={{
              backgroundColor: colors.background.card,
              backgroundGradientFrom: colors.background.card,
              backgroundGradientTo: colors.background.secondary,
              decimalPlaces: 0,
              color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
              labelColor: () => colors.text.muted,
              style: {
                borderRadius: borderRadius.lg,
              },
              propsForDots: {
                r: '4',
                strokeWidth: '2',
                stroke: colors.brand.primary,
              },
              propsForBackgroundLines: {
                stroke: colors.border.primary,
                strokeDasharray: '5,5',
              },
            }}
            bezier
            style={styles.chart}
            withInnerLines={true}
            withOuterLines={false}
            withVerticalLabels={true}
            withHorizontalLabels={true}
            fromZero={false}
          />
        )}
        <View style={styles.forecastStats}>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Min</Text>
            <Text style={[styles.statValue, { color: colors.chart.generation }]}>{smpStats.min}</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Avg</Text>
            <Text style={[styles.statValue, { color: colors.chart.smp }]}>{smpStats.avg}</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Max</Text>
            <Text style={[styles.statValue, { color: colors.status.danger }]}>{smpStats.max}</Text>
          </View>
        </View>
      </View>

      {/* Quick Actions */}
      <View style={styles.actionsContainer}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.actionsGrid}>
          <QuickActionButton
            icon="add-circle"
            label="New Bid"
            color={colors.brand.primary}
          />
          <QuickActionButton
            icon="analytics"
            label="Optimize"
            color={colors.brand.accent}
          />
          <QuickActionButton
            icon="send"
            label="Submit All"
            color={colors.status.info}
          />
          <QuickActionButton
            icon="refresh"
            label="Sync KPX"
            color={colors.status.warning}
          />
        </View>
      </View>

      {/* Recent Activity */}
      <View style={styles.activityContainer}>
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        <ActivityItem
          action="Bid Submitted"
          resource="Jeju Solar #1"
          time="2 min ago"
          status="success"
        />
        <ActivityItem
          action="Optimization Complete"
          resource="Portfolio DAM"
          time="15 min ago"
          status="info"
        />
        <ActivityItem
          action="Imbalance Warning"
          resource="Jeju Wind #3"
          time="1 hr ago"
          status="warning"
        />
      </View>
    </ScrollView>
  );
}

interface QuickActionButtonProps {
  icon: string;
  label: string;
  color: string;
}

function QuickActionButton({ icon, label, color }: QuickActionButtonProps) {
  return (
    <View style={styles.actionButton}>
      <View style={[styles.actionIconContainer, { backgroundColor: `${color}20` }]}>
        <Text style={{ fontSize: 20 }}>{iconMap[icon] || '•'}</Text>
      </View>
      <Text style={styles.actionLabel}>{label}</Text>
    </View>
  );
}

interface ActivityItemProps {
  action: string;
  resource: string;
  time: string;
  status: 'success' | 'warning' | 'info' | 'danger';
}

function ActivityItem({ action, resource, time, status }: ActivityItemProps) {
  const statusColors = {
    success: colors.status.success,
    warning: colors.status.warning,
    info: colors.status.info,
    danger: colors.status.danger,
  };

  return (
    <View style={styles.activityItem}>
      <View style={[styles.activityDot, { backgroundColor: statusColors[status] }]} />
      <View style={styles.activityContent}>
        <Text style={styles.activityAction}>{action}</Text>
        <Text style={styles.activityResource}>{resource}</Text>
      </View>
      <Text style={styles.activityTime}>{time}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  contentContainer: {
    padding: spacing.md,
    paddingBottom: spacing.xxl,
  },

  // Market Status
  marketStatusContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: spacing.lg,
  },
  marketBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background.secondary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.lg,
    gap: spacing.sm,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  marketName: {
    color: colors.text.primary,
    fontSize: fontSize.md,
    fontWeight: 'bold',
  },
  marketDeadline: {
    color: colors.text.muted,
    fontSize: fontSize.xs,
  },

  // Metrics Grid
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  metricCard: {
    // Use percentage for web compatibility
    width: Platform.OS === 'web' ? '48%' : (screenWidth - spacing.md * 2 - spacing.sm) / 2 - spacing.sm / 2,
    minWidth: 150,
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  metricHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginBottom: spacing.sm,
  },
  iconPlaceholder: {
    width: 20,
    height: 20,
    borderRadius: 4,
  },
  metricTitle: {
    color: colors.text.secondary,
    fontSize: fontSize.sm,
  },
  metricBody: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: spacing.xs,
  },
  metricValue: {
    fontSize: fontSize.xxl,
    fontWeight: 'bold',
  },
  metricUnit: {
    color: colors.text.muted,
    fontSize: fontSize.sm,
  },
  metricTrend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    marginTop: spacing.xs,
  },
  trendText: {
    fontSize: fontSize.xs,
  },

  // Chart
  chartContainer: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  chartHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  chartTitle: {
    color: colors.text.primary,
    fontSize: fontSize.lg,
    fontWeight: 'bold',
  },
  chartLegend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    color: colors.text.muted,
    fontSize: fontSize.xs,
  },
  chart: {
    borderRadius: borderRadius.md,
    marginVertical: spacing.sm,
  },
  forecastStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    color: colors.text.muted,
    fontSize: fontSize.xs,
    marginBottom: 2,
  },
  statValue: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
  },

  // Actions
  actionsContainer: {
    marginBottom: spacing.lg,
  },
  sectionTitle: {
    color: colors.text.primary,
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    marginBottom: spacing.md,
  },
  actionsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionButton: {
    alignItems: 'center',
    width: (screenWidth - spacing.md * 2) / 4 - spacing.sm,
  },
  actionIconContainer: {
    width: 48,
    height: 48,
    borderRadius: borderRadius.lg,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  actionLabel: {
    color: colors.text.secondary,
    fontSize: fontSize.xs,
    textAlign: 'center',
  },

  // Activity
  activityContainer: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  activityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  activityDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: spacing.md,
  },
  activityContent: {
    flex: 1,
  },
  activityAction: {
    color: colors.text.primary,
    fontSize: fontSize.md,
    fontWeight: '500',
  },
  activityResource: {
    color: colors.text.muted,
    fontSize: fontSize.sm,
  },
  activityTime: {
    color: colors.text.muted,
    fontSize: fontSize.xs,
  },

  // API Status Banners
  errorBanner: {
    backgroundColor: `${colors.status.danger}20`,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.status.danger,
  },
  errorText: {
    color: colors.status.danger,
    fontSize: fontSize.sm,
    textAlign: 'center',
  },
  modelBanner: {
    backgroundColor: `${colors.brand.primary}20`,
    borderRadius: borderRadius.md,
    padding: spacing.sm,
    marginBottom: spacing.md,
  },
  modelText: {
    color: colors.brand.primary,
    fontSize: fontSize.xs,
    textAlign: 'center',
    fontWeight: '500',
  },
});
