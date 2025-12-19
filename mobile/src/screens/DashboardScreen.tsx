/**
 * RE-BMS Dashboard Screen
 * Command Center with SMP Forecast and Key Metrics
 */

import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  RefreshControl,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { LineChart } from 'react-native-chart-kit';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';

const { width: screenWidth } = Dimensions.get('window');

// Mock data - will be replaced with API calls
const mockSMPData = {
  labels: ['00', '04', '08', '12', '16', '20', '24'],
  datasets: [
    {
      data: [95, 88, 102, 145, 168, 152, 112],
      color: () => colors.chart.smp,
      strokeWidth: 2,
    },
  ],
};

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
  const trendIcon = trend === 'up' ? 'arrow-up' :
                    trend === 'down' ? 'arrow-down' : 'remove';

  return (
    <View style={styles.metricCard}>
      <View style={styles.metricHeader}>
        <Ionicons name={icon as any} size={20} color={color} />
        <Text style={styles.metricTitle}>{title}</Text>
      </View>
      <View style={styles.metricBody}>
        <Text style={[styles.metricValue, { color }]}>{value}</Text>
        <Text style={styles.metricUnit}>{unit}</Text>
      </View>
      {trend && trendValue && (
        <View style={styles.metricTrend}>
          <Ionicons name={trendIcon as any} size={12} color={trendColor} />
          <Text style={[styles.trendText, { color: trendColor }]}>{trendValue}</Text>
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
  const [currentSMP, setCurrentSMP] = useState(135.07);
  const [pendingBids, setPendingBids] = useState(3);
  const [totalPortfolio, setTotalPortfolio] = useState(45.2);
  const [dailyRevenue, setDailyRevenue] = useState(12.5);

  const onRefresh = React.useCallback(() => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setRefreshing(false);
    }, 1500);
  }, []);

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
      {/* Market Status Header */}
      <View style={styles.marketStatusContainer}>
        <MarketStatusBadge market="DAM" status="open" deadline="D-1 10:00" />
        <MarketStatusBadge market="RTM" status="pending" deadline="15min" />
      </View>

      {/* Key Metrics Grid */}
      <View style={styles.metricsGrid}>
        <MetricCard
          title="Current SMP"
          value={currentSMP.toFixed(1)}
          unit="₩/kWh"
          trend="up"
          trendValue="+5.2%"
          icon="flash"
          color={colors.chart.smp}
        />
        <MetricCard
          title="Pending Bids"
          value={pendingBids.toString()}
          unit="bids"
          icon="document-text"
          color={colors.status.warning}
        />
        <MetricCard
          title="Portfolio"
          value={totalPortfolio.toFixed(1)}
          unit="MW"
          icon="layers"
          color={colors.chart.generation}
        />
        <MetricCard
          title="Daily Revenue"
          value={dailyRevenue.toFixed(1)}
          unit="M₩"
          trend="up"
          trendValue="+12.3%"
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
            <Text style={styles.legendText}>v3.1 Model</Text>
          </View>
        </View>
        <LineChart
          data={mockSMPData}
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
        <View style={styles.forecastStats}>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Min</Text>
            <Text style={[styles.statValue, { color: colors.chart.generation }]}>88.2</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Avg</Text>
            <Text style={[styles.statValue, { color: colors.chart.smp }]}>135.1</Text>
          </View>
          <View style={styles.statItem}>
            <Text style={styles.statLabel}>Max</Text>
            <Text style={[styles.statValue, { color: colors.status.danger }]}>168.5</Text>
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
        <Ionicons name={icon as any} size={24} color={color} />
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
    width: (screenWidth - spacing.md * 2 - spacing.sm) / 2 - spacing.sm / 2,
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
});
