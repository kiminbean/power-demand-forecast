/**
 * RE-BMS Settlement Screen
 * Settlement & Revenue Analytics
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Platform,
} from 'react-native';

// Conditional imports for native-only features
let Ionicons: any = null;
let LineChart: any = null;
let BarChart: any = null;

if (Platform.OS !== 'web') {
  try {
    Ionicons = require('@expo/vector-icons').Ionicons;
    const chartKit = require('react-native-chart-kit');
    LineChart = chartKit.LineChart;
    BarChart = chartKit.BarChart;
  } catch (e) {
    console.log('Native components not available');
  }
}

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';

// Icon component with emoji fallback for web
function Icon({ name, size, color }: { name: string; size: number; color: string }) {
  if (Ionicons) {
    return <Ionicons name={name as any} size={size} color={color} />;
  }
  const iconMap: { [key: string]: string } = {
    'cash': '💵',
    'warning': '⚠️',
    'wallet': '💰',
    'analytics': '📊',
    'arrow-up': '↑',
    'arrow-down': '↓',
  };
  return (
    <Text style={{ fontSize: size * 0.8 }}>{iconMap[name] || '•'}</Text>
  );
}

const { width: screenWidth } = Dimensions.get('window');

// Types
interface DailySettlement {
  date: string;
  revenue: number;
  penalty: number;
  netRevenue: number;
  avgSmp: number;
  generationMwh: number;
  imbalanceRate: number;
}

// Mock data
const mockSettlements: DailySettlement[] = [
  { date: '12/13', revenue: 15.2, penalty: 0.8, netRevenue: 14.4, avgSmp: 132.5, generationMwh: 115.3, imbalanceRate: 5.2 },
  { date: '12/14', revenue: 18.5, penalty: 0.3, netRevenue: 18.2, avgSmp: 145.2, generationMwh: 127.5, imbalanceRate: 2.1 },
  { date: '12/15', revenue: 12.8, penalty: 1.5, netRevenue: 11.3, avgSmp: 118.7, generationMwh: 108.2, imbalanceRate: 9.8 },
  { date: '12/16', revenue: 20.1, penalty: 0.2, netRevenue: 19.9, avgSmp: 155.3, generationMwh: 129.4, imbalanceRate: 1.2 },
  { date: '12/17', revenue: 16.7, penalty: 0.6, netRevenue: 16.1, avgSmp: 138.9, generationMwh: 120.2, imbalanceRate: 4.5 },
  { date: '12/18', revenue: 19.3, penalty: 0.4, netRevenue: 18.9, avgSmp: 148.6, generationMwh: 130.1, imbalanceRate: 2.8 },
  { date: '12/19', revenue: 17.5, penalty: 0.5, netRevenue: 17.0, avgSmp: 141.2, generationMwh: 124.0, imbalanceRate: 3.5 },
];

// Period Selector
type Period = '7d' | '30d' | '90d';

function PeriodSelector({
  selectedPeriod,
  onPeriodChange,
}: {
  selectedPeriod: Period;
  onPeriodChange: (period: Period) => void;
}) {
  const periods: { key: Period; label: string }[] = [
    { key: '7d', label: '7 Days' },
    { key: '30d', label: '30 Days' },
    { key: '90d', label: '90 Days' },
  ];

  return (
    <View style={styles.periodSelector}>
      {periods.map((period) => (
        <TouchableOpacity
          key={period.key}
          style={[
            styles.periodButton,
            selectedPeriod === period.key && styles.periodButtonActive,
          ]}
          onPress={() => onPeriodChange(period.key)}
        >
          <Text
            style={[
              styles.periodButtonText,
              selectedPeriod === period.key && styles.periodButtonTextActive,
            ]}
          >
            {period.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

// Summary Cards
function SummaryCard({
  title,
  value,
  unit,
  icon,
  color,
  trend,
}: {
  title: string;
  value: string;
  unit: string;
  icon: string;
  color: string;
  trend?: { value: string; direction: 'up' | 'down' };
}) {
  return (
    <View style={styles.summaryCard}>
      <View style={[styles.summaryIcon, { backgroundColor: `${color}20` }]}>
        <Icon name={icon} size={20} color={color} />
      </View>
      <Text style={styles.summaryTitle}>{title}</Text>
      <View style={styles.summaryValueRow}>
        <Text style={[styles.summaryValue, { color }]}>{value}</Text>
        <Text style={styles.summaryUnit}>{unit}</Text>
      </View>
      {trend && (
        <View style={styles.trendRow}>
          <Icon
            name={trend.direction === 'up' ? 'arrow-up' : 'arrow-down'}
            size={12}
            color={trend.direction === 'up' ? colors.status.success : colors.status.danger}
          />
          <Text
            style={[
              styles.trendText,
              { color: trend.direction === 'up' ? colors.status.success : colors.status.danger },
            ]}
          >
            {trend.value}
          </Text>
        </View>
      )}
    </View>
  );
}

// Imbalance Indicator
function ImbalanceIndicator({ rate }: { rate: number }) {
  const tolerance = 12; // ±12% tolerance band (Jeju pilot)
  const isWithinBand = Math.abs(rate) <= tolerance;
  const indicatorColor = isWithinBand ? colors.status.success : colors.status.danger;

  return (
    <View style={styles.imbalanceContainer}>
      <View style={styles.imbalanceHeader}>
        <Text style={styles.imbalanceTitle}>Imbalance Status</Text>
        <View style={[styles.toleranceBadge, { backgroundColor: `${indicatorColor}20` }]}>
          <Text style={[styles.toleranceText, { color: indicatorColor }]}>
            {isWithinBand ? 'Within Band' : 'Penalty Applied'}
          </Text>
        </View>
      </View>

      <View style={styles.toleranceBar}>
        {/* Tolerance band visualization */}
        <View style={styles.toleranceScale}>
          <Text style={styles.scaleLabel}>-12%</Text>
          <Text style={styles.scaleLabel}>0%</Text>
          <Text style={styles.scaleLabel}>+12%</Text>
        </View>
        <View style={styles.toleranceTrack}>
          <View style={styles.toleranceBand} />
          <View
            style={[
              styles.imbalanceMarker,
              {
                left: `${50 + (rate / 24) * 100}%`,
                backgroundColor: indicatorColor,
              },
            ]}
          />
        </View>
      </View>

      <View style={styles.penaltyInfo}>
        <View style={styles.penaltyRow}>
          <Text style={styles.penaltyLabel}>Over-generation penalty:</Text>
          <Text style={styles.penaltyValue}>80% of SMP</Text>
        </View>
        <View style={styles.penaltyRow}>
          <Text style={styles.penaltyLabel}>Under-generation penalty:</Text>
          <Text style={styles.penaltyValue}>120% of SMP</Text>
        </View>
      </View>
    </View>
  );
}

// Settlement Table Row
function SettlementRow({ settlement }: { settlement: DailySettlement }) {
  const penaltyColor = settlement.penalty > 1 ? colors.status.danger : colors.status.warning;

  return (
    <View style={styles.settlementRow}>
      <View style={styles.settlementCell}>
        <Text style={styles.dateText}>{settlement.date}</Text>
      </View>
      <View style={styles.settlementCell}>
        <Text style={[styles.revenueText, { color: colors.status.success }]}>
          ₩{settlement.revenue.toFixed(1)}M
        </Text>
      </View>
      <View style={styles.settlementCell}>
        <Text style={[styles.penaltyText, { color: penaltyColor }]}>
          -₩{settlement.penalty.toFixed(1)}M
        </Text>
      </View>
      <View style={styles.settlementCell}>
        <Text style={styles.netRevenueText}>₩{settlement.netRevenue.toFixed(1)}M</Text>
      </View>
    </View>
  );
}

export default function SettlementScreen() {
  const [selectedPeriod, setSelectedPeriod] = useState<Period>('7d');

  const totalRevenue = mockSettlements.reduce((sum, s) => sum + s.revenue, 0);
  const totalPenalty = mockSettlements.reduce((sum, s) => sum + s.penalty, 0);
  const totalNetRevenue = mockSettlements.reduce((sum, s) => sum + s.netRevenue, 0);
  const avgImbalance = mockSettlements.reduce((sum, s) => sum + s.imbalanceRate, 0) / mockSettlements.length;

  const revenueChartData = {
    labels: mockSettlements.map((s) => s.date),
    datasets: [
      {
        data: mockSettlements.map((s) => s.netRevenue),
        color: () => colors.brand.accent,
        strokeWidth: 2,
      },
    ],
  };

  const barChartData = {
    labels: mockSettlements.map((s) => s.date),
    datasets: [
      {
        data: mockSettlements.map((s) => s.generationMwh),
      },
    ],
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Period Selector */}
      <PeriodSelector selectedPeriod={selectedPeriod} onPeriodChange={setSelectedPeriod} />

      {/* Summary Cards */}
      <View style={styles.summaryGrid}>
        <SummaryCard
          title="Total Revenue"
          value={totalRevenue.toFixed(1)}
          unit="M₩"
          icon="cash"
          color={colors.status.success}
          trend={{ value: '+12.3%', direction: 'up' }}
        />
        <SummaryCard
          title="Penalties"
          value={totalPenalty.toFixed(1)}
          unit="M₩"
          icon="warning"
          color={colors.status.danger}
          trend={{ value: '-8.5%', direction: 'down' }}
        />
        <SummaryCard
          title="Net Revenue"
          value={totalNetRevenue.toFixed(1)}
          unit="M₩"
          icon="wallet"
          color={colors.brand.accent}
          trend={{ value: '+15.2%', direction: 'up' }}
        />
        <SummaryCard
          title="Avg Imbalance"
          value={avgImbalance.toFixed(1)}
          unit="%"
          icon="analytics"
          color={colors.brand.primary}
        />
      </View>

      {/* Imbalance Indicator */}
      <ImbalanceIndicator rate={avgImbalance} />

      {/* Revenue Chart */}
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>Net Revenue Trend</Text>
        {Platform.OS === 'web' ? (
          <View style={{ flexDirection: 'row', height: 180, alignItems: 'flex-end', justifyContent: 'space-around', paddingHorizontal: 8 }}>
            {revenueChartData.datasets[0].data.map((val, i) => (
              <View key={i} style={{ alignItems: 'center' }}>
                <View style={{ height: val * 8, width: 28, backgroundColor: colors.brand.accent, borderRadius: 4 }} />
                <Text style={{ color: colors.text.muted, fontSize: 10, marginTop: 4 }}>{revenueChartData.labels[i]}</Text>
              </View>
            ))}
          </View>
        ) : LineChart && (
          <LineChart
            data={revenueChartData}
            width={screenWidth - spacing.lg * 2 - spacing.md * 2}
            height={180}
            chartConfig={{
              backgroundColor: colors.background.card,
              backgroundGradientFrom: colors.background.card,
              backgroundGradientTo: colors.background.secondary,
              decimalPlaces: 1,
              color: (opacity = 1) => `rgba(16, 185, 129, ${opacity})`,
              labelColor: () => colors.text.muted,
              propsForDots: {
                r: '4',
                strokeWidth: '2',
                stroke: colors.brand.accent,
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
          />
        )}
      </View>

      {/* Generation Chart */}
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>Daily Generation (MWh)</Text>
        {Platform.OS === 'web' ? (
          <View style={{ flexDirection: 'row', height: 180, alignItems: 'flex-end', justifyContent: 'space-around', paddingHorizontal: 8 }}>
            {barChartData.datasets[0].data.map((val, i) => (
              <View key={i} style={{ alignItems: 'center' }}>
                <Text style={{ color: colors.text.muted, fontSize: 10, marginBottom: 4 }}>{val.toFixed(0)}</Text>
                <View style={{ height: val * 1.2, width: 28, backgroundColor: colors.brand.primary, borderRadius: 4 }} />
                <Text style={{ color: colors.text.muted, fontSize: 10, marginTop: 4 }}>{barChartData.labels[i]}</Text>
              </View>
            ))}
          </View>
        ) : BarChart && (
          <BarChart
            data={barChartData}
            width={screenWidth - spacing.lg * 2 - spacing.md * 2}
            height={180}
            yAxisLabel=""
            yAxisSuffix=""
            chartConfig={{
              backgroundColor: colors.background.card,
              backgroundGradientFrom: colors.background.card,
              backgroundGradientTo: colors.background.secondary,
              decimalPlaces: 0,
              color: (opacity = 1) => `rgba(99, 102, 241, ${opacity})`,
              labelColor: () => colors.text.muted,
              barPercentage: 0.6,
              propsForBackgroundLines: {
                stroke: colors.border.primary,
                strokeDasharray: '5,5',
              },
            }}
            style={styles.chart}
            showValuesOnTopOfBars={true}
            fromZero={true}
          />
        )}
      </View>

      {/* Settlement Table */}
      <View style={styles.tableCard}>
        <Text style={styles.chartTitle}>Settlement Details</Text>
        <View style={styles.tableHeader}>
          <Text style={styles.tableHeaderText}>Date</Text>
          <Text style={styles.tableHeaderText}>Revenue</Text>
          <Text style={styles.tableHeaderText}>Penalty</Text>
          <Text style={styles.tableHeaderText}>Net</Text>
        </View>
        {mockSettlements.map((settlement) => (
          <SettlementRow key={settlement.date} settlement={settlement} />
        ))}
      </View>
    </ScrollView>
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

  // Period Selector
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: colors.background.secondary,
    borderRadius: borderRadius.lg,
    padding: 4,
    marginBottom: spacing.lg,
  },
  periodButton: {
    flex: 1,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderRadius: borderRadius.md,
  },
  periodButtonActive: {
    backgroundColor: colors.brand.primary,
  },
  periodButtonText: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  periodButtonTextActive: {
    color: colors.text.primary,
    fontWeight: 'bold',
  },

  // Summary Grid
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    marginBottom: spacing.lg,
  },
  summaryCard: {
    width: Platform.OS === 'web' ? '48%' : (screenWidth - spacing.md * 2 - spacing.sm) / 2 - spacing.sm / 2,
    minWidth: 150,
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  summaryIcon: {
    width: 36,
    height: 36,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  summaryTitle: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginBottom: 4,
  },
  summaryValueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 2,
  },
  summaryValue: {
    fontSize: fontSize.xxl,
    fontWeight: 'bold',
  },
  summaryUnit: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  trendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    marginTop: 4,
  },
  trendText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },

  // Imbalance Indicator
  imbalanceContainer: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  imbalanceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  imbalanceTitle: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  toleranceBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: borderRadius.sm,
  },
  toleranceText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },
  toleranceBar: {
    marginBottom: spacing.md,
  },
  toleranceScale: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  scaleLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
  },
  toleranceTrack: {
    height: 12,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.sm,
    position: 'relative',
    overflow: 'visible',
  },
  toleranceBand: {
    position: 'absolute',
    left: '25%',
    right: '25%',
    top: 0,
    bottom: 0,
    backgroundColor: `${colors.status.success}30`,
    borderRadius: borderRadius.sm,
  },
  imbalanceMarker: {
    position: 'absolute',
    width: 16,
    height: 16,
    borderRadius: 8,
    top: -2,
    marginLeft: -8,
    borderWidth: 2,
    borderColor: colors.text.primary,
  },
  penaltyInfo: {
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  penaltyRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  penaltyLabel: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  penaltyValue: {
    fontSize: fontSize.sm,
    color: colors.text.secondary,
    fontWeight: '500',
  },

  // Charts
  chartCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  chartTitle: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  chart: {
    borderRadius: borderRadius.md,
  },

  // Settlement Table
  tableCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  tableHeader: {
    flexDirection: 'row',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
    marginBottom: spacing.sm,
  },
  tableHeaderText: {
    flex: 1,
    fontSize: fontSize.xs,
    color: colors.text.muted,
    textAlign: 'center',
  },
  settlementRow: {
    flexDirection: 'row',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  settlementCell: {
    flex: 1,
    alignItems: 'center',
  },
  dateText: {
    fontSize: fontSize.sm,
    color: colors.text.secondary,
  },
  revenueText: {
    fontSize: fontSize.sm,
    fontWeight: '600',
  },
  penaltyText: {
    fontSize: fontSize.sm,
    fontWeight: '600',
  },
  netRevenueText: {
    fontSize: fontSize.sm,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
});
