/**
 * RE-BMS Portfolio Screen
 * Renewable Resource Management
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Dimensions,
  Platform,
} from 'react-native';

// Conditional imports for native-only features
let Ionicons: any = null;
let PieChart: any = null;

if (Platform.OS !== 'web') {
  try {
    Ionicons = require('@expo/vector-icons').Ionicons;
    PieChart = require('react-native-chart-kit').PieChart;
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
    'sunny': '☀️',
    'cloudy': '💨',
    'layers': '📚',
    'checkmark-circle': '✅',
    'construct': '🔧',
    'location-outline': '📍',
    'flash-outline': '⚡',
    'wallet-outline': '💰',
  };
  return (
    <Text style={{ fontSize: size * 0.8 }}>{iconMap[name] || '•'}</Text>
  );
}

const { width: screenWidth } = Dimensions.get('window');

// Types
interface RenewableResource {
  id: string;
  name: string;
  type: 'solar' | 'wind';
  capacityMw: number;
  location: string;
  status: 'active' | 'maintenance' | 'offline';
  currentOutput: number;
  utilizationRate: number;
  todayGeneration: number;
  monthlyRevenue: number;
}

// Mock data
const mockResources: RenewableResource[] = [
  {
    id: 'res-001',
    name: 'Jeju Solar #1',
    type: 'solar',
    capacityMw: 50.0,
    location: 'Jeju-si',
    status: 'active',
    currentOutput: 42.5,
    utilizationRate: 85,
    todayGeneration: 312.5,
    monthlyRevenue: 145.2,
  },
  {
    id: 'res-002',
    name: 'Jeju Wind #1',
    type: 'wind',
    capacityMw: 30.0,
    location: 'Seogwipo-si',
    status: 'active',
    currentOutput: 24.3,
    utilizationRate: 81,
    todayGeneration: 215.8,
    monthlyRevenue: 98.7,
  },
  {
    id: 'res-003',
    name: 'Jeju Solar #2',
    type: 'solar',
    capacityMw: 25.0,
    location: 'Aewol',
    status: 'active',
    currentOutput: 21.2,
    utilizationRate: 85,
    todayGeneration: 156.3,
    monthlyRevenue: 72.4,
  },
  {
    id: 'res-004',
    name: 'Jeju Wind #2',
    type: 'wind',
    capacityMw: 20.0,
    location: 'Hallim',
    status: 'maintenance',
    currentOutput: 0,
    utilizationRate: 0,
    todayGeneration: 45.2,
    monthlyRevenue: 21.3,
  },
];

// Portfolio Summary Component
function PortfolioSummary({ resources }: { resources: RenewableResource[] }) {
  const totalCapacity = resources.reduce((sum, r) => sum + r.capacityMw, 0);
  const totalOutput = resources.reduce((sum, r) => sum + r.currentOutput, 0);
  const avgUtilization = totalCapacity > 0 ? (totalOutput / totalCapacity) * 100 : 0;
  const totalRevenue = resources.reduce((sum, r) => sum + r.monthlyRevenue, 0);

  const solarCapacity = resources
    .filter(r => r.type === 'solar')
    .reduce((sum, r) => sum + r.capacityMw, 0);
  const windCapacity = resources
    .filter(r => r.type === 'wind')
    .reduce((sum, r) => sum + r.capacityMw, 0);

  const pieData = [
    {
      name: 'Solar',
      capacity: solarCapacity,
      color: colors.chart.solar,
      legendFontColor: colors.text.secondary,
      legendFontSize: 12,
    },
    {
      name: 'Wind',
      capacity: windCapacity,
      color: colors.chart.wind,
      legendFontColor: colors.text.secondary,
      legendFontSize: 12,
    },
  ];

  return (
    <View style={styles.summaryContainer}>
      {/* Stats Row */}
      <View style={styles.statsRow}>
        <View style={styles.statBox}>
          <Text style={styles.statValue}>{totalCapacity.toFixed(1)}</Text>
          <Text style={styles.statLabel}>Total MW</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={[styles.statValue, { color: colors.chart.generation }]}>
            {totalOutput.toFixed(1)}
          </Text>
          <Text style={styles.statLabel}>Current MW</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={[styles.statValue, { color: colors.brand.primary }]}>
            {avgUtilization.toFixed(0)}%
          </Text>
          <Text style={styles.statLabel}>Utilization</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={[styles.statValue, { color: colors.brand.accent }]}>
            ₩{totalRevenue.toFixed(1)}M
          </Text>
          <Text style={styles.statLabel}>Revenue</Text>
        </View>
      </View>

      {/* Pie Chart */}
      <View style={styles.chartContainer}>
        <Text style={styles.chartTitle}>Capacity Mix</Text>
        {Platform.OS === 'web' ? (
          <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', height: 160, gap: 24 }}>
            {pieData.map((item, i) => (
              <View key={i} style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                <View style={{ width: 40, height: 40, borderRadius: 20, backgroundColor: item.color, justifyContent: 'center', alignItems: 'center' }}>
                  <Text style={{ color: '#fff', fontSize: 12, fontWeight: 'bold' }}>{item.capacity.toFixed(0)}</Text>
                </View>
                <Text style={{ color: colors.text.secondary }}>{item.name}</Text>
              </View>
            ))}
          </View>
        ) : PieChart && (
          <PieChart
            data={pieData}
            width={screenWidth - spacing.lg * 2 - spacing.md * 2}
            height={160}
            chartConfig={{
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            }}
            accessor="capacity"
            backgroundColor="transparent"
            paddingLeft="15"
            absolute
          />
        )}
      </View>
    </View>
  );
}

// Resource Card Component
function ResourceCard({ resource }: { resource: RenewableResource }) {
  const statusColors = {
    active: colors.status.success,
    maintenance: colors.status.warning,
    offline: colors.status.danger,
  };

  const resourceIcon = resource.type === 'solar' ? 'sunny' : 'cloudy';
  const resourceColor = resource.type === 'solar' ? colors.chart.solar : colors.chart.wind;

  return (
    <TouchableOpacity style={styles.resourceCard} activeOpacity={0.7}>
      {/* Header */}
      <View style={styles.cardHeader}>
        <View style={styles.resourceInfo}>
          <View style={[styles.resourceIconContainer, { backgroundColor: `${resourceColor}20` }]}>
            <Icon name={resourceIcon} size={24} color={resourceColor} />
          </View>
          <View>
            <Text style={styles.resourceName}>{resource.name}</Text>
            <View style={styles.locationRow}>
              <Icon name="location-outline" size={12} color={colors.text.muted} />
              <Text style={styles.resourceLocation}>{resource.location}</Text>
            </View>
          </View>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: `${statusColors[resource.status]}20` }]}>
          <View style={[styles.statusDot, { backgroundColor: statusColors[resource.status] }]} />
          <Text style={[styles.statusText, { color: statusColors[resource.status] }]}>
            {resource.status.charAt(0).toUpperCase() + resource.status.slice(1)}
          </Text>
        </View>
      </View>

      {/* Output Gauge */}
      <View style={styles.outputSection}>
        <View style={styles.outputHeader}>
          <Text style={styles.outputLabel}>Current Output</Text>
          <Text style={styles.outputValue}>
            <Text style={{ color: colors.chart.generation, fontWeight: 'bold' }}>
              {resource.currentOutput.toFixed(1)}
            </Text>
            <Text style={styles.outputCapacity}> / {resource.capacityMw} MW</Text>
          </Text>
        </View>
        <View style={styles.progressBarContainer}>
          <View
            style={[
              styles.progressBar,
              {
                width: `${resource.utilizationRate}%`,
                backgroundColor: resource.utilizationRate > 80
                  ? colors.status.success
                  : resource.utilizationRate > 50
                  ? colors.status.warning
                  : colors.status.danger,
              },
            ]}
          />
        </View>
        <Text style={styles.utilizationText}>{resource.utilizationRate}% utilization</Text>
      </View>

      {/* Stats */}
      <View style={styles.cardStats}>
        <View style={styles.cardStatItem}>
          <Icon name="flash-outline" size={16} color={colors.chart.solar} />
          <Text style={styles.cardStatValue}>{resource.todayGeneration.toFixed(1)}</Text>
          <Text style={styles.cardStatUnit}>MWh today</Text>
        </View>
        <View style={styles.cardStatDivider} />
        <View style={styles.cardStatItem}>
          <Icon name="wallet-outline" size={16} color={colors.brand.accent} />
          <Text style={styles.cardStatValue}>₩{resource.monthlyRevenue.toFixed(1)}M</Text>
          <Text style={styles.cardStatUnit}>this month</Text>
        </View>
      </View>
    </TouchableOpacity>
  );
}

// Filter Chips
type ResourceFilter = 'all' | 'solar' | 'wind' | 'active' | 'maintenance';

function FilterChips({
  activeFilter,
  onFilterChange,
}: {
  activeFilter: ResourceFilter;
  onFilterChange: (filter: ResourceFilter) => void;
}) {
  const filters: { key: ResourceFilter; label: string; icon: string }[] = [
    { key: 'all', label: 'All', icon: 'layers' },
    { key: 'solar', label: 'Solar', icon: 'sunny' },
    { key: 'wind', label: 'Wind', icon: 'cloudy' },
    { key: 'active', label: 'Active', icon: 'checkmark-circle' },
    { key: 'maintenance', label: 'Maintenance', icon: 'construct' },
  ];

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.filterChips}
      contentContainerStyle={styles.filterChipsContent}
    >
      {filters.map((filter) => (
        <TouchableOpacity
          key={filter.key}
          style={[
            styles.filterChip,
            activeFilter === filter.key && styles.filterChipActive,
          ]}
          onPress={() => onFilterChange(filter.key)}
        >
          <Icon
            name={filter.icon}
            size={14}
            color={activeFilter === filter.key ? colors.text.primary : colors.text.muted}
          />
          <Text
            style={[
              styles.filterChipText,
              activeFilter === filter.key && styles.filterChipTextActive,
            ]}
          >
            {filter.label}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

export default function PortfolioScreen() {
  const [resources] = useState<RenewableResource[]>(mockResources);
  const [activeFilter, setActiveFilter] = useState<ResourceFilter>('all');

  const filteredResources = resources.filter((resource) => {
    if (activeFilter === 'all') return true;
    if (activeFilter === 'solar' || activeFilter === 'wind') {
      return resource.type === activeFilter;
    }
    if (activeFilter === 'active' || activeFilter === 'maintenance') {
      return resource.status === activeFilter;
    }
    return true;
  });

  const renderResource = useCallback(
    ({ item }: { item: RenewableResource }) => <ResourceCard resource={item} />,
    []
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={filteredResources}
        renderItem={renderResource}
        keyExtractor={(item) => item.id}
        ListHeaderComponent={
          <>
            <PortfolioSummary resources={resources} />
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Resources</Text>
              <Text style={styles.resourceCount}>{filteredResources.length} items</Text>
            </View>
            <FilterChips activeFilter={activeFilter} onFilterChange={setActiveFilter} />
          </>
        }
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  listContent: {
    padding: spacing.md,
    paddingBottom: spacing.xxl,
  },

  // Summary
  summaryContainer: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  statBox: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: fontSize.xl,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  statLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginTop: 2,
  },
  chartContainer: {
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  chartTitle: {
    fontSize: fontSize.md,
    color: colors.text.secondary,
    marginBottom: spacing.sm,
  },

  // Section
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  sectionTitle: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  resourceCount: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },

  // Filter Chips
  filterChips: {
    marginBottom: spacing.md,
  },
  filterChipsContent: {
    gap: spacing.sm,
  },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.card,
    borderWidth: 1,
    borderColor: colors.border.primary,
    marginRight: spacing.sm,
  },
  filterChipActive: {
    backgroundColor: colors.brand.primary,
    borderColor: colors.brand.primary,
  },
  filterChipText: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  filterChipTextActive: {
    color: colors.text.primary,
    fontWeight: '600',
  },

  // Resource Card
  resourceCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: spacing.md,
  },
  resourceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  resourceIconContainer: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  resourceName: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 2,
  },
  resourceLocation: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: borderRadius.sm,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statusText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },

  // Output Section
  outputSection: {
    marginBottom: spacing.md,
  },
  outputHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xs,
  },
  outputLabel: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  outputValue: {
    fontSize: fontSize.md,
  },
  outputCapacity: {
    color: colors.text.muted,
  },
  progressBarContainer: {
    height: 8,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
  },
  progressBar: {
    height: '100%',
    borderRadius: borderRadius.sm,
  },
  utilizationText: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginTop: 4,
    textAlign: 'right',
  },

  // Card Stats
  cardStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  cardStatItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  cardStatValue: {
    fontSize: fontSize.md,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  cardStatUnit: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
  },
  cardStatDivider: {
    width: 1,
    height: 20,
    backgroundColor: colors.border.primary,
  },
});
