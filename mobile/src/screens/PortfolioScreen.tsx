/**
 * RE-BMS Portfolio Screen
 * Renewable Resource Management - Real Jeju Power Plant Data
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Dimensions,
  Platform,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';

// PieChart conditional import
let PieChart: any = null;
if (Platform.OS !== 'web') {
  try {
    PieChart = require('react-native-chart-kit').PieChart;
  } catch (e) {
    console.log('PieChart not available');
  }
}

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';
import apiService, { Resource } from '../services/api';

// Icon component using emoji for cross-platform compatibility
function Icon({ name, size, color }: { name: string; size: number; color: string }) {
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
    <Text style={{ fontSize: size * 0.8, color }}>{iconMap[name] || '•'}</Text>
  );
}

const { width: screenWidth } = Dimensions.get('window');

// Extended Resource type with additional display fields
interface DisplayResource extends Resource {
  todayGeneration: number;
  monthlyRevenue: number;
  utilizationRate: number;
  operator?: string;
  subtype?: string;
  name_en?: string;
}

// Portfolio Summary Component
function PortfolioSummary({ resources }: { resources: DisplayResource[] }) {
  const totalCapacity = resources.reduce((sum, r) => sum + r.capacity, 0);
  const totalOutput = resources.reduce((sum, r) => sum + r.current_output, 0);
  const avgUtilization = totalCapacity > 0 ? (totalOutput / totalCapacity) * 100 : 0;
  const totalRevenue = resources.reduce((sum, r) => sum + r.monthlyRevenue, 0);

  const solarCapacity = resources
    .filter(r => r.type === 'solar')
    .reduce((sum, r) => sum + r.capacity, 0);
  const windCapacity = resources
    .filter(r => r.type === 'wind')
    .reduce((sum, r) => sum + r.capacity, 0);

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
function ResourceCard({ resource }: { resource: DisplayResource }) {
  const statusColors: Record<string, string> = {
    active: colors.status.success,
    maintenance: colors.status.warning,
    offline: colors.status.danger,
  };

  const resourceIcon = resource.type === 'solar' ? 'sunny' : 'cloudy';
  const resourceColor = resource.type === 'solar' ? colors.chart.solar : colors.chart.wind;
  const statusColor = statusColors[resource.status] || colors.status.warning;

  return (
    <TouchableOpacity style={styles.resourceCard} activeOpacity={0.7}>
      {/* Header */}
      <View style={styles.cardHeader}>
        <View style={styles.resourceInfo}>
          <View style={[styles.resourceIconContainer, { backgroundColor: `${resourceColor}20` }]}>
            <Icon name={resourceIcon} size={24} color={resourceColor} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.resourceName} numberOfLines={1}>
              {resource.name_en || resource.name}
            </Text>
            <Text style={styles.resourceNameKr} numberOfLines={1}>
              {resource.name}
            </Text>
            <View style={styles.locationRow}>
              <Icon name="location-outline" size={12} color={colors.text.muted} />
              <Text style={styles.resourceLocation} numberOfLines={1}>{resource.location}</Text>
            </View>
          </View>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: `${statusColor}20` }]}>
          <View style={[styles.statusDot, { backgroundColor: statusColor }]} />
          <Text style={[styles.statusText, { color: statusColor }]}>
            {resource.status.charAt(0).toUpperCase() + resource.status.slice(1)}
          </Text>
        </View>
      </View>

      {/* Subtype & Operator */}
      {(resource.subtype || resource.operator) && (
        <View style={styles.metaRow}>
          {resource.subtype && (
            <View style={styles.metaChip}>
              <Text style={styles.metaText}>
                {resource.subtype === 'onshore' ? '🏔 Onshore' :
                 resource.subtype === 'offshore' ? '🌊 Offshore' :
                 resource.subtype === 'rooftop' ? '🏠 Rooftop' :
                 resource.subtype === 'ground' ? '🌍 Ground' : resource.subtype}
              </Text>
            </View>
          )}
          {resource.operator && (
            <Text style={styles.operatorText} numberOfLines={1}>
              {resource.operator}
            </Text>
          )}
        </View>
      )}

      {/* Output Gauge */}
      <View style={styles.outputSection}>
        <View style={styles.outputHeader}>
          <Text style={styles.outputLabel}>Current Output</Text>
          <Text style={styles.outputValue}>
            <Text style={{ color: colors.chart.generation, fontWeight: 'bold' }}>
              {resource.current_output.toFixed(1)}
            </Text>
            <Text style={styles.outputCapacity}> / {resource.capacity} MW</Text>
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
        <Text style={styles.utilizationText}>{resource.utilizationRate.toFixed(0)}% utilization</Text>
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
  const [resources, setResources] = useState<DisplayResource[]>([]);
  const [activeFilter, setActiveFilter] = useState<ResourceFilter>('all');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch resources from API
  const fetchResources = useCallback(async () => {
    try {
      setError(null);
      const apiResources = await apiService.getResources();

      // Transform API response to DisplayResource format
      const displayResources: DisplayResource[] = apiResources.map((r: any) => ({
        ...r,
        utilizationRate: r.utilization || (r.current_output / r.capacity * 100) || 0,
        todayGeneration: r.current_output * 8, // Estimated 8 hours
        monthlyRevenue: r.current_output * 24 * 30 * 0.12 / 1000, // Estimated revenue
        operator: r.operator,
        subtype: r.subtype,
        name_en: r.name_en,
      }));

      // Sort by capacity descending
      displayResources.sort((a, b) => b.capacity - a.capacity);
      setResources(displayResources);
    } catch (err: any) {
      console.error('Failed to fetch resources:', err);
      setError('Failed to load Jeju plants');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  // Initial fetch and refresh interval
  useEffect(() => {
    fetchResources();
    const interval = setInterval(fetchResources, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchResources]);

  const handleRefresh = useCallback(() => {
    setRefreshing(true);
    fetchResources();
  }, [fetchResources]);

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
    ({ item }: { item: DisplayResource }) => <ResourceCard resource={item} />,
    []
  );

  // Loading state
  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.brand.primary} />
        <Text style={styles.loadingText}>Loading Jeju Power Plants...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Error Banner */}
      {error && (
        <View style={styles.errorBanner}>
          <Text style={styles.errorText}>⚠️ {error}</Text>
        </View>
      )}

      {/* Data Source Banner */}
      <View style={styles.dataBanner}>
        <Text style={styles.dataBannerText}>
          📍 Real Jeju Power Plants • {resources.length} Renewable Resources
        </Text>
      </View>

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
        ListEmptyComponent={
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>No resources found</Text>
          </View>
        }
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            tintColor={colors.brand.primary}
            colors={[colors.brand.primary]}
          />
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background.primary,
  },
  loadingText: {
    marginTop: spacing.md,
    fontSize: fontSize.md,
    color: colors.text.muted,
  },
  errorBanner: {
    backgroundColor: `${colors.status.danger}20`,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
  },
  errorText: {
    color: colors.status.danger,
    fontSize: fontSize.sm,
    textAlign: 'center',
  },
  dataBanner: {
    backgroundColor: colors.brand.primary + '15',
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  dataBannerText: {
    color: colors.brand.primary,
    fontSize: fontSize.xs,
    textAlign: 'center',
    fontWeight: '500',
  },
  emptyState: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  emptyText: {
    color: colors.text.muted,
    fontSize: fontSize.md,
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
    fontSize: fontSize.md,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  resourceNameKr: {
    fontSize: fontSize.xs,
    color: colors.text.secondary,
    marginTop: 1,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.sm,
    paddingLeft: 56,
  },
  metaChip: {
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  metaText: {
    fontSize: fontSize.xs,
    color: colors.text.secondary,
  },
  operatorText: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    flex: 1,
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
