/**
 * RE-BMS Bidding Screen
 * 10-Segment Bidding Matrix with Monotonic Constraint
 * Connected to real SMP prediction API for AI optimization
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Alert,
  Dimensions,
  Platform,
  ActivityIndicator,
} from 'react-native';

// Conditional imports for native-only features
let useNavigation: any = null;
let Ionicons: any = null;

if (Platform.OS !== 'web') {
  try {
    useNavigation = require('@react-navigation/native').useNavigation;
    Ionicons = require('@expo/vector-icons').Ionicons;
  } catch (e) {
    console.log('Native navigation/icons not available');
  }
}

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';
import { apiService, OptimizedBids, Resource } from '../services/api';

const { width: screenWidth } = Dimensions.get('window');

// Web-compatible alert function
const showAlert = (title: string, message: string, buttons?: any[]) => {
  if (Platform.OS === 'web') {
    // Simple web alert
    const result = window.confirm(`${title}\n\n${message}\n\nClick OK for DAM Bid, Cancel to dismiss`);
    if (result && buttons?.[1]?.onPress) {
      buttons[1].onPress();
    }
  } else {
    Alert.alert(title, message, buttons);
  }
};

// Types
interface BidSummary {
  id: string;
  resourceName: string;
  resourceType: 'solar' | 'wind';
  market: 'DAM' | 'RTM';
  tradingDate: string;
  status: 'draft' | 'pending' | 'submitted' | 'accepted' | 'rejected';
  totalQuantity: number;
  avgPrice: number;
  filledSegments: number;
}

// Navigation type - only used on native

// Mock data
const mockBids: BidSummary[] = [
  {
    id: 'bid-001',
    resourceName: 'Jeju Solar #1',
    resourceType: 'solar',
    market: 'DAM',
    tradingDate: '2025-12-20',
    status: 'draft',
    totalQuantity: 45.5,
    avgPrice: 125.3,
    filledSegments: 7,
  },
  {
    id: 'bid-002',
    resourceName: 'Jeju Wind #1',
    resourceType: 'wind',
    market: 'DAM',
    tradingDate: '2025-12-20',
    status: 'pending',
    totalQuantity: 32.0,
    avgPrice: 118.5,
    filledSegments: 5,
  },
  {
    id: 'bid-003',
    resourceName: 'Jeju Solar #2',
    resourceType: 'solar',
    market: 'RTM',
    tradingDate: '2025-12-19',
    status: 'submitted',
    totalQuantity: 28.3,
    avgPrice: 142.7,
    filledSegments: 8,
  },
  {
    id: 'bid-004',
    resourceName: 'Jeju Wind #2',
    resourceType: 'wind',
    market: 'DAM',
    tradingDate: '2025-12-20',
    status: 'accepted',
    totalQuantity: 55.2,
    avgPrice: 131.2,
    filledSegments: 10,
  },
];

// Segment visualization component
function SegmentBar({ filledSegments }: { filledSegments: number }) {
  return (
    <View style={styles.segmentBar}>
      {Array.from({ length: 10 }, (_, i) => (
        <View
          key={i}
          style={[
            styles.segment,
            {
              backgroundColor: i < filledSegments
                ? colors.segments[i]
                : colors.background.tertiary,
            },
          ]}
        />
      ))}
    </View>
  );
}

// Status badge component
function StatusBadge({ status }: { status: BidSummary['status'] }) {
  const statusConfig = {
    draft: { color: colors.text.muted, bg: colors.background.tertiary, label: 'Draft' },
    pending: { color: colors.status.warning, bg: `${colors.status.warning}20`, label: 'Pending' },
    submitted: { color: colors.status.info, bg: `${colors.status.info}20`, label: 'Submitted' },
    accepted: { color: colors.status.success, bg: `${colors.status.success}20`, label: 'Accepted' },
    rejected: { color: colors.status.danger, bg: `${colors.status.danger}20`, label: 'Rejected' },
  };

  const config = statusConfig[status];

  return (
    <View style={[styles.statusBadge, { backgroundColor: config.bg }]}>
      <Text style={[styles.statusText, { color: config.color }]}>{config.label}</Text>
    </View>
  );
}

// Icon component that works on both web and native
function Icon({ name, size, color }: { name: string; size: number; color: string }) {
  if (Ionicons) {
    return <Ionicons name={name as any} size={size} color={color} />;
  }
  // Web fallback - emoji icons
  const iconMap: { [key: string]: string } = {
    'sunny': '☀️',
    'cloudy': '💨',
    'create-outline': '✏️',
    'sparkles': '✨',
    'send': '📤',
    'time-outline': '⏳',
    'eye-outline': '👁️',
    'document-outline': '📄',
    'add': '+',
  };
  return (
    <Text style={{ fontSize: size * 0.8, color }}>{iconMap[name] || '•'}</Text>
  );
}

// Bid card component
function BidCard({
  bid,
  onPress,
  onOptimize,
  loading,
}: {
  bid: BidSummary;
  onPress: () => void;
  onOptimize?: (bidId: string) => void;
  loading?: boolean;
}) {
  const resourceIcon = bid.resourceType === 'solar' ? 'sunny' : 'cloudy';
  const resourceColor = bid.resourceType === 'solar' ? colors.chart.solar : colors.chart.wind;

  return (
    <TouchableOpacity style={styles.bidCard} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.bidHeader}>
        <View style={styles.resourceInfo}>
          <Icon name={resourceIcon} size={20} color={resourceColor} />
          <Text style={styles.resourceName}>{bid.resourceName}</Text>
        </View>
        <StatusBadge status={bid.status} />
      </View>

      <View style={styles.bidDetails}>
        <View style={styles.detailRow}>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>Market</Text>
            <Text style={styles.detailValue}>{bid.market}</Text>
          </View>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>Date</Text>
            <Text style={styles.detailValue}>{bid.tradingDate}</Text>
          </View>
        </View>
        <View style={styles.detailRow}>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>Quantity</Text>
            <Text style={[styles.detailValue, { color: colors.chart.generation }]}>
              {bid.totalQuantity.toFixed(1)} MW
            </Text>
          </View>
          <View style={styles.detailItem}>
            <Text style={styles.detailLabel}>Avg Price</Text>
            <Text style={[styles.detailValue, { color: colors.chart.smp }]}>
              ₩{bid.avgPrice.toFixed(1)}
            </Text>
          </View>
        </View>
      </View>

      <View style={styles.segmentSection}>
        <Text style={styles.segmentLabel}>
          Segments: {bid.filledSegments}/10
        </Text>
        <SegmentBar filledSegments={bid.filledSegments} />
      </View>

      <View style={styles.bidActions}>
        {bid.status === 'draft' && (
          <>
            <TouchableOpacity style={styles.actionBtn}>
              <Icon name="create-outline" size={16} color={colors.text.secondary} />
              <Text style={styles.actionText}>Edit</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.actionBtn}
              onPress={() => onOptimize?.(bid.id)}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator size="small" color={colors.brand.accent} />
              ) : (
                <>
                  <Icon name="sparkles" size={16} color={colors.brand.accent} />
                  <Text style={[styles.actionText, { color: colors.brand.accent }]}>AI Optimize</Text>
                </>
              )}
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.submitBtn]}>
              <Icon name="send" size={16} color={colors.text.inverse} />
              <Text style={[styles.actionText, { color: colors.text.inverse }]}>Submit</Text>
            </TouchableOpacity>
          </>
        )}
        {bid.status === 'pending' && (
          <TouchableOpacity style={styles.actionBtn}>
            <Icon name="time-outline" size={16} color={colors.status.warning} />
            <Text style={[styles.actionText, { color: colors.status.warning }]}>
              Awaiting Submission
            </Text>
          </TouchableOpacity>
        )}
        {(bid.status === 'submitted' || bid.status === 'accepted') && (
          <TouchableOpacity style={styles.actionBtn}>
            <Icon name="eye-outline" size={16} color={colors.text.secondary} />
            <Text style={styles.actionText}>View Details</Text>
          </TouchableOpacity>
        )}
      </View>
    </TouchableOpacity>
  );
}

// Filter tabs
type FilterStatus = 'all' | 'draft' | 'pending' | 'submitted' | 'accepted';

function FilterTabs({
  activeFilter,
  onFilterChange
}: {
  activeFilter: FilterStatus;
  onFilterChange: (filter: FilterStatus) => void;
}) {
  const filters: { key: FilterStatus; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'draft', label: 'Draft' },
    { key: 'pending', label: 'Pending' },
    { key: 'submitted', label: 'Submitted' },
    { key: 'accepted', label: 'Accepted' },
  ];

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.filterContainer}
      contentContainerStyle={styles.filterContent}
    >
      {filters.map((filter) => (
        <TouchableOpacity
          key={filter.key}
          style={[
            styles.filterTab,
            activeFilter === filter.key && styles.filterTabActive,
          ]}
          onPress={() => onFilterChange(filter.key)}
        >
          <Text
            style={[
              styles.filterText,
              activeFilter === filter.key && styles.filterTextActive,
            ]}
          >
            {filter.label}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

export default function BiddingScreen() {
  // Navigation only available on native
  const navigation = useNavigation ? useNavigation() : null;
  const [activeFilter, setActiveFilter] = useState<FilterStatus>('all');
  const [bids, setBids] = useState<BidSummary[]>(mockBids);
  const [selectedBid, setSelectedBid] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [optimizedBids, setOptimizedBids] = useState<OptimizedBids | null>(null);
  const [resources, setResources] = useState<Resource[]>([]);
  const [modelInfo, setModelInfo] = useState<string>('');

  // Fetch resources and model info on mount
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [resourcesData, modelData] = await Promise.all([
          apiService.getResources(),
          apiService.getModelInfo(),
        ]);
        setResources(resourcesData);
        setModelInfo(modelData.status === 'ready' ? `${modelData.version}` : 'fallback');
      } catch (err) {
        console.error('Failed to fetch data:', err);
      }
    };
    fetchData();
  }, []);

  const filteredBids = activeFilter === 'all'
    ? bids
    : bids.filter(bid => bid.status === activeFilter);

  const handleBidPress = useCallback((bidId: string) => {
    if (Platform.OS === 'web') {
      // On web, show bid details inline or in modal
      setSelectedBid(bidId);
      console.log('Selected bid:', bidId);
    } else if (navigation) {
      navigation.navigate('BidDetail', { bidId });
    }
  }, [navigation]);

  // AI Optimize using real SMP model
  const handleOptimizeBid = useCallback(async (bidId: string, capacity: number = 50) => {
    setLoading(true);
    try {
      const optimized = await apiService.getOptimizedSegments(capacity, 'moderate');
      setOptimizedBids(optimized);

      // Update the bid with optimized data
      setBids(prevBids => prevBids.map(bid => {
        if (bid.id === bidId) {
          const firstHour = optimized.hourly_bids[0];
          return {
            ...bid,
            status: 'pending' as const,
            totalQuantity: firstHour.total_mw,
            avgPrice: firstHour.avg_price,
            filledSegments: 10,
          };
        }
        return bid;
      }));

      showAlert(
        'AI Optimization Complete',
        `Model: ${optimized.model_used}\nTotal: ${optimized.total_daily_mwh.toFixed(1)} MWh\nRisk: ${optimized.risk_level}`,
        [{ text: 'OK' }]
      );
    } catch (err) {
      console.error('Optimization failed:', err);
      showAlert('Error', 'Failed to optimize bid. API may be unavailable.', [{ text: 'OK' }]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleCreateBid = useCallback(() => {
    showAlert(
      'Create New Bid',
      'Select resource and market to create a new bid.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'DAM Bid', onPress: () => console.log('Create DAM bid') },
        { text: 'RTM Bid', onPress: () => console.log('Create RTM bid') },
      ]
    );
  }, []);

  const renderBidItem = useCallback(({ item }: { item: BidSummary }) => (
    <BidCard
      bid={item}
      onPress={() => handleBidPress(item.id)}
      onOptimize={handleOptimizeBid}
      loading={loading}
    />
  ), [handleBidPress, handleOptimizeBid, loading]);

  return (
    <View style={styles.container}>
      {/* Model Info Banner */}
      {modelInfo && (
        <View style={styles.modelBanner}>
          <Text style={styles.modelText}>🤖 SMP Model: {modelInfo} | Tap "AI Optimize" to use real predictions</Text>
        </View>
      )}

      {/* Summary Header */}
      <View style={styles.summaryContainer}>
        <View style={styles.summaryItem}>
          <Text style={styles.summaryValue}>{bids.length}</Text>
          <Text style={styles.summaryLabel}>Total</Text>
        </View>
        <View style={styles.summaryDivider} />
        <View style={styles.summaryItem}>
          <Text style={[styles.summaryValue, { color: colors.text.muted }]}>
            {bids.filter(b => b.status === 'draft').length}
          </Text>
          <Text style={styles.summaryLabel}>Draft</Text>
        </View>
        <View style={styles.summaryDivider} />
        <View style={styles.summaryItem}>
          <Text style={[styles.summaryValue, { color: colors.status.warning }]}>
            {bids.filter(b => b.status === 'pending').length}
          </Text>
          <Text style={styles.summaryLabel}>Pending</Text>
        </View>
        <View style={styles.summaryDivider} />
        <View style={styles.summaryItem}>
          <Text style={[styles.summaryValue, { color: colors.status.success }]}>
            {bids.filter(b => b.status === 'accepted').length}
          </Text>
          <Text style={styles.summaryLabel}>Accepted</Text>
        </View>
      </View>

      {/* Filter Tabs */}
      <FilterTabs activeFilter={activeFilter} onFilterChange={setActiveFilter} />

      {/* Bid List */}
      <FlatList
        data={filteredBids}
        renderItem={renderBidItem}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContent}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Icon name="document-outline" size={48} color={colors.text.muted} />
            <Text style={styles.emptyText}>No bids found</Text>
          </View>
        }
      />

      {/* Floating Action Button */}
      <TouchableOpacity style={styles.fab} onPress={handleCreateBid} activeOpacity={0.8}>
        <Icon name="add" size={28} color={colors.text.primary} />
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },

  // Summary
  summaryContainer: {
    flexDirection: 'row',
    backgroundColor: colors.background.secondary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: fontSize.xxl,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  summaryLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginTop: 2,
  },
  summaryDivider: {
    width: 1,
    height: 30,
    backgroundColor: colors.border.primary,
  },

  // Filters
  filterContainer: {
    backgroundColor: colors.background.secondary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  filterContent: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  filterTab: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
    backgroundColor: colors.background.tertiary,
    marginRight: spacing.sm,
  },
  filterTabActive: {
    backgroundColor: colors.brand.primary,
  },
  filterText: {
    fontSize: fontSize.sm,
    color: colors.text.secondary,
  },
  filterTextActive: {
    color: colors.text.primary,
    fontWeight: '600',
  },

  // List
  listContent: {
    padding: spacing.md,
    paddingBottom: 100,
  },

  // Bid Card
  bidCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  bidHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  resourceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  resourceName: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  statusBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  statusText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },
  bidDetails: {
    marginBottom: spacing.md,
  },
  detailRow: {
    flexDirection: 'row',
    marginBottom: spacing.sm,
  },
  detailItem: {
    flex: 1,
  },
  detailLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginBottom: 2,
  },
  detailValue: {
    fontSize: fontSize.md,
    color: colors.text.primary,
    fontWeight: '500',
  },

  // Segments
  segmentSection: {
    marginBottom: spacing.md,
  },
  segmentLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginBottom: spacing.xs,
  },
  segmentBar: {
    flexDirection: 'row',
    height: 8,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
    gap: 2,
  },
  segment: {
    flex: 1,
    borderRadius: 2,
  },

  // Actions
  bidActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: spacing.sm,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  actionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.tertiary,
  },
  submitBtn: {
    backgroundColor: colors.brand.primary,
  },
  actionText: {
    fontSize: fontSize.sm,
    color: colors.text.secondary,
  },

  // Empty State
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xxl,
  },
  emptyText: {
    fontSize: fontSize.md,
    color: colors.text.muted,
    marginTop: spacing.md,
  },

  // FAB
  fab: {
    position: 'absolute',
    right: spacing.lg,
    bottom: spacing.lg,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.brand.primary,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: colors.brand.primary,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },

  // Model Banner
  modelBanner: {
    backgroundColor: `${colors.brand.primary}20`,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  modelText: {
    color: colors.brand.primary,
    fontSize: fontSize.xs,
    textAlign: 'center',
    fontWeight: '500',
  },
});
