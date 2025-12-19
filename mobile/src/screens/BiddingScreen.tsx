/**
 * RE-BMS Bidding Screen
 * 10-Segment Bidding Matrix with Monotonic Constraint
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  Alert,
  Dimensions,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';
import { BiddingStackParamList } from '../navigation/AppNavigator';

const { width: screenWidth } = Dimensions.get('window');

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

type BiddingNavigationProp = NativeStackNavigationProp<BiddingStackParamList, 'BiddingList'>;

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

// Bid card component
function BidCard({ bid, onPress }: { bid: BidSummary; onPress: () => void }) {
  const resourceIcon = bid.resourceType === 'solar' ? 'sunny' : 'cloudy';
  const resourceColor = bid.resourceType === 'solar' ? colors.chart.solar : colors.chart.wind;

  return (
    <TouchableOpacity style={styles.bidCard} onPress={onPress} activeOpacity={0.7}>
      <View style={styles.bidHeader}>
        <View style={styles.resourceInfo}>
          <Ionicons name={resourceIcon as any} size={20} color={resourceColor} />
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
              <Ionicons name="create-outline" size={16} color={colors.text.secondary} />
              <Text style={styles.actionText}>Edit</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.actionBtn}>
              <Ionicons name="sparkles" size={16} color={colors.brand.accent} />
              <Text style={[styles.actionText, { color: colors.brand.accent }]}>Optimize</Text>
            </TouchableOpacity>
            <TouchableOpacity style={[styles.actionBtn, styles.submitBtn]}>
              <Ionicons name="send" size={16} color={colors.text.inverse} />
              <Text style={[styles.actionText, { color: colors.text.inverse }]}>Submit</Text>
            </TouchableOpacity>
          </>
        )}
        {bid.status === 'pending' && (
          <TouchableOpacity style={styles.actionBtn}>
            <Ionicons name="time-outline" size={16} color={colors.status.warning} />
            <Text style={[styles.actionText, { color: colors.status.warning }]}>
              Awaiting Submission
            </Text>
          </TouchableOpacity>
        )}
        {(bid.status === 'submitted' || bid.status === 'accepted') && (
          <TouchableOpacity style={styles.actionBtn}>
            <Ionicons name="eye-outline" size={16} color={colors.text.secondary} />
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
  const navigation = useNavigation<BiddingNavigationProp>();
  const [activeFilter, setActiveFilter] = useState<FilterStatus>('all');
  const [bids, setBids] = useState<BidSummary[]>(mockBids);

  const filteredBids = activeFilter === 'all'
    ? bids
    : bids.filter(bid => bid.status === activeFilter);

  const handleBidPress = useCallback((bidId: string) => {
    navigation.navigate('BidDetail', { bidId });
  }, [navigation]);

  const handleCreateBid = useCallback(() => {
    Alert.alert(
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
    <BidCard bid={item} onPress={() => handleBidPress(item.id)} />
  ), [handleBidPress]);

  return (
    <View style={styles.container}>
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
            <Ionicons name="document-outline" size={48} color={colors.text.muted} />
            <Text style={styles.emptyText}>No bids found</Text>
          </View>
        }
      />

      {/* Floating Action Button */}
      <TouchableOpacity style={styles.fab} onPress={handleCreateBid} activeOpacity={0.8}>
        <Ionicons name="add" size={28} color={colors.text.primary} />
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
});
