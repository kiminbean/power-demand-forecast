/**
 * RE-BMS Bid Detail Screen
 * 10-Segment Bid Matrix Editor
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  Alert,
  Dimensions,
} from 'react-native';

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';

const { width: screenWidth } = Dimensions.get('window');

// Icon map for cross-platform compatibility
const iconMap: { [key: string]: string } = {
  'warning': '⚠️',
  'sunny': '☀️',
  'cloudy': '💨',
  'information-circle': 'ℹ️',
  'sparkles': '✨',
  'send': '📤',
};

interface BidSegment {
  segmentId: number;
  quantityMw: number;
  priceKrwMwh: number;
}

interface HourlyBid {
  hour: number;
  segments: BidSegment[];
}

// Mock data for bid detail
const mockBidDetail = {
  id: 'bid-001',
  resourceName: 'Jeju Solar #1',
  resourceType: 'solar' as const,
  market: 'DAM' as const,
  tradingDate: '2025-12-20',
  status: 'draft' as const,
  capacityMw: 50.0,
  hourlyBids: Array.from({ length: 24 }, (_, hour) => ({
    hour: hour + 1,
    segments: Array.from({ length: 10 }, (_, seg) => ({
      segmentId: seg + 1,
      quantityMw: seg < 7 ? 5.0 + seg * 0.5 : 0,
      priceKrwMwh: seg < 7 ? 100 + seg * 10 : 0,
    })),
  })),
};

// Segment Editor Component
function SegmentEditor({
  segment,
  onQuantityChange,
  onPriceChange,
  isEditable,
  prevPrice,
}: {
  segment: BidSegment;
  onQuantityChange: (value: number) => void;
  onPriceChange: (value: number) => void;
  isEditable: boolean;
  prevPrice?: number;
}) {
  const [localQty, setLocalQty] = useState(segment.quantityMw.toString());
  const [localPrice, setLocalPrice] = useState(segment.priceKrwMwh.toString());

  const hasMonotonicError = prevPrice !== undefined &&
    segment.priceKrwMwh > 0 &&
    segment.priceKrwMwh < prevPrice;

  return (
    <View style={[styles.segmentRow, hasMonotonicError && styles.segmentError]}>
      <View style={[styles.segmentIdBadge, { backgroundColor: colors.segments[segment.segmentId - 1] }]}>
        <Text style={styles.segmentIdText}>{segment.segmentId}</Text>
      </View>

      <View style={styles.segmentInputGroup}>
        <TextInput
          style={[styles.segmentInput, !isEditable && styles.inputDisabled]}
          value={localQty}
          onChangeText={setLocalQty}
          onEndEditing={() => onQuantityChange(parseFloat(localQty) || 0)}
          keyboardType="decimal-pad"
          editable={isEditable}
          placeholder="0.0"
          placeholderTextColor={colors.text.muted}
        />
        <Text style={styles.unitLabel}>MW</Text>
      </View>

      <View style={styles.segmentInputGroup}>
        <TextInput
          style={[
            styles.segmentInput,
            !isEditable && styles.inputDisabled,
            hasMonotonicError && styles.inputError,
          ]}
          value={localPrice}
          onChangeText={setLocalPrice}
          onEndEditing={() => onPriceChange(parseFloat(localPrice) || 0)}
          keyboardType="decimal-pad"
          editable={isEditable}
          placeholder="0"
          placeholderTextColor={colors.text.muted}
        />
        <Text style={styles.unitLabel}>₩</Text>
      </View>

      {hasMonotonicError && (
        <Text style={{ fontSize: 14 }}>{iconMap['warning']}</Text>
      )}
    </View>
  );
}

// Hour Selector
function HourSelector({
  selectedHour,
  onHourChange,
}: {
  selectedHour: number;
  onHourChange: (hour: number) => void;
}) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.hourSelector}
      contentContainerStyle={styles.hourSelectorContent}
    >
      {Array.from({ length: 24 }, (_, i) => i + 1).map((hour) => (
        <TouchableOpacity
          key={hour}
          style={[
            styles.hourButton,
            selectedHour === hour && styles.hourButtonActive,
          ]}
          onPress={() => onHourChange(hour)}
        >
          <Text
            style={[
              styles.hourButtonText,
              selectedHour === hour && styles.hourButtonTextActive,
            ]}
          >
            {hour.toString().padStart(2, '0')}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
}

// Step Chart Visualization
function StepChart({ segments }: { segments: BidSegment[] }) {
  const maxPrice = Math.max(...segments.map(s => s.priceKrwMwh), 200);
  const chartHeight = 100;
  const chartWidth = screenWidth - spacing.lg * 2 - spacing.md * 2;
  const barWidth = chartWidth / 10;

  return (
    <View style={styles.stepChartContainer}>
      <Text style={styles.chartTitle}>Bid Curve (Step Chart)</Text>
      <View style={styles.stepChart}>
        {segments.map((segment, index) => {
          const height = segment.priceKrwMwh > 0
            ? (segment.priceKrwMwh / maxPrice) * chartHeight
            : 0;
          const hasQty = segment.quantityMw > 0;

          return (
            <View
              key={segment.segmentId}
              style={[
                styles.stepBar,
                {
                  height: height,
                  width: barWidth - 4,
                  backgroundColor: hasQty
                    ? colors.segments[index]
                    : colors.background.tertiary,
                },
              ]}
            >
              {hasQty && height > 20 && (
                <Text style={styles.stepBarLabel}>{segment.priceKrwMwh}</Text>
              )}
            </View>
          );
        })}
      </View>
      <View style={styles.chartLabels}>
        {segments.map((segment) => (
          <Text key={segment.segmentId} style={styles.chartLabel}>
            {segment.quantityMw > 0 ? segment.quantityMw.toFixed(1) : '-'}
          </Text>
        ))}
      </View>
    </View>
  );
}

interface BidDetailScreenProps {
  bidId?: string;
  onBack?: () => void;
}

export default function BidDetailScreen({ bidId = 'bid-001', onBack }: BidDetailScreenProps) {

  const [bid, setBid] = useState(mockBidDetail);
  const [selectedHour, setSelectedHour] = useState(1);
  const [isEditing, setIsEditing] = useState(true);

  const currentHourBid = bid.hourlyBids.find(h => h.hour === selectedHour);
  const segments = currentHourBid?.segments || [];

  const handleQuantityChange = useCallback((segmentId: number, value: number) => {
    setBid(prev => ({
      ...prev,
      hourlyBids: prev.hourlyBids.map(hb =>
        hb.hour === selectedHour
          ? {
              ...hb,
              segments: hb.segments.map(s =>
                s.segmentId === segmentId ? { ...s, quantityMw: value } : s
              ),
            }
          : hb
      ),
    }));
  }, [selectedHour]);

  const handlePriceChange = useCallback((segmentId: number, value: number) => {
    setBid(prev => ({
      ...prev,
      hourlyBids: prev.hourlyBids.map(hb =>
        hb.hour === selectedHour
          ? {
              ...hb,
              segments: hb.segments.map(s =>
                s.segmentId === segmentId ? { ...s, priceKrwMwh: value } : s
              ),
            }
          : hb
      ),
    }));
  }, [selectedHour]);

  const handleOptimize = useCallback(() => {
    Alert.alert(
      'AI Optimization',
      'Apply AI-powered bid optimization based on SMP forecast?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Conservative',
          onPress: () => console.log('Conservative optimization'),
        },
        {
          text: 'Aggressive',
          onPress: () => console.log('Aggressive optimization'),
        },
      ]
    );
  }, []);

  const handleSubmit = useCallback(() => {
    Alert.alert(
      'Submit Bid',
      'Submit this bid to KPX? This action cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Submit',
          style: 'destructive',
          onPress: () => {
            console.log('Submitting bid:', bidId);
            onBack?.();
          },
        },
      ]
    );
  }, [bidId, onBack]);

  const totalQuantity = segments.reduce((sum, s) => sum + s.quantityMw, 0);
  const avgPrice = segments.filter(s => s.quantityMw > 0).length > 0
    ? segments.reduce((sum, s) => sum + s.priceKrwMwh * s.quantityMw, 0) / totalQuantity
    : 0;

  return (
    <View style={styles.container}>
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
        {/* Header Info */}
        <View style={styles.headerCard}>
          <View style={styles.headerRow}>
            <View style={styles.resourceBadge}>
              <Text style={{ fontSize: 18 }}>
                {iconMap[bid.resourceType === 'solar' ? 'sunny' : 'cloudy']}
              </Text>
              <Text style={styles.resourceName}>{bid.resourceName}</Text>
            </View>
            <View style={styles.marketBadge}>
              <Text style={styles.marketText}>{bid.market}</Text>
            </View>
          </View>
          <View style={styles.headerStats}>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Trading Date</Text>
              <Text style={styles.statValue}>{bid.tradingDate}</Text>
            </View>
            <View style={styles.statItem}>
              <Text style={styles.statLabel}>Capacity</Text>
              <Text style={styles.statValue}>{bid.capacityMw} MW</Text>
            </View>
          </View>
        </View>

        {/* Hour Selector */}
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Select Hour</Text>
          <Text style={styles.sectionSubtitle}>Hour {selectedHour}:00</Text>
        </View>
        <HourSelector selectedHour={selectedHour} onHourChange={setSelectedHour} />

        {/* Step Chart */}
        <StepChart segments={segments} />

        {/* Segment Editor */}
        <View style={styles.segmentEditorCard}>
          <View style={styles.segmentEditorHeader}>
            <Text style={styles.sectionTitle}>10-Segment Bid</Text>
            <View style={styles.summaryBadges}>
              <View style={styles.summaryBadge}>
                <Text style={styles.summaryValue}>{totalQuantity.toFixed(1)}</Text>
                <Text style={styles.summaryUnit}>MW</Text>
              </View>
              <View style={styles.summaryBadge}>
                <Text style={styles.summaryValue}>₩{avgPrice.toFixed(0)}</Text>
                <Text style={styles.summaryUnit}>avg</Text>
              </View>
            </View>
          </View>

          <View style={styles.segmentHeader}>
            <Text style={[styles.segmentHeaderText, { flex: 0.5 }]}>Seg</Text>
            <Text style={[styles.segmentHeaderText, { flex: 1 }]}>Quantity</Text>
            <Text style={[styles.segmentHeaderText, { flex: 1 }]}>Price</Text>
          </View>

          {segments.map((segment, index) => (
            <SegmentEditor
              key={segment.segmentId}
              segment={segment}
              onQuantityChange={(value) => handleQuantityChange(segment.segmentId, value)}
              onPriceChange={(value) => handlePriceChange(segment.segmentId, value)}
              isEditable={isEditing && bid.status === 'draft'}
              prevPrice={index > 0 ? segments[index - 1].priceKrwMwh : undefined}
            />
          ))}

          <View style={styles.monotonicNote}>
            <Text style={{ fontSize: 12 }}>{iconMap['information-circle']}</Text>
            <Text style={styles.monotonicNoteText}>
              Prices must be monotonically increasing (Seg 1 ≤ Seg 2 ≤ ... ≤ Seg 10)
            </Text>
          </View>
        </View>
      </ScrollView>

      {/* Bottom Actions */}
      {bid.status === 'draft' && (
        <View style={styles.bottomActions}>
          <TouchableOpacity style={styles.optimizeBtn} onPress={handleOptimize}>
            <Text style={{ fontSize: 18 }}>{iconMap['sparkles']}</Text>
            <Text style={styles.optimizeBtnText}>AI Optimize</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.submitBtn} onPress={handleSubmit}>
            <Text style={{ fontSize: 18 }}>{iconMap['send']}</Text>
            <Text style={styles.submitBtnText}>Submit to KPX</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: spacing.md,
    paddingBottom: 100,
  },

  // Header Card
  headerCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  resourceBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  resourceName: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  marketBadge: {
    backgroundColor: colors.brand.primary,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.md,
  },
  marketText: {
    color: colors.text.primary,
    fontWeight: 'bold',
    fontSize: fontSize.sm,
  },
  headerStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  statItem: {
    alignItems: 'center',
  },
  statLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginBottom: 2,
  },
  statValue: {
    fontSize: fontSize.md,
    color: colors.text.primary,
    fontWeight: '500',
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
  sectionSubtitle: {
    fontSize: fontSize.sm,
    color: colors.brand.primary,
    fontWeight: '600',
  },

  // Hour Selector
  hourSelector: {
    marginBottom: spacing.md,
  },
  hourSelectorContent: {
    gap: spacing.xs,
  },
  hourButton: {
    width: 44,
    height: 44,
    borderRadius: borderRadius.md,
    backgroundColor: colors.background.card,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: spacing.xs,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  hourButtonActive: {
    backgroundColor: colors.brand.primary,
    borderColor: colors.brand.primary,
  },
  hourButtonText: {
    fontSize: fontSize.md,
    color: colors.text.secondary,
    fontWeight: '500',
  },
  hourButtonTextActive: {
    color: colors.text.primary,
    fontWeight: 'bold',
  },

  // Step Chart
  stepChartContainer: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  chartTitle: {
    fontSize: fontSize.md,
    color: colors.text.secondary,
    marginBottom: spacing.md,
  },
  stepChart: {
    flexDirection: 'row',
    height: 100,
    alignItems: 'flex-end',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.xs,
  },
  stepBar: {
    borderRadius: borderRadius.sm,
    justifyContent: 'flex-end',
    alignItems: 'center',
    paddingBottom: 4,
  },
  stepBarLabel: {
    fontSize: 8,
    color: colors.text.inverse,
    fontWeight: 'bold',
  },
  chartLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: spacing.xs,
    paddingHorizontal: spacing.xs,
  },
  chartLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    width: (screenWidth - spacing.lg * 2 - spacing.md * 2) / 10 - 4,
    textAlign: 'center',
  },

  // Segment Editor
  segmentEditorCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  segmentEditorHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  summaryBadges: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  summaryBadge: {
    flexDirection: 'row',
    alignItems: 'baseline',
    backgroundColor: colors.background.tertiary,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
    gap: 2,
  },
  summaryValue: {
    fontSize: fontSize.md,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  summaryUnit: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
  },
  segmentHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.xs,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
    marginBottom: spacing.sm,
  },
  segmentHeaderText: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    textAlign: 'center',
  },
  segmentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.xs,
    gap: spacing.sm,
  },
  segmentError: {
    backgroundColor: `${colors.status.danger}10`,
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.xs,
  },
  segmentIdBadge: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  segmentIdText: {
    fontSize: fontSize.xs,
    fontWeight: 'bold',
    color: colors.text.inverse,
  },
  segmentInputGroup: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  segmentInput: {
    flex: 1,
    height: 36,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.sm,
    paddingHorizontal: spacing.sm,
    color: colors.text.primary,
    fontSize: fontSize.md,
    textAlign: 'right',
  },
  inputDisabled: {
    opacity: 0.5,
  },
  inputError: {
    borderWidth: 1,
    borderColor: colors.status.danger,
  },
  unitLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    width: 20,
  },
  monotonicNote: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
    marginTop: spacing.md,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  monotonicNoteText: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    flex: 1,
  },

  // Bottom Actions
  bottomActions: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    gap: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.background.secondary,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
  },
  optimizeBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    backgroundColor: `${colors.brand.accent}20`,
    borderWidth: 1,
    borderColor: colors.brand.accent,
  },
  optimizeBtnText: {
    fontSize: fontSize.md,
    fontWeight: 'bold',
    color: colors.brand.accent,
  },
  submitBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.brand.primary,
  },
  submitBtnText: {
    fontSize: fontSize.md,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
});
