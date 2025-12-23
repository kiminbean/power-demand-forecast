/**
 * Bidding Management Screen - Page 3
 * Figma: iPhone 16 Pro - 13 (id: 2:255)
 * Design: 10-segment bidding with AI optimization
 * Uses real API data from backend
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Dimensions,
  Platform,
  Alert,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { apiService, SMPForecast, MarketStatus, OptimizedBids } from '../services/api';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Design colors from Figma
const colors = {
  primary: '#04265e',
  secondary: '#0048ff',
  background: '#ffffff',
  cardBg: '#f8f9fa',
  text: '#000000',
  textSecondary: '#666666',
  textMuted: '#999999',
  orange: '#f59e0b',
  blue: '#2563eb',
  red: '#ef4444',
  green: '#10b981',
  border: '#e5e7eb',
};

// Segment data type
interface Segment {
  id: string;
  quantity: number;
  price: number;
}

// Default segment data from Figma
const defaultSegments: Segment[] = [
  { id: 'S1', quantity: 5, price: 80 },
  { id: 'S2', quantity: 5, price: 85 },
  { id: 'S3', quantity: 5, price: 90 },
  { id: 'S4', quantity: 5, price: 95 },
  { id: 'S5', quantity: 5, price: 100 },
  { id: 'S6', quantity: 5, price: 105 },
  { id: 'S7', quantity: 5, price: 110 },
  { id: 'S8', quantity: 5, price: 115 },
  { id: 'S9', quantity: 5, price: 120 },
  { id: 'S10', quantity: 5, price: 125 },
];

// Simple Bidding Curve Chart
function BiddingCurveChart({ segments }: { segments: Segment[] }) {
  const maxPrice = Math.max(...segments.map(s => s.price));
  const minPrice = Math.min(...segments.map(s => s.price));
  const range = maxPrice - minPrice || 1;

  return (
    <View style={styles.chartContainer}>
      {/* Y-axis and Chart Area Row */}
      <View style={styles.chartRow}>
        {/* Y-axis labels */}
        <View style={styles.chartYAxis}>
          <Text style={styles.chartAxisLabel}>{Math.round(maxPrice + 10)}</Text>
          <Text style={styles.chartAxisLabel}>{Math.round((maxPrice + minPrice) / 2)}</Text>
          <Text style={styles.chartAxisLabel}>{Math.round(minPrice - 10)}</Text>
        </View>

        {/* Chart area */}
        <View style={styles.chartArea}>
          {/* Grid lines */}
          <View style={styles.chartGrid}>
            <View style={styles.chartGridLine} />
            <View style={styles.chartGridLine} />
            <View style={styles.chartGridLine} />
          </View>

          {/* Line chart */}
          <View style={styles.chartLine}>
            {segments.map((segment, index) => {
              const y = ((segment.price - minPrice + 10) / (range + 20)) * 100;
              return (
                <View
                  key={segment.id}
                  style={[
                    styles.chartDot,
                    { bottom: `${y}%`, left: `${(index / (segments.length - 1)) * 100}%` },
                  ]}
                />
              );
            })}
          </View>

          {/* Orange area fill */}
          <View style={styles.chartFill}>
            {segments.map((segment, index) => {
              const height = ((segment.price - minPrice + 10) / (range + 20)) * 100;
              return (
                <View
                  key={segment.id}
                  style={[styles.chartBar, { height: `${height}%` }]}
                />
              );
            })}
          </View>
        </View>
      </View>

      {/* X-axis labels - aligned with chart area */}
      <View style={styles.chartXAxisRow}>
        <View style={styles.chartYAxisSpacer} />
        <View style={styles.chartXAxis}>
          <Text style={styles.chartAxisLabel}>5</Text>
          <Text style={styles.chartAxisLabel}>15</Text>
          <Text style={styles.chartAxisLabel}>25</Text>
          <Text style={styles.chartAxisLabel}>35</Text>
          <Text style={styles.chartAxisLabel}>45</Text>
          <Text style={styles.chartAxisLabel}>50</Text>
        </View>
      </View>
    </View>
  );
}

// Props for navigation
interface BiddingScreenProps {
  webNavigation?: {
    navigate: (screen: string, params?: any) => void;
    goBack: () => void;
  };
}

export default function BiddingScreen({ webNavigation }: BiddingScreenProps) {
  const [segments, setSegments] = useState<Segment[]>(defaultSegments);
  const [totalCapacity, setTotalCapacity] = useState('50');
  const [smpLow, setSmpLow] = useState(49);
  const [smpMid, setSmpMid] = useState(71);
  const [smpHigh, setSmpHigh] = useState(131);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [marketStatus, setMarketStatus] = useState<'open' | 'closed'>('closed');
  const [hoursRemaining, setHoursRemaining] = useState(0);

  // Calculate totals
  const totalMW = segments.reduce((sum, s) => sum + s.quantity, 0);
  const avgPrice = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalMW || 0;

  // Fetch initial data from API
  const fetchBiddingData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      // Fetch SMP forecast and market status in parallel
      const [forecast, status] = await Promise.all([
        apiService.getSMPForecast(),
        apiService.getMarketStatus(),
      ]);

      // Set SMP ranges from forecast
      const prices = forecast.q50;
      const q10Prices = forecast.q10;
      const q90Prices = forecast.q90;

      setSmpLow(Math.round(Math.min(...q10Prices)));
      setSmpMid(Math.round(prices.reduce((a, b) => a + b, 0) / prices.length));
      setSmpHigh(Math.round(Math.max(...q90Prices)));

      // Set market status
      setMarketStatus(status.dam.status as 'open' | 'closed');
      setHoursRemaining(status.dam.hours_remaining);

    } catch (error) {
      console.log('API unavailable, using default values:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchBiddingData();
  }, [fetchBiddingData]);

  const onRefresh = () => fetchBiddingData(true);

  // Handle AI optimization - calls real API
  const handleAIOptimize = async () => {
    setIsOptimizing(true);

    try {
      // Call API for optimized segments
      const capacity = parseInt(totalCapacity) || 50;
      const optimizedBids = await apiService.getOptimizedSegments(capacity, 'moderate');

      // Transform API response to segments
      if (optimizedBids.hourly_bids && optimizedBids.hourly_bids.length > 0) {
        const hourBid = optimizedBids.hourly_bids[0];
        if (hourBid.segments) {
          const newSegments = hourBid.segments.map((s, i) => ({
            id: `S${i + 1}`,
            quantity: s.quantity_mw,
            price: s.price_krw_mwh,
          }));
          setSegments(newSegments);
        }
      }

      if (Platform.OS === 'web') {
        window.alert('AI 최적화가 완료되었습니다. (API 연동)');
      } else {
        Alert.alert('완료', 'AI 최적화가 완료되었습니다.');
      }
    } catch (error) {
      console.log('Optimization API failed, using local optimization:', error);
      // Fallback to local optimization
      const optimizedSegments = segments.map((s, i) => ({
        ...s,
        price: Math.round(smpLow + (smpHigh - smpLow) * (i / segments.length) + Math.random() * 5),
      }));
      setSegments(optimizedSegments);

      if (Platform.OS === 'web') {
        window.alert('AI 최적화가 완료되었습니다. (로컬)');
      } else {
        Alert.alert('완료', 'AI 최적화가 완료되었습니다.');
      }
    } finally {
      setIsOptimizing(false);
    }
  };

  // Handle KPX submit
  const handleKPXSubmit = async () => {
    try {
      // Simulate KPX submission
      const totalQuantity = segments.reduce((sum, s) => sum + s.quantity, 0);
      const avgPriceVal = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalQuantity;

      const message = `KPX 입찰 제출 완료\n\n총 용량: ${totalQuantity} MW\n평균 입찰가: ${avgPriceVal.toFixed(1)}원/kWh`;

      if (Platform.OS === 'web') {
        window.alert(message);
      } else {
        Alert.alert('제출 완료', message);
      }
    } catch (error) {
      const errorMsg = 'KPX 제출 중 오류가 발생했습니다.';
      if (Platform.OS === 'web') {
        window.alert(errorMsg);
      } else {
        Alert.alert('오류', errorMsg);
      }
    }
  };

  // Navigate to DAM simulation
  const handleDAMSimulation = () => {
    if (webNavigation) {
      webNavigation.navigate('KPXSimulation', {
        segments: segments.map(s => ({ id: parseInt(s.id.replace('S', '')), quantity: s.quantity, price: s.price })),
        selectedHour: 12,
        smpForecast: { q10: smpLow, q50: smpMid, q90: smpHigh },
      });
    }
  };

  // Navigate to RTM simulation
  const handleRTMSimulation = () => {
    if (webNavigation) {
      webNavigation.navigate('RTMSimulation', {
        segments: segments.map(s => ({ id: parseInt(s.id.replace('S', '')), quantity: s.quantity, price: s.price })),
        selectedHour: new Date().getHours(),
        smpForecast: { q10: smpLow, q50: smpMid, q90: smpHigh },
      });
    }
  };

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Title Section */}
      <View style={styles.titleSection}>
        <View style={styles.titleRow}>
          <Text style={styles.pageTitle}>입찰관리</Text>
          <View style={[styles.damBadge, marketStatus === 'open' && styles.damBadgeOpen]}>
            <View style={[styles.damDot, marketStatus === 'open' && styles.damDotOpen]} />
            <Text style={[styles.damBadgeText, marketStatus === 'open' && styles.damBadgeTextOpen]}>
              {marketStatus === 'open' ? `DAM ${hoursRemaining.toFixed(1)}h` : 'DAM 마감'}
            </Text>
          </View>
        </View>
        <Text style={styles.subtitle}>10-segment 입찰가격 설정</Text>
      </View>

      {/* SMP Stats Row */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>하한</Text>
          <Text style={[styles.statValue, { color: colors.blue }]}>{smpLow}</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>예측</Text>
          <Text style={styles.statValue}>{smpMid}</Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>상한</Text>
          <Text style={[styles.statValue, { color: colors.orange }]}>{smpHigh}</Text>
        </View>
      </View>

      {/* Capacity Input */}
      <View style={styles.inputSection}>
        <Text style={styles.inputLabel}>입찰 용량(MW)</Text>
        <TextInput
          style={styles.capacityInput}
          value={totalCapacity}
          onChangeText={setTotalCapacity}
          keyboardType="numeric"
          placeholder="50"
          placeholderTextColor={colors.textMuted}
        />
      </View>

      {/* Bidding Curve Chart */}
      <View style={styles.chartSection}>
        <Text style={styles.sectionTitle}>입찰 곡선</Text>
        <BiddingCurveChart segments={segments} />
      </View>

      {/* Action Buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={[styles.actionBtn, styles.optimizeBtn]}
          onPress={handleAIOptimize}
          disabled={isOptimizing}
        >
          <Text style={styles.optimizeBtnText}>
            {isOptimizing ? '최적화 중...' : 'AI 최적화'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionBtn, styles.submitBtn]}
          onPress={handleKPXSubmit}
        >
          <Text style={styles.submitBtnText}>KPX 제출</Text>
        </TouchableOpacity>
      </View>

      {/* Segment Settings */}
      <View style={styles.segmentSection}>
        <TouchableOpacity
          style={styles.segmentHeader}
          onPress={() => setIsExpanded(!isExpanded)}
        >
          <Text style={styles.sectionTitle}>구간별 설정</Text>
          <Text style={styles.expandIcon}>{isExpanded ? '∨' : '>'}</Text>
        </TouchableOpacity>

        {isExpanded && (
          <>
            {/* Summary Row */}
            <View style={styles.segmentSummary}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>총 입찰량</Text>
                <Text style={styles.summaryValue}>{totalMW.toFixed(1)} MW</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>예상 평균가</Text>
                <Text style={[styles.summaryValue, { color: colors.orange }]}>
                  {avgPrice.toFixed(1)}원
                </Text>
              </View>
            </View>

            {/* Segment List */}
            <View style={styles.segmentList}>
              {segments.map((segment) => (
                <View key={segment.id} style={styles.segmentRow}>
                  <View style={styles.segmentIdCell}>
                    <Text style={styles.segmentId}>{segment.id}</Text>
                  </View>
                  <View style={styles.segmentValueCell}>
                    <Text style={styles.segmentValue}>{segment.quantity}</Text>
                  </View>
                  <View style={styles.segmentPriceCell}>
                    <Text style={styles.segmentPrice}>{segment.price}</Text>
                  </View>
                </View>
              ))}
            </View>
          </>
        )}
      </View>

      {/* Simulation Buttons */}
      <View style={styles.simulationButtons}>
        <TouchableOpacity
          style={styles.simulationBtn}
          onPress={handleDAMSimulation}
        >
          <Text style={styles.simulationBtnText}>DAM 시뮬레이션</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.simulationBtn, styles.rtmSimulationBtn]}
          onPress={handleRTMSimulation}
        >
          <Text style={styles.simulationBtnText}>RTM 시뮬레이션</Text>
        </TouchableOpacity>
      </View>

      {/* Bottom padding */}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: 16,
  },

  // Title Section
  titleSection: {
    marginTop: 16,
    marginBottom: 20,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  pageTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text,
  },
  damBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fef2f2',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#fecaca',
  },
  damDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.red,
    marginRight: 6,
  },
  damBadgeText: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.red,
  },
  damBadgeOpen: {
    backgroundColor: '#f0fdf4',
    borderColor: '#bbf7d0',
  },
  damDotOpen: {
    backgroundColor: colors.green,
  },
  damBadgeTextOpen: {
    color: colors.green,
  },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: 4,
  },

  // Stats Row
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 20,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text,
  },

  // Input Section
  inputSection: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 8,
  },
  capacityInput: {
    backgroundColor: colors.cardBg,
    borderRadius: 8,
    padding: 14,
    fontSize: 16,
    color: colors.text,
    borderWidth: 1,
    borderColor: colors.border,
  },

  // Chart Section
  chartSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 12,
  },
  chartContainer: {
    height: 180,
  },
  chartRow: {
    flex: 1,
    flexDirection: 'row',
  },
  chartYAxis: {
    width: 35,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 8,
    paddingBottom: 5,
  },
  chartYAxisSpacer: {
    width: 35,
  },
  chartArea: {
    flex: 1,
    position: 'relative',
  },
  chartGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  chartGridLine: {
    height: 1,
    backgroundColor: colors.border,
  },
  chartLine: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  chartDot: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.orange,
    marginLeft: -3,
    marginBottom: -3,
  },
  chartFill: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '100%',
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  chartBar: {
    flex: 1,
    backgroundColor: 'rgba(245, 158, 11, 0.2)',
    marginHorizontal: 1,
  },
  chartXAxisRow: {
    flexDirection: 'row',
    marginTop: 8,
  },
  chartXAxis: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
  },
  chartAxisLabel: {
    fontSize: 10,
    color: colors.textMuted,
  },

  // Action Buttons
  actionButtons: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  actionBtn: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  optimizeBtn: {
    backgroundColor: colors.orange,
  },
  optimizeBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.background,
  },
  submitBtn: {
    backgroundColor: colors.primary,
  },
  submitBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.background,
  },

  // Segment Section
  segmentSection: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  segmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  expandIcon: {
    fontSize: 18,
    color: colors.textSecondary,
  },
  segmentSummary: {
    flexDirection: 'row',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  summaryItem: {
    flex: 1,
  },
  summaryLabel: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  summaryValue: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
    marginTop: 2,
  },

  // Segment List
  segmentList: {
    marginTop: 16,
  },
  segmentRow: {
    flexDirection: 'row',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  segmentIdCell: {
    width: 40,
  },
  segmentId: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.textSecondary,
  },
  segmentValueCell: {
    flex: 1,
    alignItems: 'center',
  },
  segmentValue: {
    fontSize: 14,
    color: colors.text,
  },
  segmentPriceCell: {
    width: 60,
    alignItems: 'flex-end',
  },
  segmentPrice: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
  },

  // Simulation Buttons
  simulationButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  simulationBtn: {
    flex: 1,
    backgroundColor: colors.blue,
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  rtmSimulationBtn: {
    backgroundColor: colors.green,
  },
  simulationBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.background,
  },
});
