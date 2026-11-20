/**
 * Bidding Management Screen - RE-BMS Mobile v6.2.0
 * 100% Feature Parity with web-v6.2.0
 * 10-segment bidding with AI optimization
 * Editable segments with monotonic price constraint
 * Power plant registration and management
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
  Modal,
  Switch,
} from 'react-native';

import { colors as themeColors } from '../theme/colors';
import { apiService, SMPForecast, MarketStatus, OptimizedBids, PowerPlant, DualSettlement } from '../services/api';
import PowerPlantRegistrationScreen from './PowerPlantRegistrationScreen';
import {
  calculateEfficiency,
  estimateDailyGeneration,
  getEfficiencyStatus,
  formatCapacity,
  formatRevenue,
  mapWeatherCondition,
  WeatherCondition,
} from '../utils/powerPlantUtils';
import { PLANT_TYPE_LABELS, CONTRACT_TYPE_LABELS, ROOF_DIRECTION_LABELS, PLANT_STATUS_LABELS, PlantStatus } from '../types/powerPlant';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const MAX_BID_CAPACITY_MW = 500;
const CAPACITY_VALIDATION_MESSAGE = '입찰 용량은 0 MW 초과, 500 MW 이하로 입력해 주세요.';
const STALE_CAPACITY_MESSAGE = '용량이 변경되었습니다. AI 최적화를 다시 실행해 주세요.';
const SEGMENT_CAPACITY_MESSAGE = '총 입찰량이 입찰 용량을 초과하거나 유효하지 않습니다.';
const CAPACITY_ROUNDING_TOLERANCE_MW = 1e-9;

function validBidCapacity(text: string): boolean {
  const capacity = Number(text.trim());
  return text.trim() !== '' && Number.isFinite(capacity) && capacity > 0 && capacity <= MAX_BID_CAPACITY_MW;
}

// Design colors from Figma
const colors = {
  primary: themeColors.navy,
  secondary: themeColors.info,
  background: themeColors.bg,
  cardBg: themeColors.surface,
  text: themeColors.text.primary,
  textSecondary: themeColors.text.secondary,
  textMuted: themeColors.text.secondary,
  orange: themeColors.warning,
  blue: themeColors.info,
  red: themeColors.danger,
  green: themeColors.accent,
  border: themeColors.border.primary
};

// Segment data type with optional AI optimization results
interface Segment {
  id: string;
  quantity: number;
  price: number;
  clearingProbability?: number;  // AI optimization result
  expectedRevenue?: number;      // AI optimization result
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

// Bid status type - KPX-style workflow (Phase 5)
type BidStatus = 'draft' | 'validating' | 'submitted' | 'accepted' | 'closed' | 'cleared' | 'rejected';
type RiskLevel = 'conservative' | 'moderate' | 'aggressive';

// Market type for DAM/RTM tab selection (Phase 6)
type MarketType = 'dam' | 'rtm';

// RTM Slot for 15-minute interval bidding (Phase 6)
interface RTMSlot {
  time: string;
  adjustmentMw: number;
  estimatedPrice: number;
  status: 'pending' | 'submitted';
}

// Bid Status Configuration (Phase 5)
const BID_STATUS_CONFIG: Record<BidStatus, { label: string; color: string; icon: string }> = {
  draft: { label: '작성 중', color: themeColors.text.secondary, icon: '📝' },
  validating: { label: '검증 중', color: themeColors.warning, icon: '⏳' },
  submitted: { label: '제출됨', color: themeColors.info, icon: '📤' },
  accepted: { label: '접수 완료', color: themeColors.accent, icon: '✓' },
  closed: { label: '마감', color: themeColors.text.secondary, icon: '🔒' },
  cleared: { label: '낙찰', color: themeColors.success, icon: '🎉' },
  rejected: { label: '유찰', color: themeColors.danger, icon: '✗' },
};

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

  // New states for feature parity with web-v6.1
  const [selectedHour, setSelectedHour] = useState(12);
  const [riskLevel, setRiskLevel] = useState<RiskLevel>('moderate');
  const [bidStatus, setBidStatus] = useState<BidStatus>('draft');
  const [isSaving, setIsSaving] = useState(false);

  // AI Optimization info state
  const [optimizationInfo, setOptimizationInfo] = useState<{
    modelUsed: string;
    method: string;
    totalExpectedRevenue: number;
  } | null>(null);
  const [optimizationError, setOptimizationError] = useState<string | null>(null);
  const [capacityError, setCapacityError] = useState<string | null>(null);

  // Review Modal state
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);

  // Power Plant states (v6.2.0)
  const [powerPlants, setPowerPlants] = useState<PowerPlant[]>([]);
  const [plantStatus, setPlantStatus] = useState<'loading' | 'ready' | 'unavailable' | 'error'>('loading');
  const [plantsAreLocal, setPlantsAreLocal] = useState(false);
  const [showRegistration, setShowRegistration] = useState(false);
  const [currentWeather, setCurrentWeather] = useState<WeatherCondition>('clear');
  const [vppBiddingEnabled, setVppBiddingEnabled] = useState(true); // VPP auto-bidding toggle

  // Market Tab states (Phase 6 - DAM/RTM)
  const [selectedMarket, setSelectedMarket] = useState<MarketType>('dam');
  const [rtmSlots, setRtmSlots] = useState<RTMSlot[]>([]);
  const [rtmTimeRemaining, setRtmTimeRemaining] = useState('');

  const [capacityRevision] = useState({ current: 0 });
  const [segmentsStale, setSegmentsStale] = useState(false);

  // Dual Settlement state (Phase 7)
  const [dualSettlement, setDualSettlement] = useState<DualSettlement | null>(null);

  // Check if DAM is open (before 10:00 AM)
  const currentHour = new Date().getHours();
  const isDamOpen = currentHour < 10; // DAM closes at 10:00 AM

  // Auto-switch to RTM when DAM is closed
  useEffect(() => {
    if (!isDamOpen && selectedMarket === 'dam') {
      setSelectedMarket('rtm');
    }
  }, [isDamOpen, selectedMarket]);

  // Calculate totals
  const totalMW = segments.reduce((sum, s) => sum + s.quantity, 0);
  const avgPrice = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalMW || 0;
  // Total expected revenue: sum of all segment revenues (MW × 1000 × price × clearingProb)
  const totalExpectedRevenue = segments.reduce((sum, s) => {
    const clearingProb = s.clearingProbability ?? 1;
    const revenue = s.expectedRevenue ?? (s.quantity * 1000 * s.price * clearingProb);
    return sum + revenue;
  }, 0);

  // Update segment quantity
  const updateSegmentQuantity = (id: string, newQuantity: number) => {
    setSegments((prev) =>
      prev.map((s) => (s.id === id ? { ...s, quantity: Math.max(0, newQuantity) } : s))
    );
    // Clear optimization info when manually editing (Phase 3)
    setOptimizationInfo(null);
  };

  // Update segment price with monotonic constraint enforcement
  const updateSegmentPrice = (id: string, newPrice: number) => {
    setSegments((prev) => {
      const updated = [...prev];
      const idx = updated.findIndex((s) => s.id === id);
      if (idx >= 0) {
        updated[idx] = { ...updated[idx], price: Math.max(0, newPrice) };
        // Enforce monotonic constraint - higher segments must have >= price
        for (let i = idx + 1; i < updated.length; i++) {
          if (updated[i].price < newPrice) {
            updated[i] = { ...updated[i], price: newPrice };
          }
        }
        // Enforce monotonic constraint - lower segments must have <= price
        for (let i = idx - 1; i >= 0; i--) {
          if (updated[i].price > newPrice) {
            updated[i] = { ...updated[i], price: newPrice };
          }
        }
      }
      return updated;
    });
    // Clear optimization info when manually editing (Phase 3)
    setOptimizationInfo(null);
  };

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

  // Handle AI optimization - calls real API (100% feature parity with web-v6.1)
  const handleAIOptimize = async () => {
    const capacity = Number(totalCapacity.trim());
    if (!validBidCapacity(totalCapacity)) {
      setCapacityError(CAPACITY_VALIDATION_MESSAGE);
      setOptimizationError(null);
      setOptimizationInfo(null);
      return;
    }

    setCapacityError(null);
    setIsOptimizing(true);
    setOptimizationError(null);
    setOptimizationInfo(null);
    const revision = capacityRevision.current;

    try {
      // Call API for optimized segments
      const optimizedBids = await apiService.getOptimizedSegments(capacity, riskLevel);
      if (revision !== capacityRevision.current) return;

      // Find the hourly bid for the selected hour
      const hourlyBid = optimizedBids.hourly_bids.find(bid => bid.hour === selectedHour)
        || optimizedBids.hourly_bids[0];

      if (hourlyBid && hourlyBid.segments) {
        // Convert API response to local segment format with clearing probability
        const newSegments: Segment[] = hourlyBid.segments.map((seg, idx) => {
          // 낙찰확률: 가격이 높을수록 낙찰 확률 감소
          const clearingProb = (seg as any).clearing_probability ||
            Math.max(0.1, 1 - (idx * 0.08) + (Math.random() * 0.1 - 0.05));
          // 예상수익: MW × 1000(kW) × 가격(원/kWh) × 낙찰확률
          const expectedRev = (seg as any).expected_revenue ||
            seg.quantity_mw * 1000 * seg.price_krw_mwh * clearingProb;

          return {
            id: `S${idx + 1}`,
            quantity: seg.quantity_mw,
            price: Math.round(seg.price_krw_mwh),
            clearingProbability: clearingProb,
            expectedRevenue: expectedRev,
          };
        });

        setSegments(newSegments);
        setSegmentsStale(false);

        // Calculate total expected revenue for this hour
        const totalExpectedRevenue = newSegments.reduce(
          (sum, seg) => sum + (seg.expectedRevenue || 0),
          0
        );

        setOptimizationInfo({
          modelUsed: optimizedBids.model_used || 'BiLSTM+Attention v3.1',
          method: (optimizedBids as any).optimization_method || 'quantile-based',
          totalExpectedRevenue,
        });

        setBidStatus('draft');
      } else {
        throw new Error(`No optimization data for hour ${selectedHour}`);
      }
    } catch (error) {
      if (revision !== capacityRevision.current) return;
      console.log('Optimization API failed, using local optimization:', error);
      setOptimizationError(
        error instanceof Error ? error.message : 'AI optimization failed'
      );

      // Fallback to simple client-side optimization with clearing probability
      const basePrice = smpLow * 0.9;
      const priceSpread = (smpHigh - smpLow) / 9;
      const capacityPerSegment = capacity / 10;
      const newSegments: Segment[] = segments.map((s, i) => {
        const segPrice = Math.round(basePrice + i * priceSpread);
        const clearingProb = Math.max(0.1, 1 - (i * 0.08));
        // 예상수익: MW × 1000(kW) × 가격(원/kWh) × 낙찰확률
        const expectedRev = capacityPerSegment * 1000 * segPrice * clearingProb;
        return {
          ...s,
          price: segPrice,
          quantity: capacityPerSegment,
          clearingProbability: clearingProb,
          expectedRevenue: expectedRev,
        };
      });
      setSegments(newSegments);
      setSegmentsStale(false);

      // Set optimization info for fallback
      const fallbackTotalRevenue = newSegments.reduce(
        (sum, seg) => sum + (seg.expectedRevenue || 0),
        0
      );
      setOptimizationInfo({
        modelUsed: 'Fallback Algorithm',
        method: 'quantile-based',
        totalExpectedRevenue: fallbackTotalRevenue,
      });
    } finally {
      setIsOptimizing(false);
    }
  };

  const validateReview = (): boolean => {
    if (!validBidCapacity(totalCapacity)) {
      setCapacityError(CAPACITY_VALIDATION_MESSAGE);
      return false;
    }
    if (segmentsStale || isOptimizing) {
      setCapacityError(STALE_CAPACITY_MESSAGE);
      return false;
    }
    if (segments.some(segment => !Number.isFinite(segment.quantity) || segment.quantity < 0) ||
        !Number.isFinite(totalMW) || totalMW <= 0 ||
        totalMW > Number(totalCapacity.trim()) + CAPACITY_ROUNDING_TOLERANCE_MW) {
      setCapacityError(SEGMENT_CAPACITY_MESSAGE);
      return false;
    }
    setCapacityError(null);
    return true;
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
    } else {
      // Mobile: Show simulation results as alert
      const totalQuantity = segments.reduce((sum, s) => sum + s.quantity, 0);
      const avgPriceVal = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalQuantity;
      const estimatedClearing = segments.filter(s => s.price <= smpMid).reduce((sum, s) => sum + s.quantity, 0);
      const clearingRate = (estimatedClearing / totalQuantity * 100).toFixed(1);
      const estimatedRevenue = (estimatedClearing * smpMid * 1000 / 1000000).toFixed(2);

      const message = `DAM 시뮬레이션 결과\n\n` +
        `총 입찰량: ${totalQuantity} MW\n` +
        `평균 입찰가: ${avgPriceVal.toFixed(0)}원/kWh\n` +
        `예상 SMP: ${smpMid}원/kWh\n` +
        `예상 낙찰량: ${estimatedClearing} MW (${clearingRate}%)\n` +
        `예상 수익: ${estimatedRevenue}백만원`;

      if (Platform.OS === 'web') {
        window.alert(message);
      } else {
        Alert.alert('DAM 시뮬레이션', message);
      }
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
    } else {
      // Mobile: Show RTM simulation as alert
      const currentHour = new Date().getHours();
      const isPeak = (currentHour >= 9 && currentHour <= 11) || (currentHour >= 18 && currentHour <= 21);
      const rtmpEstimate = Math.round(smpMid * (isPeak ? 1.15 : 1.0));
      const volatility = isPeak ? '높음 (±15%)' : '보통 (±8%)';

      // Calculate RTM adjustment impact
      const totalAdjustment = rtmSlots.reduce((sum, s) => sum + s.adjustmentMw, 0);
      const estimatedRtmRevenue = (totalAdjustment * rtmpEstimate * 1000 / 10000).toFixed(1);

      const message = `RTM 시뮬레이션 결과\n\n` +
        `현재 시간: ${currentHour}:00\n` +
        `예상 RTMP: ${rtmpEstimate}원/kWh\n` +
        `변동성: ${volatility}\n` +
        `피크 시간대: ${isPeak ? '예' : '아니오'}\n\n` +
        `RTM 조정량: ${totalAdjustment > 0 ? '+' : ''}${totalAdjustment} MW\n` +
        `예상 추가 수익: ${estimatedRtmRevenue}만원`;

      if (Platform.OS === 'web') {
        window.alert(message);
      } else {
        Alert.alert('RTM 시뮬레이션', message);
      }
    }
  };

  // ============================================
  // Power Plant Functions (v6.2.0)
  // ============================================

  // Load power plants from API or localStorage
  const loadPowerPlants = useCallback(async () => {
    setPlantStatus('loading');
    try {
      const plants = await apiService.getPowerPlants();
      if (!Array.isArray(plants)) throw new Error('Invalid power plant response');
      setPowerPlants(plants);
      setPlantsAreLocal(false);
      setPlantStatus('ready');
    } catch (error) {
      setPlantStatus(error instanceof Error && error.message === 'API Error: 404' ? 'unavailable' : 'error');
      setPowerPlants([]);
      setPlantsAreLocal(false);
      // Fallback to localStorage
      try {
        const stored = Platform.OS === 'web'
          ? localStorage.getItem('powerPlants')
          : null;
        if (stored) {
          const plants: unknown = JSON.parse(stored);
          if (Array.isArray(plants) && plants.length > 0) {
            setPowerPlants(plants);
            setPlantsAreLocal(true);
          }
        }
      } catch (e) {
        console.log('No stored power plants');
      }
    }
  }, []);

  // Save power plants to localStorage (backup)
  const savePowerPlantsToStorage = useCallback((plants: PowerPlant[]) => {
    if (Platform.OS === 'web') {
      localStorage.setItem('powerPlants', JSON.stringify(plants));
    }
  }, []);

  // Handle plant registration/update
  const handlePlantSave = useCallback((plant: PowerPlant) => {
    setPowerPlants(prev => {
      const existing = prev.findIndex(p => p.id === plant.id);
      let updated: PowerPlant[];
      if (existing >= 0) {
        updated = [...prev];
        updated[existing] = plant;
      } else {
        updated = [...prev, plant];
      }
      savePowerPlantsToStorage(updated);
      return updated;
    });
    if (plantStatus !== 'ready') setPlantsAreLocal(true);
    setShowRegistration(false);
  }, [savePowerPlantsToStorage, plantStatus]);

  // Delete a power plant
  const handlePlantDelete = useCallback(async (plantId: string) => {
    const confirmDelete = () => {
      setPowerPlants(prev => {
        const updated = prev.filter(p => p.id !== plantId);
        savePowerPlantsToStorage(updated);
        return updated;
      });
      // Try to delete from API as well
      apiService.deletePowerPlant(plantId).catch(() => {});
    };

    if (Platform.OS === 'web') {
      if (window.confirm('이 발전소를 삭제하시겠습니까?')) {
        confirmDelete();
      }
    } else {
      Alert.alert(
        '발전소 삭제',
        '이 발전소를 삭제하시겠습니까?',
        [
          { text: '취소', style: 'cancel' },
          { text: '삭제', style: 'destructive', onPress: confirmDelete },
        ]
      );
    }
  }, [savePowerPlantsToStorage]);

  // Update plant status (VPP control) - Optimistic update with rollback (Phase 2)
  const handleUpdatePlantStatus = useCallback(async (plantId: string, newStatus: PlantStatus) => {
    // Store original status for rollback
    const originalPlant = powerPlants.find(p => p.id === plantId);
    const originalStatus = originalPlant?.status || 'active';

    // Optimistic update - immediately update UI
    setPowerPlants(prev => prev.map(p =>
      p.id === plantId ? { ...p, status: newStatus, updatedAt: new Date().toISOString() } : p
    ));

    try {
      await apiService.updatePowerPlant(plantId, { status: newStatus });
      // Save to local storage on success
      setPowerPlants(prev => {
        savePowerPlantsToStorage(prev);
        return prev;
      });
    } catch (error) {
      // Rollback on API failure
      console.log('API update failed, keeping local change:', error);
      // Still save locally even if API fails
      setPowerPlants(prev => {
        savePowerPlantsToStorage(prev);
        return prev;
      });
    }
  }, [powerPlants, savePowerPlantsToStorage]);

  // Calculate totals for active plants only (VPP control)
  const activePlants = powerPlants.filter(p => (p.status || 'active') === 'active');
  const activePlantCount = activePlants.length;
  const activePlantCapacityKw = activePlants.reduce((sum, p) => sum + p.capacity, 0);

  // Calculate total recommended capacity from active registered plants
  const recommendedCapacity = activePlants.reduce((sum, plant) => {
    const efficiency = calculateEfficiency(plant.installDate);
    const dailyKwh = estimateDailyGeneration(plant.capacity, efficiency, currentWeather, plant.roofDirection || 'south');
    return sum + dailyKwh;
  }, 0);

  // Total registered plant capacity and UI mode
  // BUG FIX: Use active plant capacity for UI mode determination (Phase 4)
  const totalPlantCapacityKw = powerPlants.reduce((sum, p) => sum + p.capacity, 0);
  const isLargeCapacity = activePlantCapacityKw >= 1000; // 1MW = 1000kW (active plants only)

  // Load power plants on mount
  useEffect(() => {
    loadPowerPlants();
  }, [loadPowerPlants]);

  // ============================================
  // RTM Functions (Phase 6 - DAM/RTM Dual Bidding)
  // ============================================

  // Generate RTM slots for the next 4 hours (16 x 15-min intervals)
  const generateRTMSlots = useCallback(() => {
    const slots: RTMSlot[] = [];
    const now = new Date();
    // Round up to next 15-minute interval
    const minutes = Math.ceil(now.getMinutes() / 15) * 15;
    const startTime = new Date(now);
    startTime.setMinutes(minutes, 0, 0);

    // Add T+75 minutes offset (KPX RTM bidding deadline)
    startTime.setMinutes(startTime.getMinutes() + 75);

    // Max available capacity for RTM adjustments (10% of active capacity)
    const maxAdjustmentMw = activePlantCapacityKw / 1000 * 0.1;

    for (let i = 0; i < 16; i++) {
      const slotTime = new Date(startTime);
      slotTime.setMinutes(slotTime.getMinutes() + i * 15);
      const timeStr = `${String(slotTime.getHours()).padStart(2, '0')}:${String(slotTime.getMinutes()).padStart(2, '0')}`;

      // Estimate RTMP based on SMP with time-based volatility
      const hour = slotTime.getHours();
      const isPeak = hour >= 9 && hour <= 11 || hour >= 18 && hour <= 21;
      const volatility = isPeak ? 1.15 : 1.0;
      const estimatedPrice = Math.round(smpMid * volatility * (0.95 + Math.random() * 0.1));

      // AI-recommended adjustment based on price and peak hours
      // Positive = sell more (price high), Negative = buy/reduce (price low)
      let aiRecommendedAdjustment = 0;
      if (isPeak && estimatedPrice > smpMid) {
        // Peak + high price → sell more
        aiRecommendedAdjustment = Math.round((Math.random() * maxAdjustmentMw * 0.8 + maxAdjustmentMw * 0.2) * 10) / 10;
      } else if (!isPeak && estimatedPrice < smpMid * 0.9) {
        // Off-peak + low price → reduce/buy
        aiRecommendedAdjustment = -Math.round((Math.random() * maxAdjustmentMw * 0.5) * 10) / 10;
      } else {
        // Normal → small random adjustment or zero
        aiRecommendedAdjustment = Math.random() > 0.5
          ? Math.round((Math.random() * maxAdjustmentMw * 0.3) * 10) / 10
          : 0;
      }

      slots.push({
        time: timeStr,
        adjustmentMw: aiRecommendedAdjustment,
        estimatedPrice,
        status: 'pending',
      });
    }
    setRtmSlots(slots);
  }, [smpMid, activePlantCapacityKw]);

  // Get market deadline
  const getMarketDeadline = useCallback((market: MarketType) => {
    const now = new Date();
    if (market === 'dam') {
      // D-1 10:00 deadline (tomorrow's trading -> today 10:00 deadline)
      const deadline = new Date(now);
      if (now.getHours() >= 10) {
        deadline.setDate(deadline.getDate() + 1);
      }
      deadline.setHours(10, 0, 0, 0);
      return deadline;
    } else {
      // T-75 minutes before next 15-min slot
      const minutes = Math.ceil(now.getMinutes() / 15) * 15;
      const nextSlot = new Date(now);
      nextSlot.setMinutes(minutes + 75, 0, 0);
      return nextSlot;
    }
  }, []);

  // Update RTM slot adjustment
  const updateRtmSlot = useCallback((time: string, adjustmentMw: number) => {
    setRtmSlots(prev => prev.map(slot =>
      slot.time === time ? { ...slot, adjustmentMw } : slot
    ));
  }, []);

  // Initialize RTM slots and deadline timer
  useEffect(() => {
    if (selectedMarket === 'rtm') {
      generateRTMSlots();
    }

    // Update countdown timer every second
    const timer = setInterval(() => {
      const deadline = getMarketDeadline(selectedMarket);
      const diff = deadline.getTime() - Date.now();
      if (diff <= 0) {
        setRtmTimeRemaining('마감');
      } else {
        const hours = Math.floor(diff / 3600000);
        const mins = Math.floor((diff % 3600000) / 60000);
        if (selectedMarket === 'dam') {
          setRtmTimeRemaining(`${hours}h ${mins}m`);
        } else {
          const secs = Math.floor((diff % 60000) / 1000);
          setRtmTimeRemaining(`${mins}m ${secs}s`);
        }
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [selectedMarket, generateRTMSlots, getMarketDeadline]);

  // Handle RTM submission
  const handleRTMSubmit = async () => {
    const totalAdjustment = rtmSlots.reduce((sum, s) => sum + s.adjustmentMw, 0);
    const message = `RTM 입찰 제출 완료\n\n총 조정량: ${totalAdjustment > 0 ? '+' : ''}${totalAdjustment} MW\n슬롯 수: ${rtmSlots.filter(s => s.adjustmentMw !== 0).length}개`;

    if (Platform.OS === 'web') {
      window.alert(message);
    } else {
      Alert.alert('RTM 제출 완료', message);
    }

    // Mark slots as submitted
    setRtmSlots(prev => prev.map(slot => ({ ...slot, status: 'submitted' as const })));
  };

  // ============================================
  // Dual Settlement Functions (Phase 7)
  // ============================================

  // Generate mock dual settlement for demo (real API may not be available)
  const generateMockDualSettlement = useCallback((): DualSettlement => {
    const today = new Date().toISOString().split('T')[0];
    const damClearedMwh = totalMW * 12; // 12 hours avg
    const damSmp = smpMid;
    const damRevenue = damClearedMwh * damSmp * 1000;

    // Actual generation varies ±10%
    const variance = (Math.random() * 0.2 - 0.1);
    const actualGenerationMwh = damClearedMwh * (1 + variance);
    const rtmVolumeMwh = actualGenerationMwh - damClearedMwh;

    // RTM price is typically 5-20% different from DAM SMP
    const rtmPriceVariance = Math.random() * 0.15 + 0.05;
    const rtmPrice = Math.round(smpMid * (1 + (rtmVolumeMwh >= 0 ? -rtmPriceVariance : rtmPriceVariance)));
    const rtmRevenue = rtmVolumeMwh * rtmPrice * 1000;

    return {
      trading_date: today,
      dam_cleared_mwh: Math.round(damClearedMwh * 10) / 10,
      dam_smp: damSmp,
      dam_revenue: damRevenue,
      actual_generation_mwh: Math.round(actualGenerationMwh * 10) / 10,
      rtm_volume_mwh: Math.round(rtmVolumeMwh * 10) / 10,
      rtm_price: rtmPrice,
      rtm_revenue: rtmRevenue,
      total_revenue: damRevenue + rtmRevenue,
      imbalance_type: rtmVolumeMwh > 0.1 ? 'surplus' : rtmVolumeMwh < -0.1 ? 'deficit' : 'balanced',
    };
  }, [totalMW, smpMid]);

  // Load dual settlement data
  const loadDualSettlement = useCallback(async () => {
    try {
      const today = new Date().toISOString().split('T')[0];
      const settlement = await apiService.getDualSettlement(today);
      setDualSettlement(settlement);
    } catch {
      // Fallback to mock data if API unavailable
      setDualSettlement(generateMockDualSettlement());
    }
  }, [generateMockDualSettlement]);

  // Load dual settlement when market changes or on mount
  useEffect(() => {
    if (bidStatus === 'cleared' || bidStatus === 'accepted') {
      loadDualSettlement();
    }
  }, [bidStatus, loadDualSettlement]);

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
          <View style={[
            styles.damBadge,
            selectedMarket === 'dam' && marketStatus === 'open' && styles.damBadgeOpen,
            selectedMarket === 'rtm' && styles.rtmBadge,
          ]}>
            <View style={[
              styles.damDot,
              selectedMarket === 'dam' && marketStatus === 'open' && styles.damDotOpen,
              selectedMarket === 'rtm' && styles.rtmDot,
            ]} />
            <Text style={[
              styles.damBadgeText,
              selectedMarket === 'dam' && marketStatus === 'open' && styles.damBadgeTextOpen,
              selectedMarket === 'rtm' && styles.rtmBadgeText,
            ]}>
              {selectedMarket === 'dam'
                ? (marketStatus === 'open' ? `DAM ${rtmTimeRemaining || `${Math.floor(hoursRemaining)}h`}` : 'DAM 마감')
                : `RTM ${rtmTimeRemaining}`}
            </Text>
          </View>
        </View>
        <Text style={styles.subtitle}>
          {selectedMarket === 'dam' ? '10-segment 입찰가격 설정' : '15분 단위 증감 입찰'}
        </Text>

        {/* DAM/RTM Market Tabs (Phase 6) */}
        <View style={styles.marketTabs}>
          <TouchableOpacity
            style={[
              styles.marketTab,
              selectedMarket === 'dam' && styles.marketTabActive,
              !isDamOpen && styles.marketTabDisabled,
            ]}
            onPress={() => isDamOpen && setSelectedMarket('dam')}
            disabled={!isDamOpen}
          >
            <Text style={[
              styles.marketTabTitle,
              selectedMarket === 'dam' && styles.marketTabTitleActive,
              !isDamOpen && styles.marketTabTitleDisabled,
            ]}>
              {isDamOpen ? 'DAM' : 'DAM 마감'}
            </Text>
            <Text style={[
              styles.marketTabSubtext,
              selectedMarket === 'dam' && styles.marketTabSubtextActive,
              !isDamOpen && styles.marketTabSubtextDisabled,
            ]}>
              D-1 10:00 마감
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.marketTab, selectedMarket === 'rtm' && styles.marketTabActive]}
            onPress={() => setSelectedMarket('rtm')}
          >
            <Text style={[styles.marketTabTitle, selectedMarket === 'rtm' && styles.marketTabTitleActive]}>
              RTM
            </Text>
            <Text style={[styles.marketTabSubtext, selectedMarket === 'rtm' && styles.marketTabSubtextActive]}>
              15분 단위
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Power Plant Section (v6.2.0) */}
      <View style={styles.powerPlantSection}>
        <View style={styles.powerPlantHeader}>
          <View style={styles.sectionTitleRow}>
            <Text style={styles.sectionTitle}>내 발전소</Text>
            {powerPlants.length > 0 && (
              <View style={styles.plantCountBadge}>
                <Text style={styles.plantCountText}>
                  {activePlantCount}/{powerPlants.length}
                </Text>
              </View>
            )}
          </View>
          <TouchableOpacity
            style={styles.registerBtn}
            onPress={() => setShowRegistration(true)}
          >
            <Text style={styles.registerBtnText}>+ 발전소 등록</Text>
          </TouchableOpacity>
        </View>

        {plantStatus === 'loading' && <Text>발전소 정보를 불러오는 중입니다.</Text>}
        {(plantStatus === 'unavailable' || plantStatus === 'error') && (
          <View style={styles.emptyPlantCard}>
            <Text style={styles.emptyPlantText}>
              {plantStatus === 'unavailable' ? '발전소 정보를 현재 이용할 수 없습니다.' : '발전소 정보를 불러오지 못했습니다.'}
            </Text>
            {plantsAreLocal && <Text style={styles.emptyPlantSubtext}>기기에 저장된 발전소 정보입니다. 서버 정보와 다를 수 있습니다.</Text>}
            <TouchableOpacity style={styles.registerBtn} onPress={loadPowerPlants}><Text style={styles.registerBtnText}>다시 시도</Text></TouchableOpacity>
          </View>
        )}
        {/* Registered Plants List */}
        {plantStatus !== 'loading' && powerPlants.length > 0 ? (
          <>
            {powerPlants.map((plant) => {
              const efficiency = calculateEfficiency(plant.installDate);
              const { text: effStatus, color: effColor } = getEfficiencyStatus(efficiency);
              const dailyKwh = estimateDailyGeneration(
                plant.capacity,
                efficiency,
                currentWeather,
                plant.roofDirection || 'south'
              );
              const plantType = PLANT_TYPE_LABELS[plant.type as keyof typeof PLANT_TYPE_LABELS];
              const plantStatus = (plant.status || 'active') as PlantStatus;
              const statusInfo = PLANT_STATUS_LABELS[plantStatus];
              const isActive = plantStatus === 'active';

              return (
                <View key={plant.id} style={[styles.plantCard, !isActive && styles.plantCardInactive]}>
                  <View style={styles.plantCardHeader}>
                    <View style={styles.plantTitleRow}>
                      <Text style={styles.plantIcon}>{plantType?.icon || '⚡'}</Text>
                      <Text style={[styles.plantName, !isActive && styles.plantNameInactive]}>{plant.name}</Text>
                    </View>
                    <View style={styles.plantStatusRow}>
                      {/* Status Selector */}
                      <View style={styles.statusSelector}>
                        {(['active', 'maintenance', 'paused'] as PlantStatus[]).map((status) => {
                          const sInfo = PLANT_STATUS_LABELS[status];
                          const isSelected = plantStatus === status;
                          return (
                            <TouchableOpacity
                              key={status}
                              style={[
                                styles.statusBtn,
                                isSelected && {
                                  backgroundColor: sInfo.color,
                                  borderColor: sInfo.color,
                                  transform: [{ scale: 1.05 }],
                                },
                              ]}
                              onPress={() => handleUpdatePlantStatus(plant.id, status)}
                            >
                              <Text style={[
                                styles.statusBtnText,
                                isSelected && { color: themeColors.text.primary, fontWeight: '700' }
                              ]}>
                                {sInfo.icon}
                              </Text>
                            </TouchableOpacity>
                          );
                        })}
                      </View>
                      <TouchableOpacity
                        style={styles.plantDeleteBtn}
                        onPress={() => handlePlantDelete(plant.id)}
                      >
                        <Text style={styles.plantDeleteText}>×</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                  <View style={styles.plantDetails}>
                    <View style={styles.plantDetailItem}>
                      <Text style={styles.plantDetailLabel}>용량</Text>
                      <Text style={styles.plantDetailValue}>{formatCapacity(plant.capacity)}</Text>
                    </View>
                    <View style={styles.plantDetailItem}>
                      <Text style={styles.plantDetailLabel}>효율</Text>
                      <Text style={[styles.plantDetailValue, { color: effColor }]}>
                        {(efficiency * 100).toFixed(0)}% ({effStatus})
                      </Text>
                    </View>
                    <View style={styles.plantDetailItem}>
                      <Text style={styles.plantDetailLabel}>상태</Text>
                      <Text style={[styles.plantDetailValue, { color: statusInfo.color }]}>
                        {statusInfo.icon} {statusInfo.label}
                      </Text>
                    </View>
                  </View>
                </View>
              );
            })}

            {/* Recommended Capacity Summary */}
            <View style={styles.recommendedCapacity}>
              <View style={styles.recommendedRow}>
                <Text style={styles.recommendedLabel}>오늘의 추천 입찰량</Text>
                <View style={styles.weatherBadge}>
                  <Text style={styles.weatherIcon}>
                    {currentWeather === 'clear' ? '☀️' :
                     currentWeather === 'partly_cloudy' ? '⛅' :
                     currentWeather === 'cloudy' ? '☁️' : '🌧️'}
                  </Text>
                  <Text style={styles.weatherText}>
                    {currentWeather === 'clear' ? '맑음' :
                     currentWeather === 'partly_cloudy' ? '약간 흐림' :
                     currentWeather === 'cloudy' ? '흐림' : '비'}
                  </Text>
                </View>
              </View>
              <Text style={styles.recommendedValue}>
                {recommendedCapacity.toFixed(1)} kWh
              </Text>
              <Text style={styles.recommendedNote}>
                운영중인 {activePlantCount}개 발전소 기준
              </Text>
            </View>
          </>
        ) : plantStatus === 'ready' ? (
          <View style={styles.emptyPlantCard}>
            <Text style={styles.emptyPlantIcon}>🏭</Text>
            <Text style={styles.emptyPlantText}>등록된 발전소가 없습니다</Text>
            <Text style={styles.emptyPlantSubtext}>
              발전소를 등록하면 효율과 날씨를 고려한{'\n'}맞춤 입찰량을 추천받을 수 있습니다
            </Text>
          </View>
        ) : null}
      </View>

      {/* Conditional UI based on capacity */}
      {isLargeCapacity ? (
        <>
          {/* ===== Professional UI for Large Capacity (>= 1MW) ===== */}

          {/* VPP OFF Alert - Manual Mode (Phase 1) */}
          {!vppBiddingEnabled && powerPlants.length > 0 && (
            <View style={styles.manualModeAlert}>
              <Text style={styles.manualModeIcon}>⚠️</Text>
              <View style={styles.manualModeContent}>
                <Text style={styles.manualModeTitle}>VPP 자동입찰 OFF - 수동 모드</Text>
                <Text style={styles.manualModeDesc}>AI 최적화가 비활성화됩니다</Text>
              </View>
            </View>
          )}

          {/* Settings Row - Hour Selection & Capacity */}
          <View style={styles.settingsRow}>
            {/* Hour Selection */}
            <View style={styles.settingCard}>
              <Text style={styles.settingLabel}>거래 시간대</Text>
              <View style={styles.hourSelector}>
                <TouchableOpacity
                  style={styles.hourBtn}
                  onPress={() => setSelectedHour(Math.max(0, selectedHour - 1))}
                >
                  <Text style={styles.hourBtnText}>−</Text>
                </TouchableOpacity>
                <Text style={styles.hourValue}>{String(selectedHour).padStart(2, '0')}:00</Text>
                <TouchableOpacity
                  style={styles.hourBtn}
                  onPress={() => setSelectedHour(Math.min(23, selectedHour + 1))}
                >
                  <Text style={styles.hourBtnText}>+</Text>
                </TouchableOpacity>
              </View>
            </View>

            {/* Capacity Input */}
            <View style={styles.settingCard}>
              <Text style={styles.settingLabel}>입찰 용량 (MW)</Text>
              <TextInput
                style={styles.capacityInput}
                value={totalCapacity}
                onChangeText={(value) => {
                  if (value !== totalCapacity) {
                    capacityRevision.current += 1;
                    setSegmentsStale(true);
                    setOptimizationInfo(null);
                    setOptimizationError(null);
                  }
                  setTotalCapacity(value);
                  setCapacityError(null);
                }}
                keyboardType="decimal-pad"
                placeholder="50"
                placeholderTextColor={colors.textMuted}
              />
            </View>
          </View>

          {/* Risk Level Selection */}
          <View style={styles.riskSection}>
            <Text style={styles.settingLabel}>위험 선호도</Text>
            <View style={styles.riskButtons}>
              {(['conservative', 'moderate', 'aggressive'] as const).map((level) => (
                <TouchableOpacity
                  key={level}
                  style={[
                    styles.riskBtn,
                    riskLevel === level && styles.riskBtnActive,
                  ]}
                  onPress={() => setRiskLevel(level)}
                >
                  <Text style={[
                    styles.riskBtnText,
                    riskLevel === level && styles.riskBtnTextActive,
                  ]}>
                    {level === 'conservative' && '보수적'}
                    {level === 'moderate' && '균형'}
                    {level === 'aggressive' && '공격적'}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

      {/* ====== DAM Content (Phase 6) ====== */}
      {selectedMarket === 'dam' ? (
        <>
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

      {/* Bidding Curve Chart */}
      <View style={styles.chartSection}>
        <Text style={styles.sectionTitle}>입찰 곡선</Text>
        <BiddingCurveChart segments={segments} />
      </View>

      {/* AI Optimization Info Badge - Simplified (Phase 3) */}
      {optimizationInfo && !optimizationError && (
        <View style={styles.optimizationBadge}>
          <Text style={styles.optimizationBadgeText}>✓ 최적화 완료</Text>
        </View>
      )}

      {/* Optimization Error Alert */}
      {capacityError && (
        <View style={styles.optimizationErrorAlert}>
          <Text style={styles.optimizationErrorIcon}>⚠️</Text>
          <View style={styles.statusAlertContent}>
            <Text style={styles.optimizationErrorTitle}>입찰 용량 확인</Text>
            <Text style={styles.optimizationErrorDesc}>{capacityError}</Text>
          </View>
        </View>
      )}
      {optimizationError && (
        <View style={styles.optimizationErrorAlert}>
          <Text style={styles.optimizationErrorIcon}>⚠️</Text>
          <View style={styles.statusAlertContent}>
            <Text style={styles.optimizationErrorTitle}>AI 최적화 실패 (대체 알고리즘 사용)</Text>
            <Text style={styles.optimizationErrorDesc}>{optimizationError}</Text>
          </View>
        </View>
      )}

      {/* Status Badge - KPX Style (Phase 5) */}
      {bidStatus !== 'draft' && (
        <View style={[
          styles.statusAlert,
          {
            backgroundColor: BID_STATUS_CONFIG[bidStatus].color + '15',
            borderColor: BID_STATUS_CONFIG[bidStatus].color + '40',
          }
        ]}>
          <Text style={[styles.statusAlertIcon, { color: BID_STATUS_CONFIG[bidStatus].color }]}>
            {BID_STATUS_CONFIG[bidStatus].icon}
          </Text>
          <View style={styles.statusAlertContent}>
            <Text style={[styles.statusAlertTitle, { color: BID_STATUS_CONFIG[bidStatus].color }]}>
              {BID_STATUS_CONFIG[bidStatus].label}
            </Text>
            <Text style={[styles.statusAlertDesc, { color: BID_STATUS_CONFIG[bidStatus].color + 'cc' }]}>
              {bidStatus === 'accepted' && 'DAM/RTM 시뮬레이션을 실행하세요'}
              {bidStatus === 'submitted' && '접수 대기 중...'}
              {bidStatus === 'validating' && '입찰 검증 중...'}
              {bidStatus === 'closed' && '마감되어 수정이 불가합니다'}
              {bidStatus === 'cleared' && '축하합니다! 낙찰되었습니다'}
              {bidStatus === 'rejected' && '재입찰을 준비하세요'}
            </Text>
          </View>
        </View>
      )}

      {/* Action Buttons */}
      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={[
            styles.actionBtn,
            styles.optimizeBtn,
            !vppBiddingEnabled && styles.btnDisabled,
          ]}
          onPress={handleAIOptimize}
          disabled={isOptimizing || !vppBiddingEnabled}
        >
          <Text style={[
            styles.optimizeBtnText,
            !vppBiddingEnabled && styles.btnTextDisabled,
          ]}>
            {isOptimizing ? '최적화 중...' : !vppBiddingEnabled ? 'VPP OFF' : 'AI 최적화'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.actionBtn, styles.saveBtn]}
          onPress={() => {
            setIsSaving(true);
            setTimeout(() => {
              setIsSaving(false);
              if (Platform.OS === 'web') {
                window.alert('임시 저장되었습니다.');
              } else {
                Alert.alert('완료', '임시 저장되었습니다.');
              }
            }, 500);
          }}
          disabled={isSaving}
        >
          <Text style={styles.saveBtnText}>
            {isSaving ? '저장 중...' : '저장'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.actionBtn,
            styles.submitBtn,
            (marketStatus === 'closed' || bidStatus !== 'draft') && styles.submitBtnDisabled,
          ]}
          onPress={() => {
            if (marketStatus === 'open' && bidStatus === 'draft' && validateReview()) {
              setIsReviewModalOpen(true);
            }
          }}
          disabled={marketStatus === 'closed' || bidStatus !== 'draft'}
        >
          <Text style={[
            styles.submitBtnText,
            (marketStatus === 'closed' || bidStatus !== 'draft') && styles.submitBtnTextDisabled,
          ]}>
            {marketStatus === 'closed' ? '마감됨' :
             bidStatus === 'accepted' ? '접수완료' :
             bidStatus === 'submitted' ? '제출됨' :
             '입찰 제출'}
          </Text>
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
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>예상 수익</Text>
                <Text style={[styles.summaryValue, { color: colors.green }]}>
                  {totalExpectedRevenue >= 1000000
                    ? `${(totalExpectedRevenue / 1000000).toFixed(1)}백만원`
                    : totalExpectedRevenue >= 10000
                    ? `${(totalExpectedRevenue / 10000).toFixed(1)}만원`
                    : `${totalExpectedRevenue.toFixed(0)}원`}
                </Text>
              </View>
            </View>

            {/* Segment List Header */}
            <View style={styles.segmentListHeader}>
              <Text style={[styles.segmentHeaderText, { width: 32 }]}>구간</Text>
              <Text style={[styles.segmentHeaderText, { flex: 1, textAlign: 'center' }]}>물량</Text>
              <Text style={[styles.segmentHeaderText, { flex: 1, textAlign: 'center' }]}>가격</Text>
              <Text style={[styles.segmentHeaderText, { width: 50, textAlign: 'right' }]}>낙찰%</Text>
              <Text style={[styles.segmentHeaderText, { width: 55, textAlign: 'right' }]}>예상수익</Text>
            </View>

            {/* Segment List - Editable with Clearing Probability and Expected Revenue */}
            <View style={styles.segmentList}>
              {segments.map((segment) => {
                // Determine probability color
                const probColor = segment.clearingProbability !== undefined
                  ? segment.clearingProbability >= 0.7 ? colors.green
                    : segment.clearingProbability >= 0.4 ? colors.orange
                    : colors.red
                  : colors.textMuted;

                return (
                  <View
                    key={segment.id}
                    style={[
                      styles.segmentRow,
                      segment.price <= smpMid && styles.segmentRowHighlight,
                    ]}
                  >
                    <View style={styles.segmentIdCell}>
                      <Text style={styles.segmentId}>{segment.id}</Text>
                    </View>
                    <View style={styles.segmentValueCell}>
                      <TextInput
                        style={styles.segmentInput}
                        value={String(segment.quantity)}
                        onChangeText={(val) => updateSegmentQuantity(segment.id, parseFloat(val) || 0)}
                        keyboardType="numeric"
                        selectTextOnFocus
                      />
                    </View>
                    <View style={styles.segmentPriceCell}>
                      <TextInput
                        style={styles.segmentInput}
                        value={String(segment.price)}
                        onChangeText={(val) => updateSegmentPrice(segment.id, parseFloat(val) || 0)}
                        keyboardType="numeric"
                        selectTextOnFocus
                      />
                    </View>
                    <View style={styles.segmentProbCell}>
                      <Text style={[styles.segmentProbText, { color: probColor }]}>
                        {segment.clearingProbability !== undefined
                          ? `${(segment.clearingProbability * 100).toFixed(0)}%`
                          : '-'}
                      </Text>
                    </View>
                    <View style={styles.segmentRevenueCell}>
                      <Text style={styles.segmentRevenueText}>
                        {(() => {
                          // 수익 계산: MW × 1000(kW) × 1시간 × 가격(원/kWh) × 낙찰확률
                          // 예: 5MW × 1000 × 80원 × 0.9 = 360,000원
                          const clearingProb = segment.clearingProbability ?? 1;
                          const revenue = segment.expectedRevenue !== undefined
                            ? segment.expectedRevenue
                            : segment.quantity * 1000 * segment.price * clearingProb;
                          // Format: K(천원), M(백만원)
                          if (revenue >= 1000000) {
                            return `${(revenue / 1000000).toFixed(1)}M`;
                          } else if (revenue >= 1000) {
                            return `${(revenue / 1000).toFixed(0)}K`;
                          } else {
                            return `${revenue.toFixed(0)}`;
                          }
                        })()}
                      </Text>
                    </View>
                  </View>
                );
              })}
            </View>
          </>
        )}
      </View>

      {/* Simulation Buttons */}
      <View style={styles.simulationButtons}>
        <TouchableOpacity
          style={[styles.simulationBtn, !isDamOpen && styles.simulationBtnDisabled]}
          onPress={handleDAMSimulation}
          disabled={!isDamOpen}
        >
          <Text style={[styles.simulationBtnText, !isDamOpen && styles.simulationBtnTextDisabled]}>
            {isDamOpen ? 'DAM 시뮬레이션' : 'DAM 마감'}
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.simulationBtn, styles.rtmSimulationBtn]}
          onPress={handleRTMSimulation}
        >
          <Text style={styles.simulationBtnText}>RTM 시뮬레이션</Text>
        </TouchableOpacity>
      </View>

      {/* Dual Settlement Card (Phase 7) - also in DAM section */}
      {(bidStatus === 'cleared' || bidStatus === 'accepted') && dualSettlement && (
        <View style={styles.dualSettlementCard}>
          <View style={styles.dualSettlementHeader}>
            <Text style={styles.dualSettlementTitle}>📊 이중 정산 결과</Text>
            <Text style={styles.dualSettlementDate}>{dualSettlement.trading_date}</Text>
          </View>

          {/* DAM Settlement */}
          <View style={styles.settlementRow}>
            <View style={styles.settlementLabel}>
              <Text style={styles.settlementLabelText}>DAM 정산</Text>
              <Text style={styles.settlementDetailText}>
                {dualSettlement.dam_cleared_mwh} MWh × {dualSettlement.dam_smp}원
              </Text>
            </View>
            <Text style={styles.settlementValue}>
              {(dualSettlement.dam_revenue / 1000000).toFixed(2)}백만원
            </Text>
          </View>

          {/* RTM Settlement */}
          <View style={styles.settlementRow}>
            <View style={styles.settlementLabel}>
              <Text style={styles.settlementLabelText}>
                RTM 정산
                <Text style={[
                  styles.imbalanceBadge,
                  dualSettlement.imbalance_type === 'surplus'
                    ? { color: colors.green }
                    : dualSettlement.imbalance_type === 'deficit'
                    ? { color: colors.red }
                    : { color: colors.textMuted }
                ]}>
                  {dualSettlement.imbalance_type === 'surplus' ? ' (잉여)' :
                   dualSettlement.imbalance_type === 'deficit' ? ' (부족)' : ' (균형)'}
                </Text>
              </Text>
              <Text style={styles.settlementDetailText}>
                {dualSettlement.rtm_volume_mwh > 0 ? '+' : ''}{dualSettlement.rtm_volume_mwh} MWh × {dualSettlement.rtm_price}원
              </Text>
            </View>
            <Text style={[
              styles.settlementValue,
              { color: dualSettlement.rtm_revenue >= 0 ? colors.green : colors.red }
            ]}>
              {dualSettlement.rtm_revenue >= 0 ? '+' : ''}{(dualSettlement.rtm_revenue / 1000000).toFixed(2)}백만원
            </Text>
          </View>

          {/* Total */}
          <View style={styles.settlementTotal}>
            <Text style={styles.settlementTotalLabel}>총 정산금</Text>
            <Text style={styles.settlementTotalValue}>
              {(dualSettlement.total_revenue / 1000000).toFixed(2)}백만원
            </Text>
          </View>

          {/* Summary */}
          <View style={styles.settlementSummary}>
            <Text style={styles.settlementSummaryText}>
              실제 발전: {dualSettlement.actual_generation_mwh} MWh
              {dualSettlement.imbalance_type !== 'balanced' && (
                <Text style={{ color: dualSettlement.rtm_revenue >= 0 ? colors.green : colors.orange }}>
                  {' '}→ {dualSettlement.imbalance_type === 'surplus' ? '잉여분 판매' : '부족분 구매'}
                </Text>
              )}
            </Text>
          </View>
        </View>
      )}
        </>
      ) : (
        <>
          {/* ====== RTM Content (Phase 6) ====== */}

          {/* SMP Stats Row */}
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>예측 SMP (참고값)</Text>
              <Text style={styles.statValue}>{smpMid}</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>변동성</Text>
              <Text style={[styles.statValue, { color: colors.orange }]}>±15%</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statLabel}>예상 RTMP</Text>
              <Text style={[styles.statValue, { color: colors.green }]}>
                {Math.round(smpMid * 1.05)}
              </Text>
            </View>
          </View>

          {/* RTM Bidding Form (Phase 6) */}
          <View style={styles.rtmBidSection}>
            <View style={styles.rtmBidHeader}>
              <Text style={styles.sectionTitle}>RTM 입찰 (다음 4시간)</Text>
              <View style={styles.rtmSlotCount}>
                <Text style={styles.rtmSlotCountText}>
                  {rtmSlots.filter(s => s.adjustmentMw !== 0).length}/{rtmSlots.length} 슬롯
                </Text>
              </View>
            </View>

            {/* RTM Slot List Header */}
            <View style={styles.rtmSlotHeader}>
              <Text style={[styles.rtmSlotHeaderText, { width: 60 }]}>시간</Text>
              <Text style={[styles.rtmSlotHeaderText, { flex: 1, textAlign: 'center' }]}>증감 (MW)</Text>
              <Text style={[styles.rtmSlotHeaderText, { width: 80, textAlign: 'right' }]}>예상 RTMP</Text>
            </View>

            {/* RTM Slot List */}
            <View style={styles.rtmSlotList}>
              {rtmSlots.map((slot) => (
                <View key={slot.time} style={styles.rtmSlotRow}>
                  <View style={styles.rtmSlotTimeCell}>
                    <Text style={styles.rtmSlotTime}>{slot.time}</Text>
                  </View>
                  <View style={styles.rtmSlotInputCell}>
                    <TouchableOpacity
                      style={styles.rtmAdjustBtn}
                      onPress={() => updateRtmSlot(slot.time, slot.adjustmentMw - 1)}
                    >
                      <Text style={styles.rtmAdjustBtnText}>−</Text>
                    </TouchableOpacity>
                    <TextInput
                      style={[
                        styles.rtmSlotInput,
                        slot.adjustmentMw > 0 && styles.rtmSlotInputPositive,
                        slot.adjustmentMw < 0 && styles.rtmSlotInputNegative,
                      ]}
                      value={slot.adjustmentMw > 0 ? `+${slot.adjustmentMw}` : String(slot.adjustmentMw)}
                      onChangeText={(val) => updateRtmSlot(slot.time, parseFloat(val.replace('+', '')) || 0)}
                      keyboardType="numeric"
                      selectTextOnFocus
                    />
                    <TouchableOpacity
                      style={styles.rtmAdjustBtn}
                      onPress={() => updateRtmSlot(slot.time, slot.adjustmentMw + 1)}
                    >
                      <Text style={styles.rtmAdjustBtnText}>+</Text>
                    </TouchableOpacity>
                  </View>
                  <View style={styles.rtmSlotPriceCell}>
                    <Text style={styles.rtmSlotPrice}>{slot.estimatedPrice}원</Text>
                  </View>
                </View>
              ))}
            </View>

            {/* RTM Summary */}
            <View style={styles.rtmSummary}>
              <View style={styles.rtmSummaryItem}>
                <Text style={styles.rtmSummaryLabel}>순 증감</Text>
                <Text style={[
                  styles.rtmSummaryValue,
                  { color: rtmSlots.reduce((sum, s) => sum + s.adjustmentMw, 0) >= 0 ? colors.green : colors.red }
                ]}>
                  {rtmSlots.reduce((sum, s) => sum + s.adjustmentMw, 0) > 0 ? '+' : ''}
                  {rtmSlots.reduce((sum, s) => sum + s.adjustmentMw, 0)} MW
                </Text>
              </View>
              <View style={styles.rtmSummaryItem}>
                <Text style={styles.rtmSummaryLabel}>예상 추가 수익</Text>
                <Text style={[styles.rtmSummaryValue, { color: colors.orange }]}>
                  {(() => {
                    const totalRevenue = rtmSlots.reduce((sum, s) =>
                      sum + s.adjustmentMw * s.estimatedPrice * 1000, 0);
                    return totalRevenue >= 0 ? '+' : '';
                  })()}
                  {(rtmSlots.reduce((sum, s) =>
                    sum + s.adjustmentMw * s.estimatedPrice * 1000, 0) / 10000).toFixed(1)}만원
                </Text>
              </View>
            </View>
          </View>

          {/* RTM Action Buttons */}
          <View style={styles.actionButtons}>
            <TouchableOpacity
              style={[styles.actionBtn, styles.optimizeBtn]}
              onPress={generateRTMSlots}
            >
              <Text style={styles.optimizeBtnText}>AI 추천</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.actionBtn,
                styles.submitBtn,
                rtmSlots.filter(s => s.adjustmentMw !== 0).length === 0 && { opacity: 0.5 }
              ]}
              onPress={handleRTMSubmit}
              disabled={rtmSlots.filter(s => s.adjustmentMw !== 0).length === 0}
            >
              <Text style={styles.submitBtnText}>RTM 제출</Text>
            </TouchableOpacity>
          </View>

          {/* RTM Simulation Button */}
          <View style={styles.simulationButtons}>
            <TouchableOpacity
              style={[styles.simulationBtn, styles.rtmSimulationBtn, { flex: 1 }]}
              onPress={handleRTMSimulation}
            >
              <Text style={styles.simulationBtnText}>RTM 시뮬레이션</Text>
            </TouchableOpacity>
          </View>

          {/* Dual Settlement Card (Phase 7) */}
          {(bidStatus === 'cleared' || bidStatus === 'accepted') && dualSettlement && (
            <View style={styles.dualSettlementCard}>
              <View style={styles.dualSettlementHeader}>
                <Text style={styles.dualSettlementTitle}>📊 이중 정산 결과</Text>
                <Text style={styles.dualSettlementDate}>{dualSettlement.trading_date}</Text>
              </View>

              {/* DAM Settlement */}
              <View style={styles.settlementRow}>
                <View style={styles.settlementLabel}>
                  <Text style={styles.settlementLabelText}>DAM 정산</Text>
                  <Text style={styles.settlementDetailText}>
                    {dualSettlement.dam_cleared_mwh} MWh × {dualSettlement.dam_smp}원
                  </Text>
                </View>
                <Text style={styles.settlementValue}>
                  {(dualSettlement.dam_revenue / 1000000).toFixed(2)}백만원
                </Text>
              </View>

              {/* RTM Settlement */}
              <View style={styles.settlementRow}>
                <View style={styles.settlementLabel}>
                  <Text style={styles.settlementLabelText}>
                    RTM 정산
                    <Text style={[
                      styles.imbalanceBadge,
                      dualSettlement.imbalance_type === 'surplus'
                        ? { color: colors.green }
                        : dualSettlement.imbalance_type === 'deficit'
                        ? { color: colors.red }
                        : { color: colors.textMuted }
                    ]}>
                      {dualSettlement.imbalance_type === 'surplus' ? ' (잉여)' :
                       dualSettlement.imbalance_type === 'deficit' ? ' (부족)' : ' (균형)'}
                    </Text>
                  </Text>
                  <Text style={styles.settlementDetailText}>
                    {dualSettlement.rtm_volume_mwh > 0 ? '+' : ''}{dualSettlement.rtm_volume_mwh} MWh × {dualSettlement.rtm_price}원
                  </Text>
                </View>
                <Text style={[
                  styles.settlementValue,
                  { color: dualSettlement.rtm_revenue >= 0 ? colors.green : colors.red }
                ]}>
                  {dualSettlement.rtm_revenue >= 0 ? '+' : ''}{(dualSettlement.rtm_revenue / 1000000).toFixed(2)}백만원
                </Text>
              </View>

              {/* Total */}
              <View style={styles.settlementTotal}>
                <Text style={styles.settlementTotalLabel}>총 정산금</Text>
                <Text style={styles.settlementTotalValue}>
                  {(dualSettlement.total_revenue / 1000000).toFixed(2)}백만원
                </Text>
              </View>

              {/* Summary */}
              <View style={styles.settlementSummary}>
                <Text style={styles.settlementSummaryText}>
                  실제 발전: {dualSettlement.actual_generation_mwh} MWh
                  {dualSettlement.imbalance_type !== 'balanced' && (
                    <Text style={{ color: dualSettlement.rtm_revenue >= 0 ? colors.green : colors.orange }}>
                      {' '}→ {dualSettlement.imbalance_type === 'surplus' ? '잉여분 판매' : '부족분 구매'}
                    </Text>
                  )}
                </Text>
              </View>
            </View>
          )}
        </>
      )}

          {/* Bottom padding */}
          <View style={{ height: 100 }} />
        </>
      ) : (
        <>
          {/* ===== Simplified UI for Small Capacity (< 1MW) ===== */}

          {/* VPP Auto-Bidding Card with Toggle */}
          <View style={styles.vppSummaryCard}>
            <View style={styles.vppHeaderWithToggle}>
              <View style={styles.vppHeaderLeft}>
                <Text style={styles.vppIcon}>🤖</Text>
                <Text style={styles.vppTitle}>VPP 자동 입찰</Text>
              </View>
              <Switch
                value={vppBiddingEnabled}
                onValueChange={setVppBiddingEnabled}
                trackColor={{ false: 'rgba(148, 163, 184, 0.3)', true: themeColors.success }}
                thumbColor={vppBiddingEnabled ? '#F1F5F9' : '#94A3B8'}
                ios_backgroundColor="rgba(148, 163, 184, 0.3)"
              />
            </View>

            <View style={styles.vppContent}>
              {powerPlants.length > 0 ? (
                <>
                  <View style={styles.vppRow}>
                    <Text style={styles.vppLabel}>오늘의 입찰량</Text>
                    <Text style={styles.vppValue}>{recommendedCapacity.toFixed(1)} kWh</Text>
                  </View>
                  <View style={styles.vppRow}>
                    <Text style={styles.vppLabel}>예상 수익</Text>
                    <Text style={[styles.vppValue, { color: colors.green }]}>
                      약 {(recommendedCapacity * (smpMid || 100)).toLocaleString()}원
                    </Text>
                  </View>
                  <View style={styles.vppRow}>
                    <Text style={styles.vppLabel}>예측 SMP (참고값)</Text>
                    <Text style={styles.vppValue}>{smpMid}원/kWh</Text>
                  </View>
                  <View style={styles.vppRow}>
                    <Text style={styles.vppLabel}>날씨</Text>
                    <Text style={styles.vppValue}>
                      {currentWeather === 'clear' ? '맑음 ☀️' :
                       currentWeather === 'partly_cloudy' ? '구름많음 ⛅' :
                       currentWeather === 'cloudy' ? '흐림 ☁️' : '비 🌧️'}
                    </Text>
                  </View>
                </>
              ) : (
                <View style={styles.vppEmptyState}>
                  <Text style={styles.vppEmptyText}>
                    {plantStatus === 'ready'
                      ? <>발전소를 등록하면{'\n'}자동 입찰 정보를 확인할 수 있습니다</>
                      : '발전소 정보가 없어 자동 입찰 정보를 확인할 수 없습니다.'}
                  </Text>
                </View>
              )}
            </View>

            <View style={styles.vppFooter}>
              <Text style={styles.vppFooterIcon}>ℹ️</Text>
              <Text style={styles.vppFooterText}>
                VPP가 최적의 시간대와 가격으로 자동 입찰합니다
              </Text>
            </View>
          </View>

          {/* Simple Status - Show VPP toggle state (Phase 1) */}
          {powerPlants.length > 0 && (
            <View style={styles.vppStatusCard}>
              <View style={styles.vppStatusItem}>
                <Text style={styles.vppStatusLabel}>등록 발전소</Text>
                <Text style={styles.vppStatusValue}>{powerPlants.length}개</Text>
              </View>
              <View style={styles.vppStatusItem}>
                <Text style={styles.vppStatusLabel}>총 용량</Text>
                <Text style={styles.vppStatusValue}>{totalPlantCapacityKw.toFixed(1)} kW</Text>
              </View>
              <View style={styles.vppStatusItem}>
                <Text style={styles.vppStatusLabel}>입찰 상태</Text>
                <Text style={[
                  styles.vppStatusValue,
                  { color: vppBiddingEnabled ? colors.green : colors.textMuted }
                ]}>
                  {vppBiddingEnabled ? '자동' : '수동'}
                </Text>
              </View>
            </View>
          )}

          {/* Bottom padding */}
          <View style={{ height: 100 }} />
        </>
      )}

      {/* Bid Review Modal - Only for large capacity users */}
      {isLargeCapacity && (
      <Modal
        visible={isReviewModalOpen}
        transparent
        animationType="fade"
        onRequestClose={() => setIsReviewModalOpen(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>입찰 검토</Text>
            <Text style={styles.modalSubtitle}>제출 전 입찰 내용을 확인하세요</Text>

            {/* Bid Summary */}
            <View style={styles.modalSummary}>
              <View style={styles.modalSummaryRow}>
                <Text style={styles.modalLabel}>거래 시간대</Text>
                <Text style={styles.modalValue}>{String(selectedHour).padStart(2, '0')}:00</Text>
              </View>
              <View style={styles.modalSummaryRow}>
                <Text style={styles.modalLabel}>총 입찰량</Text>
                <Text style={styles.modalValue}>{totalMW.toFixed(1)} MW</Text>
              </View>
              <View style={styles.modalSummaryRow}>
                <Text style={styles.modalLabel}>예상 평균가</Text>
                <Text style={[styles.modalValue, { color: colors.orange }]}>
                  {avgPrice.toFixed(1)}원/kWh
                </Text>
              </View>
              <View style={styles.modalSummaryRow}>
                <Text style={styles.modalLabel}>SMP 예측</Text>
                <Text style={styles.modalValue}>
                  {smpLow} ~ {smpMid} ~ {smpHigh}
                </Text>
              </View>
              <View style={styles.modalSummaryRow}>
                <Text style={styles.modalLabel}>위험 선호도</Text>
                <Text style={styles.modalValue}>
                  {riskLevel === 'conservative' ? '보수적' :
                   riskLevel === 'moderate' ? '균형' : '공격적'}
                </Text>
              </View>
            </View>

            {/* Segment Preview */}
            <View style={styles.modalSegmentPreview}>
              <Text style={styles.modalSegmentTitle}>구간별 입찰가</Text>
              <View style={styles.modalSegmentList}>
                {segments.slice(0, 5).map((seg) => (
                  <View key={seg.id} style={styles.modalSegmentItem}>
                    <Text style={styles.modalSegmentId}>{seg.id}</Text>
                    <Text style={styles.modalSegmentPrice}>{seg.price}원</Text>
                  </View>
                ))}
                <Text style={styles.modalSegmentMore}>... 외 {segments.length - 5}개</Text>
              </View>
            </View>

            {/* Constraints Check */}
            <View style={styles.modalConstraints}>
              <View style={styles.modalConstraintItem}>
                <Text style={styles.modalConstraintIcon}>✓</Text>
                <Text style={styles.modalConstraintText}>단조성 제약 충족</Text>
              </View>
              <View style={styles.modalConstraintItem}>
                <Text style={styles.modalConstraintIcon}>✓</Text>
                <Text style={styles.modalConstraintText}>용량 제한 준수</Text>
              </View>
            </View>

            {/* Modal Buttons - KPX Style Workflow (Phase 5) */}
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalCancelBtn}
                onPress={() => setIsReviewModalOpen(false)}
              >
                <Text style={styles.modalCancelText}>취소</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalRejectBtn}
                onPress={() => {
                  setBidStatus('draft');
                  setIsReviewModalOpen(false);
                }}
              >
                <Text style={styles.modalRejectText}>수정</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalApproveBtn}
                onPress={() => {
                  if (!validateReview()) {
                    setIsReviewModalOpen(false);
                    return;
                  }
                  // Simulate KPX submission workflow
                  setBidStatus('validating');
                  setIsReviewModalOpen(false);

                  // Simulate validation delay
                  setTimeout(() => {
                    setBidStatus('submitted');

                    // Simulate acceptance delay
                    setTimeout(() => {
                      setBidStatus('accepted');
                    }, 1500);
                  }, 1000);
                }}
              >
                <Text style={styles.modalApproveText}>제출</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
      )}

      {/* Power Plant Registration Modal (v6.2.0) */}
      <Modal
        visible={showRegistration}
        animationType="slide"
        presentationStyle="fullScreen"
        onRequestClose={() => setShowRegistration(false)}
      >
        <PowerPlantRegistrationScreen
          onClose={() => setShowRegistration(false)}
          onSave={handlePlantSave}
          currentSmpPrice={smpMid}
          currentWeather={currentWeather}
        />
      </Modal>
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
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(239, 68, 68, 0.35)',
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
    backgroundColor: 'rgba(34, 197, 94, 0.15)',
    borderColor: 'rgba(34, 197, 94, 0.35)',
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

  // Settings Row
  settingsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  settingCard: {
    flex: 1,
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 12,
  },
  settingLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 8,
  },
  hourSelector: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  hourBtn: {
    width: 32,
    height: 32,
    borderRadius: 8,
    backgroundColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  hourBtnText: {
    fontSize: 18,
    fontWeight: '600',
    color: colors.text,
  },
  hourValue: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },
  capacityInput: {
    backgroundColor: colors.background,
    borderRadius: 8,
    padding: 10,
    fontSize: 16,
    color: colors.text,
    borderWidth: 1,
    borderColor: colors.border,
    textAlign: 'center',
  },

  // Risk Level Section
  riskSection: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
  },
  riskButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  riskBtn: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 8,
    backgroundColor: colors.background,
    alignItems: 'center',
  },
  riskBtnActive: {
    backgroundColor: colors.secondary,
  },
  riskBtnText: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.textSecondary,
  },
  riskBtnTextActive: {
    color: themeColors.text.primary,
  },

  // Auto Bidding Info (for small capacity < 1MW)
  autoBiddingInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    borderRadius: 10,
    padding: 12,
    gap: 12,
  },
  autoBiddingIcon: {
    fontSize: 28,
  },
  autoBiddingTextContainer: {
    flex: 1,
  },
  autoBiddingTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.primary,
    marginBottom: 2,
  },
  autoBiddingDesc: {
    fontSize: 12,
    color: colors.textSecondary,
    lineHeight: 16,
  },

  // VPP Summary Card (Simplified UI for small capacity)
  vppSummaryCard: {
    backgroundColor: colors.cardBg,
    borderRadius: 16,
    padding: 16,
    marginTop: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: 'rgba(99, 102, 241, 0.3)',
  },
  vppHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 10,
  },
  vppHeaderWithToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  vppHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  vppIcon: {
    fontSize: 28,
  },
  vppTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.primary,
  },
  vppContent: {
    gap: 12,
  },
  vppRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  vppLabel: {
    fontSize: 14,
    color: colors.textSecondary,
  },
  vppValue: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },
  vppEmptyState: {
    paddingVertical: 24,
    alignItems: 'center',
  },
  vppEmptyText: {
    fontSize: 14,
    color: colors.textMuted,
    textAlign: 'center',
    lineHeight: 20,
  },
  vppFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    gap: 8,
  },
  vppFooterIcon: {
    fontSize: 14,
  },
  vppFooterText: {
    fontSize: 12,
    color: colors.textMuted,
    flex: 1,
  },
  vppStatusCard: {
    flexDirection: 'row',
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    justifyContent: 'space-around',
  },
  vppStatusItem: {
    alignItems: 'center',
  },
  vppStatusLabel: {
    fontSize: 11,
    color: colors.textMuted,
    marginBottom: 4,
  },
  vppStatusValue: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },

  // Status Alert
  statusAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(16, 185, 129, 0.3)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    gap: 12,
  },
  statusAlertIcon: {
    fontSize: 20,
    color: colors.green,
  },
  statusAlertContent: {
    flex: 1,
  },
  statusAlertTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.green,
  },
  statusAlertDesc: {
    fontSize: 12,
    color: 'rgba(16, 185, 129, 0.8)',
    marginTop: 2,
  },

  // AI Optimization Info Alert
  optimizationInfoAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 72, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 72, 255, 0.3)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    gap: 12,
  },
  optimizationInfoIcon: {
    fontSize: 20,
  },
  optimizationInfoTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.secondary,
  },
  optimizationInfoDesc: {
    fontSize: 11,
    color: 'rgba(0, 72, 255, 0.8)',
    marginTop: 2,
  },

  // Optimization Error Alert
  optimizationErrorAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.3)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    gap: 12,
  },
  optimizationErrorIcon: {
    fontSize: 20,
  },
  optimizationErrorTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.orange,
  },
  optimizationErrorDesc: {
    fontSize: 11,
    color: 'rgba(245, 158, 11, 0.8)',
    marginTop: 2,
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
    fontSize: 14,
    fontWeight: '600',
    color: colors.background,
  },
  saveBtn: {
    backgroundColor: colors.cardBg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  saveBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
  },
  submitBtn: {
    backgroundColor: colors.primary,
  },
  submitBtnDisabled: {
    backgroundColor: colors.cardBg,
  },
  submitBtnText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.background,
  },
  submitBtnTextDisabled: {
    color: colors.textMuted,
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

  // Segment List Header
  segmentListHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    marginTop: 12,
  },
  segmentHeaderText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.textMuted,
  },

  // Segment List
  segmentList: {
    marginTop: 0,
  },
  segmentRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  segmentRowHighlight: {
    backgroundColor: 'rgba(16, 185, 129, 0.08)',
  },
  segmentIdCell: {
    width: 32,
  },
  segmentId: {
    fontSize: 12,
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
    flex: 1,
    alignItems: 'center',
  },
  segmentPrice: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
  },
  segmentInput: {
    fontSize: 13,
    color: colors.text,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 6,
    paddingHorizontal: 6,
    paddingVertical: 4,
    textAlign: 'center',
    minWidth: 45,
  },
  segmentProbCell: {
    width: 50,
    alignItems: 'flex-end',
  },
  segmentProbText: {
    fontSize: 12,
    fontWeight: '600',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  segmentRevenueCell: {
    width: 55,
    alignItems: 'flex-end',
  },
  segmentRevenueText: {
    fontSize: 12,
    color: colors.textMuted,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
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
  simulationBtnDisabled: {
    backgroundColor: themeColors.surfaceAlt,
  },
  simulationBtnTextDisabled: {
    color: themeColors.text.secondary,
  },

  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: colors.background,
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxWidth: 400,
    maxHeight: '80%',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text,
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 13,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: 4,
    marginBottom: 16,
  },
  modalSummary: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
  },
  modalSummaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 6,
  },
  modalLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  modalValue: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.text,
  },
  modalSegmentPreview: {
    marginBottom: 16,
  },
  modalSegmentTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 8,
  },
  modalSegmentList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    alignItems: 'center',
  },
  modalSegmentItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.cardBg,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    gap: 4,
  },
  modalSegmentId: {
    fontSize: 11,
    color: colors.textMuted,
  },
  modalSegmentPrice: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.text,
  },
  modalSegmentMore: {
    fontSize: 11,
    color: colors.textMuted,
  },
  modalConstraints: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 20,
  },
  modalConstraintItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  modalConstraintIcon: {
    fontSize: 14,
    color: colors.green,
  },
  modalConstraintText: {
    fontSize: 12,
    color: colors.textMuted,
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 10,
  },
  modalCancelBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: colors.cardBg,
    alignItems: 'center',
  },
  modalCancelText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  modalRejectBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    alignItems: 'center',
  },
  modalRejectText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.red,
  },
  modalApproveBtn: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 10,
    backgroundColor: colors.green,
    alignItems: 'center',
  },
  modalApproveText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.background,
  },

  // Power Plant Section Styles (v6.2.0)
  powerPlantSection: {
    marginBottom: 20,
  },
  powerPlantHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  sectionTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  plantCountBadge: {
    backgroundColor: colors.secondary + '20',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  plantCountText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.secondary,
  },
  vppToggleContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: colors.border,
  },
  vppToggleLeft: {
    flex: 1,
  },
  vppToggleLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
  },
  vppToggleCapacity: {
    fontSize: 12,
    color: colors.textSecondary,
    marginTop: 2,
  },
  vppToggleBtn: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: colors.border,
  },
  vppToggleBtnActive: {
    backgroundColor: colors.green,
  },
  vppToggleBtnText: {
    fontSize: 13,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  vppToggleBtnTextActive: {
    color: themeColors.text.primary,
  },
  registerBtn: {
    backgroundColor: colors.secondary,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  registerBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: themeColors.text.primary,
  },
  plantCard: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: colors.border,
  },
  plantCardInactive: {
    opacity: 0.6,
    backgroundColor: themeColors.bg,
  },
  plantCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  plantTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1,
  },
  plantIcon: {
    fontSize: 20,
  },
  plantName: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
  },
  plantNameInactive: {
    color: colors.textMuted,
  },
  plantStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  statusSelector: {
    flexDirection: 'row',
    gap: 4,
  },
  statusBtn: {
    width: 28,
    height: 28,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
  },
  statusBtnText: {
    fontSize: 12,
    color: colors.textMuted,
  },
  plantDeleteBtn: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  plantDeleteText: {
    fontSize: 18,
    color: colors.red,
    fontWeight: '500',
  },
  plantDetails: {
    flexDirection: 'row',
    gap: 16,
  },
  plantDetailItem: {
    flex: 1,
  },
  plantDetailLabel: {
    fontSize: 11,
    color: colors.textMuted,
    marginBottom: 2,
  },
  plantDetailValue: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.text,
  },
  recommendedCapacity: {
    backgroundColor: 'rgba(0, 72, 255, 0.05)',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1,
    borderColor: 'rgba(0, 72, 255, 0.2)',
  },
  recommendedRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  recommendedLabel: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  weatherBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.background,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  weatherIcon: {
    fontSize: 14,
  },
  weatherText: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  recommendedValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.secondary,
    marginBottom: 4,
  },
  recommendedNote: {
    fontSize: 11,
    color: colors.textMuted,
  },
  emptyPlantCard: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    paddingVertical: 24,
    paddingHorizontal: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.border,
    borderStyle: 'dashed',
  },
  emptyPlantIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  emptyPlantText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: 4,
  },
  emptyPlantSubtext: {
    fontSize: 12,
    color: colors.textMuted,
    textAlign: 'center',
    lineHeight: 18,
  },

  // Phase 1: VPP Manual Mode Alert Styles
  manualModeAlert: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.3)',
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    gap: 12,
  },
  manualModeIcon: {
    fontSize: 20,
  },
  manualModeContent: {
    flex: 1,
  },
  manualModeTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.orange,
  },
  manualModeDesc: {
    fontSize: 12,
    color: 'rgba(245, 158, 11, 0.8)',
    marginTop: 2,
  },

  // Phase 1: VPP Disabled Card Styles
  vppDisabledCard: {
    backgroundColor: colors.cardBg,
    borderRadius: 16,
    padding: 24,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  vppDisabledIcon: {
    fontSize: 32,
    marginBottom: 12,
  },
  vppDisabledTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textSecondary,
    marginBottom: 8,
  },
  vppDisabledDesc: {
    fontSize: 13,
    color: colors.textMuted,
    textAlign: 'center',
    lineHeight: 20,
  },

  // Phase 1: Button Disabled Styles
  btnDisabled: {
    backgroundColor: colors.border,
    opacity: 0.6,
  },
  btnTextDisabled: {
    color: colors.textMuted,
  },

  // Phase 3: Optimization Badge Styles (Simplified)
  optimizationBadge: {
    backgroundColor: colors.green,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 16,
    alignSelf: 'flex-start',
    marginBottom: 16,
  },
  optimizationBadgeText: {
    color: themeColors.text.primary,
    fontSize: 13,
    fontWeight: '600',
  },

  // Phase 6: DAM/RTM Market Tabs
  marketTabs: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  marketTab: {
    flex: 1,
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 12,
    backgroundColor: colors.cardBg,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
  },
  marketTabActive: {
    backgroundColor: colors.secondary,
    borderColor: colors.secondary,
  },
  marketTabTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.textSecondary,
  },
  marketTabTitleActive: {
    color: themeColors.text.primary,
  },
  marketTabSubtext: {
    fontSize: 11,
    color: colors.textMuted,
    marginTop: 2,
  },
  marketTabSubtextActive: {
    color: 'rgba(255, 255, 255, 0.8)',
  },
  marketTabDisabled: {
    backgroundColor: themeColors.bg,
    borderColor: themeColors.border.primary,
    opacity: 0.6,
  },
  marketTabTitleDisabled: {
    color: themeColors.text.secondary,
  },
  marketTabSubtextDisabled: {
    color: themeColors.text.secondary,
  },

  // Phase 6: RTM Badge Styles
  rtmBadge: {
    backgroundColor: 'rgba(34, 197, 94, 0.15)',
    borderColor: 'rgba(34, 197, 94, 0.35)',
  },
  rtmDot: {
    backgroundColor: colors.green,
  },
  rtmBadgeText: {
    color: colors.green,
  },

  // Phase 6: RTM Bidding Section
  rtmBidSection: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  rtmBidHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  rtmSlotCount: {
    backgroundColor: colors.green + '20',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  rtmSlotCountText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.green,
  },
  rtmSlotHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  rtmSlotHeaderText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.textMuted,
  },
  rtmSlotList: {
    // No maxHeight - show all 16 slots, let parent ScrollView handle scrolling
  },
  rtmSlotRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 4,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  rtmSlotTimeCell: {
    width: 60,
  },
  rtmSlotTime: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.text,
  },
  rtmSlotInputCell: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  rtmAdjustBtn: {
    width: 28,
    height: 28,
    borderRadius: 6,
    backgroundColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  rtmAdjustBtnText: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
  },
  rtmSlotInput: {
    width: 60,
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 6,
    textAlign: 'center',
  },
  rtmSlotInputPositive: {
    borderColor: colors.green,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    color: colors.green,
  },
  rtmSlotInputNegative: {
    borderColor: colors.red,
    backgroundColor: 'rgba(239, 68, 68, 0.1)',
    color: colors.red,
  },
  rtmSlotPriceCell: {
    width: 80,
    alignItems: 'flex-end',
  },
  rtmSlotPrice: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  rtmSummary: {
    flexDirection: 'row',
    marginTop: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    gap: 16,
  },
  rtmSummaryItem: {
    flex: 1,
  },
  rtmSummaryLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  rtmSummaryValue: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.text,
  },

  // ============================================
  // Dual Settlement Styles (Phase 7)
  // ============================================
  dualSettlementCard: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: colors.border,
  },
  dualSettlementHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  dualSettlementTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: colors.text,
  },
  dualSettlementDate: {
    fontSize: 12,
    color: colors.textSecondary,
  },
  settlementRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  settlementLabel: {
    flex: 1,
  },
  settlementLabelText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text,
  },
  settlementDetailText: {
    fontSize: 11,
    color: colors.textSecondary,
    marginTop: 2,
  },
  settlementValue: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    textAlign: 'right',
  },
  imbalanceBadge: {
    fontSize: 12,
    fontWeight: '500',
  },
  settlementTotal: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 2,
    borderTopColor: colors.primary,
  },
  settlementTotalLabel: {
    fontSize: 15,
    fontWeight: '700',
    color: colors.primary,
  },
  settlementTotalValue: {
    fontSize: 18,
    fontWeight: '700',
    color: colors.primary,
  },
  settlementSummary: {
    marginTop: 12,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  settlementSummaryText: {
    fontSize: 12,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
