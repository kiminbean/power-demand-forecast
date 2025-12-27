/**
 * KPX DAM (Day-Ahead Market) Simulation Screen - RE-BMS Mobile v6.1
 * Realistic KPX electricity market bid matching simulation
 * Matches web-v6.1.0 features
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Dimensions,
  Platform,
} from 'react-native';

// Navigation and icons are handled by parent component
// Using emoji icons for cross-platform compatibility

import { colors, spacing, borderRadius, fontSize } from '../theme/colors';

const { width: screenWidth } = Dimensions.get('window');

// Types
interface MarketBid {
  id: string;
  bidder: string;
  type: 'supply' | 'demand';
  resourceType?: 'wind' | 'solar' | 'thermal' | 'lng' | 'hydro' | 'biomass';
  quantity: number;
  price: number;
  isOurs?: boolean;
  status: 'pending' | 'accepted' | 'rejected' | 'partial';
  acceptedQuantity?: number;
}

interface SimulationState {
  phase: 'idle' | 'collecting' | 'sorting' | 'matching' | 'clearing' | 'complete';
  progress: number;
  clearingPrice: number | null;
  clearingQuantity: number | null;
  totalSupply: number;
  totalDemand: number;
  ourAccepted: number;
  ourRevenue: number;
  marketType: 'normal' | 'oversupply' | 'shortage';
}

// Jeju power generators
const JEJU_GENERATORS = [
  { name: '한림풍력', type: 'wind' as const, capacity: 21, minPrice: 0, maxPrice: 15 },
  { name: '김녕풍력', type: 'wind' as const, capacity: 15, minPrice: 0, maxPrice: 18 },
  { name: '가시리풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 12 },
  { name: '행원풍력', type: 'wind' as const, capacity: 10, minPrice: 0, maxPrice: 20 },
  { name: '탐라해상풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 10 },
  { name: '성산태양광', type: 'solar' as const, capacity: 20, minPrice: 0, maxPrice: 22 },
  { name: '제주태양광1', type: 'solar' as const, capacity: 25, minPrice: 0, maxPrice: 20 },
  { name: '제주태양광2', type: 'solar' as const, capacity: 18, minPrice: 0, maxPrice: 25 },
  { name: '제주화력#1', type: 'thermal' as const, capacity: 75, minPrice: 85, maxPrice: 110 },
  { name: '제주화력#2', type: 'thermal' as const, capacity: 75, minPrice: 88, maxPrice: 115 },
  { name: '남제주화력', type: 'thermal' as const, capacity: 100, minPrice: 90, maxPrice: 120 },
  { name: '제주LNG발전', type: 'lng' as const, capacity: 150, minPrice: 100, maxPrice: 140 },
  { name: '제주바이오매스', type: 'biomass' as const, capacity: 8, minPrice: 70, maxPrice: 90 },
];

// Jeju demand sources
const JEJU_DEMAND_SOURCES = [
  { name: '제주본도', baseQuantity: 250, priceWillingness: 'high' as const },
  { name: '한전제주지사', baseQuantity: 80, priceWillingness: 'high' as const },
  { name: '산업단지', baseQuantity: 60, priceWillingness: 'medium' as const },
  { name: '관광시설', baseQuantity: 40, priceWillingness: 'medium' as const },
  { name: '농업용전력', baseQuantity: 30, priceWillingness: 'low' as const },
];

// Demand multiplier by hour
const DEMAND_MULTIPLIER: Record<number, number> = {
  0: 0.65, 1: 0.60, 2: 0.55, 3: 0.55, 4: 0.58, 5: 0.65,
  6: 0.80, 7: 0.95, 8: 1.10, 9: 1.15, 10: 1.18, 11: 1.20,
  12: 1.15, 13: 1.10, 14: 1.12, 15: 1.15, 16: 1.18, 17: 1.22,
  18: 1.25, 19: 1.20, 20: 1.10, 21: 0.95, 22: 0.85, 23: 0.75,
};

// Solar availability by hour
const SOLAR_AVAILABILITY: Record<number, number> = {
  0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0.1,
  6: 0.3, 7: 0.5, 8: 0.7, 9: 0.85, 10: 0.95, 11: 1.00,
  12: 1.00, 13: 0.95, 14: 0.90, 15: 0.80, 16: 0.65, 17: 0.45,
  18: 0.25, 19: 0.1, 20: 0, 21: 0, 22: 0, 23: 0,
};

const phaseLabels: Record<SimulationState['phase'], string> = {
  idle: '대기 중',
  collecting: '입찰 수집',
  sorting: 'Merit Order',
  matching: '수급 매칭',
  clearing: '청산가격',
  complete: '매칭 완료',
};

// Icon component - using emoji for cross-platform compatibility
function Icon({ name, size, color }: { name: string; size: number; color: string }) {
  const iconMap: { [key: string]: string } = {
    'arrow-back': '←',
    'play': '▶',
    'refresh': '↻',
    'checkmark-circle': '✓',
    'close-circle': '✗',
    'flash': '⚡',
    'business': '🏢',
    'trending-up': '📈',
    'time': '⏱',
    'sunny': '☀️',
    'cloudy': '💨',
    'flame': '🔥',
  };
  return <Text style={{ fontSize: size * 0.8, color }}>{iconMap[name] || '•'}</Text>;
}

// Calculate clearing price
function calculateClearingPrice(
  supplyBids: MarketBid[],
  demandBids: MarketBid[]
): { clearingPrice: number; clearingQuantity: number } | null {
  if (supplyBids.length === 0 || demandBids.length === 0) return null;

  const sortedSupply = [...supplyBids].sort((a, b) => a.price - b.price);
  const sortedDemand = [...demandBids].sort((a, b) => b.price - a.price);

  let cumSupply = 0;
  let cumDemand = 0;
  sortedDemand.forEach(bid => cumDemand += bid.quantity);

  let clearingPrice = 0;
  let clearingQuantity = 0;
  let remainingDemand = cumDemand;

  for (const bid of sortedSupply) {
    if (remainingDemand <= 0) break;

    const accepted = Math.min(bid.quantity, remainingDemand);
    clearingQuantity += accepted;
    clearingPrice = bid.price;
    remainingDemand -= accepted;
  }

  return clearingQuantity > 0 ? { clearingPrice, clearingQuantity } : null;
}

// Update bid statuses
function updateBidStatuses(
  bids: MarketBid[],
  clearingPrice: number,
  clearingQuantity: number,
  bidType: 'supply' | 'demand'
): MarketBid[] {
  const sortedBids = [...bids].sort((a, b) =>
    bidType === 'supply' ? a.price - b.price : b.price - a.price
  );

  let remainingQuantity = clearingQuantity;

  return sortedBids.map(bid => {
    if (remainingQuantity <= 0) {
      return { ...bid, status: 'rejected' as const, acceptedQuantity: 0 };
    }

    const meetsPrice = bidType === 'supply'
      ? bid.price <= clearingPrice
      : bid.price >= clearingPrice;

    if (!meetsPrice) {
      return { ...bid, status: 'rejected' as const, acceptedQuantity: 0 };
    }

    if (remainingQuantity >= bid.quantity) {
      remainingQuantity -= bid.quantity;
      return { ...bid, status: 'accepted' as const, acceptedQuantity: bid.quantity };
    } else {
      const accepted = remainingQuantity;
      remainingQuantity = 0;
      return { ...bid, status: 'partial' as const, acceptedQuantity: accepted };
    }
  });
}

interface KPXSimulationScreenProps {
  onBack?: () => void;
}

export default function KPXSimulationScreen({ onBack }: KPXSimulationScreenProps = {}) {
  // Using default values - navigation params are passed via App.tsx
  const selectedHour = 12;
  const smpForecast = { q10: 54, q50: 77, q90: 126 };
  const ourSegments = [
    { id: 1, quantity: 5, price: 80 },
    { id: 2, quantity: 5, price: 85 },
    { id: 3, quantity: 5, price: 90 },
    { id: 4, quantity: 5, price: 95 },
    { id: 5, quantity: 5, price: 100 },
  ];

  const [simulation, setSimulation] = useState<SimulationState>({
    phase: 'idle',
    progress: 0,
    clearingPrice: null,
    clearingQuantity: null,
    totalSupply: 0,
    totalDemand: 0,
    ourAccepted: 0,
    ourRevenue: 0,
    marketType: 'normal',
  });

  const [supplyBids, setSupplyBids] = useState<MarketBid[]>([]);
  const [demandBids, setDemandBids] = useState<MarketBid[]>([]);

  // Generate supply bids (memoized to prevent re-generation)
  const initialSupplyBids = useMemo(() => {
    const bids: MarketBid[] = [];
    const solarAvail = SOLAR_AVAILABILITY[selectedHour] || 0;

    // Use seeded random for consistent values
    const seededRandom = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    JEJU_GENERATORS.forEach((gen, idx) => {
      let effectiveCapacity = gen.capacity;
      if (gen.type === 'solar') {
        effectiveCapacity = Math.round(gen.capacity * solarAvail);
      }
      if (effectiveCapacity <= 0) return;

      const priceVariation = (seededRandom(idx * 100 + selectedHour) - 0.5) * 0.2;
      const price = Math.round(gen.minPrice + (gen.maxPrice - gen.minPrice) * (0.5 + priceVariation));

      bids.push({
        id: `gen-${idx}`,
        bidder: gen.name,
        type: 'supply',
        resourceType: gen.type,
        quantity: Math.round(effectiveCapacity * (0.7 + seededRandom(idx * 200 + selectedHour) * 0.3)),
        price: Math.max(0, Math.min(200, price)),
        status: 'pending',
      });
    });

    // Add our bids
    ourSegments.forEach((seg, idx) => {
      bids.push({
        id: `our-${idx}`,
        bidder: '우리회사',
        type: 'supply',
        resourceType: 'solar',
        quantity: seg.quantity,
        price: seg.price,
        isOurs: true,
        status: 'pending',
      });
    });

    return bids;
  }, [selectedHour, ourSegments]);

  // Generate demand bids (memoized to prevent re-generation)
  const initialDemandBids = useMemo(() => {
    const demandMultiplier = DEMAND_MULTIPLIER[selectedHour] || 1.0;

    // Use seeded random for consistent values
    const seededRandom = (seed: number) => {
      const x = Math.sin(seed) * 10000;
      return x - Math.floor(x);
    };

    return JEJU_DEMAND_SOURCES.map((d, idx) => {
      const basePrice = d.priceWillingness === 'high' ? smpForecast.q90 * 1.2 :
                        d.priceWillingness === 'medium' ? smpForecast.q90 :
                        smpForecast.q50;

      return {
        id: `dem-${idx}`,
        bidder: d.name,
        type: 'demand' as const,
        quantity: Math.round(d.baseQuantity * demandMultiplier * (0.9 + seededRandom(idx * 300 + selectedHour) * 0.2)),
        price: Math.round(basePrice * (0.95 + seededRandom(idx * 400 + selectedHour) * 0.1)),
        status: 'pending' as const,
      };
    });
  }, [selectedHour, smpForecast]);

  // Initialize bids only once on mount
  useEffect(() => {
    setSupplyBids(initialSupplyBids);
    setDemandBids(initialDemandBids);
  }, []);

  // Run simulation
  const runSimulation = useCallback(() => {
    const phases: SimulationState['phase'][] = ['collecting', 'sorting', 'matching', 'clearing', 'complete'];
    let currentPhaseIdx = 0;

    const interval = setInterval(() => {
      setSimulation(prev => {
        if (prev.progress >= 100) {
          currentPhaseIdx++;
          if (currentPhaseIdx >= phases.length) {
            clearInterval(interval);

            const result = calculateClearingPrice(supplyBids, demandBids);

            if (result) {
              const updatedSupply = updateBidStatuses(supplyBids, result.clearingPrice, result.clearingQuantity, 'supply');
              const updatedDemand = updateBidStatuses(demandBids, result.clearingPrice, result.clearingQuantity, 'demand');

              setSupplyBids(updatedSupply);
              setDemandBids(updatedDemand);

              const ourAccepted = updatedSupply
                .filter(b => b.isOurs && (b.status === 'accepted' || b.status === 'partial'))
                .reduce((sum, b) => sum + (b.acceptedQuantity || 0), 0);

              const totalSupply = supplyBids.reduce((sum, b) => sum + b.quantity, 0);
              const totalDemand = demandBids.reduce((sum, b) => sum + b.quantity, 0);

              let marketType: 'normal' | 'oversupply' | 'shortage' = 'normal';
              if (totalSupply > totalDemand * 1.2) marketType = 'oversupply';
              else if (totalDemand > totalSupply * 1.2) marketType = 'shortage';

              return {
                ...prev,
                phase: 'complete',
                progress: 100,
                clearingPrice: result.clearingPrice,
                clearingQuantity: result.clearingQuantity,
                totalSupply,
                totalDemand,
                ourAccepted,
                ourRevenue: ourAccepted * result.clearingPrice,
                marketType,
              };
            }

            return { ...prev, phase: 'complete', progress: 100 };
          }
          return { ...prev, phase: phases[currentPhaseIdx], progress: 0 };
        }
        return { ...prev, progress: prev.progress + 10 };
      });
    }, 150);

    setSimulation(prev => ({ ...prev, phase: 'collecting', progress: 0 }));

    return () => clearInterval(interval);
  }, [supplyBids, demandBids]);

  // Reset simulation
  const resetSimulation = useCallback(() => {
    setSupplyBids(initialSupplyBids);
    setDemandBids(initialDemandBids);
    setSimulation({
      phase: 'idle',
      progress: 0,
      clearingPrice: null,
      clearingQuantity: null,
      totalSupply: 0,
      totalDemand: 0,
      ourAccepted: 0,
      ourRevenue: 0,
      marketType: 'normal',
    });
  }, [initialSupplyBids, initialDemandBids]);

  const totalSupplyBids = supplyBids.reduce((sum, b) => sum + b.quantity, 0);
  const totalDemandBids = demandBids.reduce((sum, b) => sum + b.quantity, 0);
  const ourTotalBid = supplyBids.filter(b => b.isOurs).reduce((sum, b) => sum + b.quantity, 0);

  return (
    <ScrollView style={styles.container} showsVerticalScrollIndicator={false}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => onBack?.()}
        >
          <Icon name="arrow-back" size={24} color={colors.text.primary} />
        </TouchableOpacity>
        <View style={styles.headerText}>
          <Text style={styles.headerTitle}>KPX 하루전시장 시뮬레이션</Text>
          <Text style={styles.headerSubtitle}>
            {String(selectedHour).padStart(2, '0')}:00 거래시간 | SMP 예측: {smpForecast.q50}원/kWh
          </Text>
        </View>
      </View>

      {/* Summary Cards */}
      <View style={styles.summaryRow}>
        <View style={[styles.summaryCard, { borderLeftColor: colors.status.success }]}>
          <Text style={styles.summaryLabel}>총 공급</Text>
          <Text style={[styles.summaryValue, { color: colors.status.success }]}>
            {totalSupplyBids} MW
          </Text>
        </View>
        <View style={[styles.summaryCard, { borderLeftColor: colors.status.warning }]}>
          <Text style={styles.summaryLabel}>총 수요</Text>
          <Text style={[styles.summaryValue, { color: colors.status.warning }]}>
            {totalDemandBids} MW
          </Text>
        </View>
        <View style={[styles.summaryCard, { borderLeftColor: colors.brand.primary }]}>
          <Text style={styles.summaryLabel}>우리 입찰</Text>
          <Text style={[styles.summaryValue, { color: colors.brand.primary }]}>
            {ourTotalBid} MW
          </Text>
        </View>
      </View>

      {/* Phase Progress */}
      <View style={styles.phaseContainer}>
        <View style={styles.phaseHeader}>
          <Icon name="time" size={18} color={colors.brand.primary} />
          <Text style={styles.phaseTitle}>처리 단계</Text>
          <View style={[
            styles.phaseBadge,
            { backgroundColor: simulation.phase === 'complete' ? `${colors.status.success}30` : `${colors.status.warning}30` }
          ]}>
            <Text style={[
              styles.phaseBadgeText,
              { color: simulation.phase === 'complete' ? colors.status.success : colors.status.warning }
            ]}>
              {phaseLabels[simulation.phase]}
            </Text>
          </View>
        </View>
        <View style={styles.progressBar}>
          {(['collecting', 'sorting', 'matching', 'clearing', 'complete'] as const).map((phase, idx) => {
            const phases = ['collecting', 'sorting', 'matching', 'clearing', 'complete'];
            const currentIdx = phases.indexOf(simulation.phase);
            return (
              <View
                key={phase}
                style={[
                  styles.progressSegment,
                  {
                    backgroundColor: currentIdx > idx ? colors.status.success :
                      currentIdx === idx ? colors.status.warning : colors.background.tertiary,
                  },
                ]}
              />
            );
          })}
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actionRow}>
        {simulation.phase === 'idle' ? (
          <TouchableOpacity style={styles.startButton} onPress={runSimulation}>
            <Icon name="play" size={20} color={colors.text.primary} />
            <Text style={styles.startButtonText}>시뮬레이션 시작</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity style={styles.resetButton} onPress={resetSimulation}>
            <Icon name="refresh" size={20} color={colors.text.primary} />
            <Text style={styles.resetButtonText}>다시 실행</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Results Panel */}
      {simulation.phase === 'complete' && simulation.clearingPrice && (
        <View style={styles.resultsContainer}>
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>청산 결과</Text>
            <View style={styles.resultMain}>
              <Text style={styles.clearingPrice}>{simulation.clearingPrice.toFixed(1)}원</Text>
              <Text style={styles.resultLabel}>청산가격 (SMP)</Text>
            </View>
            <View style={styles.resultGrid}>
              <View style={styles.resultItem}>
                <Text style={styles.resultValue}>{simulation.clearingQuantity?.toFixed(0)}</Text>
                <Text style={styles.resultItemLabel}>거래량 (MW)</Text>
              </View>
              <View style={styles.resultItem}>
                <Text style={styles.resultValue}>
                  {supplyBids.filter(b => b.status === 'accepted' || b.status === 'partial').length}
                </Text>
                <Text style={styles.resultItemLabel}>낙찰 발전사</Text>
              </View>
            </View>
          </View>

          <View style={[
            styles.resultCard,
            simulation.ourAccepted > 0 && { borderColor: colors.status.success, borderWidth: 2 }
          ]}>
            <Text style={styles.resultTitle}>우리 입찰 결과</Text>
            {simulation.ourAccepted > 0 ? (
              <>
                <View style={styles.successBadge}>
                  <Icon name="checkmark-circle" size={24} color={colors.status.success} />
                  <Text style={styles.successText}>낙찰 성공!</Text>
                </View>
                <View style={styles.resultGrid}>
                  <View style={styles.resultItem}>
                    <Text style={[styles.resultValue, { color: colors.status.success }]}>
                      {simulation.ourAccepted.toFixed(1)} MW
                    </Text>
                    <Text style={styles.resultItemLabel}>낙찰 물량</Text>
                  </View>
                  <View style={styles.resultItem}>
                    <Text style={[styles.resultValue, { color: colors.status.success }]}>
                      {(simulation.ourRevenue / 1000).toFixed(1)}천원
                    </Text>
                    <Text style={styles.resultItemLabel}>예상 수익</Text>
                  </View>
                </View>
              </>
            ) : (
              <View style={styles.failBadge}>
                <Icon name="close-circle" size={24} color={colors.status.danger} />
                <Text style={styles.failText}>낙찰 실패</Text>
              </View>
            )}
          </View>
        </View>
      )}

      {/* Supply Bid Table */}
      <View style={styles.tableContainer}>
        <View style={styles.tableHeader}>
          <Icon name="flash" size={18} color={colors.status.success} />
          <Text style={styles.tableTitle}>공급 입찰 (발전사) - Merit Order</Text>
        </View>
        <View style={styles.tableHeaderRow}>
          <Text style={[styles.tableHeaderCell, { flex: 2 }]}>발전사</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>물량</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>가격</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>상태</Text>
        </View>
        {supplyBids.slice(0, 15).map((bid) => (
          <View
            key={bid.id}
            style={[
              styles.tableRow,
              bid.isOurs && styles.tableRowOurs,
              bid.status === 'accepted' && styles.tableRowAccepted,
              bid.status === 'partial' && styles.tableRowPartial,
              bid.status === 'rejected' && styles.tableRowRejected,
            ]}
          >
            <View style={[styles.tableCellContainer, { flex: 2, flexDirection: 'row', alignItems: 'center' }]}>
              <Text style={styles.tableCellText}>{bid.bidder}</Text>
              {bid.isOurs && (
                <View style={styles.oursBadge}>
                  <Text style={styles.oursBadgeText}>우리</Text>
                </View>
              )}
            </View>
            <Text style={[styles.tableCell, { flex: 1 }]}>{bid.quantity.toFixed(1)}</Text>
            <Text style={[styles.tableCell, { flex: 1 }]}>{bid.price.toFixed(0)}원</Text>
            <View style={[styles.tableCellContainer, { flex: 1 }]}>
              {bid.status === 'accepted' && (
                <Text style={styles.statusAccepted}>✓ 낙찰</Text>
              )}
              {bid.status === 'partial' && (
                <Text style={styles.statusPartial}>△ 부분</Text>
              )}
              {bid.status === 'rejected' && (
                <Text style={styles.statusRejected}>✗ 탈락</Text>
              )}
              {bid.status === 'pending' && (
                <Text style={styles.statusPending}>-</Text>
              )}
            </View>
          </View>
        ))}
      </View>

      {/* Demand Bid Table */}
      <View style={styles.tableContainer}>
        <View style={styles.tableHeader}>
          <Icon name="business" size={18} color={colors.status.warning} />
          <Text style={styles.tableTitle}>수요 입찰 (수요처)</Text>
        </View>
        <View style={styles.tableHeaderRow}>
          <Text style={[styles.tableHeaderCell, { flex: 2 }]}>수요처</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>물량</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>희망가</Text>
          <Text style={[styles.tableHeaderCell, { flex: 1 }]}>상태</Text>
        </View>
        {demandBids.map((bid) => (
          <View
            key={bid.id}
            style={[
              styles.tableRow,
              bid.status === 'accepted' && styles.tableRowAccepted,
              bid.status === 'partial' && styles.tableRowPartial,
              bid.status === 'rejected' && styles.tableRowRejected,
            ]}
          >
            <Text style={[styles.tableCell, { flex: 2 }]}>{bid.bidder}</Text>
            <Text style={[styles.tableCell, { flex: 1 }]}>{bid.quantity.toFixed(1)}</Text>
            <Text style={[styles.tableCell, { flex: 1 }]}>{bid.price.toFixed(0)}원</Text>
            <View style={[styles.tableCellContainer, { flex: 1 }]}>
              {bid.status === 'accepted' && (
                <Text style={styles.statusAccepted}>✓ 체결</Text>
              )}
              {bid.status === 'partial' && (
                <Text style={styles.statusPartial}>△ 부분</Text>
              )}
              {bid.status === 'rejected' && (
                <Text style={styles.statusRejected}>✗ 미체결</Text>
              )}
              {bid.status === 'pending' && (
                <Text style={styles.statusPending}>-</Text>
              )}
            </View>
          </View>
        ))}
      </View>

      {/* Info Box */}
      <View style={styles.infoBox}>
        <Text style={styles.infoTitle}>KPX 하루전시장(DAM) 입찰 매칭 원리</Text>
        <Text style={styles.infoText}>
          • Merit Order: 발전사는 가격 오름차순으로 정렬{'\n'}
          • 균일가격(SMP): 모든 낙찰 발전사는 동일 청산가격으로 정산{'\n'}
          • 마지널 발전기: 청산가격은 마지막 급전 발전기의 입찰가{'\n'}
          • 재생에너지 우선: 태양광/풍력은 Merit Order 상위 배치
        </Text>
      </View>

      <View style={{ height: 40 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    backgroundColor: colors.background.secondary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  backButton: {
    padding: spacing.sm,
    marginRight: spacing.sm,
  },
  headerText: {
    flex: 1,
  },
  headerTitle: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  headerSubtitle: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
    marginTop: 2,
  },

  summaryRow: {
    flexDirection: 'row',
    padding: spacing.md,
    gap: spacing.sm,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: colors.background.card,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderLeftWidth: 3,
  },
  summaryLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
  },
  summaryValue: {
    fontSize: fontSize.xl,
    fontWeight: 'bold',
    marginTop: 4,
  },

  phaseContainer: {
    margin: spacing.md,
    marginTop: 0,
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.md,
    padding: spacing.md,
  },
  phaseHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.md,
    gap: spacing.sm,
  },
  phaseTitle: {
    flex: 1,
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.text.primary,
  },
  phaseBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: borderRadius.full,
  },
  phaseBadgeText: {
    fontSize: fontSize.xs,
    fontWeight: '600',
  },
  progressBar: {
    flexDirection: 'row',
    gap: 4,
  },
  progressSegment: {
    flex: 1,
    height: 6,
    borderRadius: 3,
  },

  actionRow: {
    paddingHorizontal: spacing.md,
    marginBottom: spacing.md,
  },
  startButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.brand.primary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
  },
  startButtonText: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.text.primary,
  },
  resetButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    backgroundColor: colors.background.tertiary,
    padding: spacing.md,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border.primary,
  },
  resetButtonText: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.text.primary,
  },

  resultsContainer: {
    paddingHorizontal: spacing.md,
    gap: spacing.md,
    marginBottom: spacing.md,
  },
  resultCard: {
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.md,
    padding: spacing.md,
  },
  resultTitle: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing.md,
  },
  resultMain: {
    alignItems: 'center',
    paddingVertical: spacing.md,
    backgroundColor: `${colors.brand.primary}20`,
    borderRadius: borderRadius.md,
    marginBottom: spacing.md,
  },
  clearingPrice: {
    fontSize: fontSize.xxxl,
    fontWeight: 'bold',
    color: colors.brand.primary,
  },
  resultLabel: {
    fontSize: fontSize.sm,
    color: colors.text.muted,
  },
  resultGrid: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  resultItem: {
    flex: 1,
    alignItems: 'center',
    padding: spacing.sm,
    backgroundColor: colors.background.tertiary,
    borderRadius: borderRadius.md,
  },
  resultValue: {
    fontSize: fontSize.xl,
    fontWeight: 'bold',
    color: colors.text.primary,
  },
  resultItemLabel: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    marginTop: 2,
  },
  successBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  successText: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.status.success,
  },
  failBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  failText: {
    fontSize: fontSize.lg,
    fontWeight: 'bold',
    color: colors.status.danger,
  },

  tableContainer: {
    margin: spacing.md,
    marginTop: 0,
    backgroundColor: colors.background.card,
    borderRadius: borderRadius.md,
    overflow: 'hidden',
  },
  tableHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    padding: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  tableTitle: {
    fontSize: fontSize.md,
    fontWeight: '600',
    color: colors.text.primary,
  },
  tableHeaderRow: {
    flexDirection: 'row',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: colors.background.tertiary,
  },
  tableHeaderCell: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
    fontWeight: '600',
  },
  tableRow: {
    flexDirection: 'row',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
  },
  tableRowOurs: {
    backgroundColor: `${colors.brand.primary}15`,
  },
  tableRowAccepted: {
    backgroundColor: `${colors.status.success}10`,
  },
  tableRowPartial: {
    backgroundColor: `${colors.status.warning}10`,
  },
  tableRowRejected: {
    backgroundColor: `${colors.status.danger}10`,
  },
  tableCell: {
    fontSize: fontSize.sm,
    color: colors.text.primary,
  },
  tableCellContainer: {
    justifyContent: 'center',
  },
  tableCellText: {
    fontSize: fontSize.sm,
    color: colors.text.primary,
  },
  oursBadge: {
    marginLeft: spacing.xs,
    paddingHorizontal: 4,
    paddingVertical: 1,
    backgroundColor: `${colors.brand.primary}30`,
    borderRadius: borderRadius.sm,
  },
  oursBadgeText: {
    fontSize: 8,
    color: colors.brand.primary,
    fontWeight: '600',
  },
  statusAccepted: {
    fontSize: fontSize.xs,
    color: colors.status.success,
    fontWeight: '600',
  },
  statusPartial: {
    fontSize: fontSize.xs,
    color: colors.status.warning,
    fontWeight: '600',
  },
  statusRejected: {
    fontSize: fontSize.xs,
    color: colors.status.danger,
    fontWeight: 'bold',
  },
  statusPending: {
    fontSize: fontSize.xs,
    color: colors.text.muted,
  },

  infoBox: {
    margin: spacing.md,
    marginTop: 0,
    backgroundColor: `${colors.brand.primary}15`,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: `${colors.brand.primary}30`,
  },
  infoTitle: {
    fontSize: fontSize.sm,
    fontWeight: '600',
    color: colors.brand.primary,
    marginBottom: spacing.sm,
  },
  infoText: {
    fontSize: fontSize.xs,
    color: colors.text.secondary,
    lineHeight: 18,
  },
});
