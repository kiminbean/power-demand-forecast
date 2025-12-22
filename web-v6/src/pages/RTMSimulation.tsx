/**
 * RTM (Real-Time Market) Simulation Page - RE-BMS v6.1.1
 * Realistic KPX real-time electricity market bid matching simulation
 *
 * RTM Characteristics vs DAM:
 * - Shorter time horizon (5-15 minutes before delivery)
 * - More volatile prices
 * - Smaller volumes (balancing purposes)
 * - Higher price volatility (+/- 20-30%)
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  RotateCcw,
  CheckCircle,
  XCircle,
  Zap,
  Building2,
  Factory,
  Users,
  Award,
  Clock,
  BarChart3,
  AlertTriangle,
  Info,
  TrendingUp,
  Activity,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { useTheme } from '../contexts/ThemeContext';
import clsx from 'clsx';

// ==================== Types ====================

interface MarketBid {
  id: string;
  bidder: string;
  type: 'supply' | 'demand';
  resourceType?: 'wind' | 'solar' | 'thermal' | 'ess' | 'hydro' | 'biomass';
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
  priceVolatility: number; // RTM-specific: price volatility indicator
}

interface ClearingResult {
  clearingPrice: number;
  clearingQuantity: number;
  intersectionPoint: { x: number; y: number };
  marginalSupplier: string | null;
  marginalDemander: string | null;
}

// ==================== RTM-Specific Data ====================

// RTM participants (subset of DAM participants + balancing resources)
const RTM_GENERATORS = [
  // Fast-response resources for RTM
  { name: '제주ESS1', type: 'ess' as const, capacity: 40, minPrice: 70, maxPrice: 100 },
  { name: '서귀포ESS', type: 'ess' as const, capacity: 30, minPrice: 75, maxPrice: 105 },
  { name: '한라풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 15 },
  { name: '탐라해상풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 18 },
  { name: '제주태양광1', type: 'solar' as const, capacity: 25, minPrice: 0, maxPrice: 20 },
  // Thermal for balancing
  { name: '제주화력#1', type: 'thermal' as const, capacity: 50, minPrice: 95, maxPrice: 130 },
  { name: '제주화력#2', type: 'thermal' as const, capacity: 50, minPrice: 98, maxPrice: 135 },
  { name: '남제주화력', type: 'thermal' as const, capacity: 60, minPrice: 100, maxPrice: 140 },
];

// RTM demand sources (real-time balancing needs)
const RTM_DEMAND_SOURCES = [
  { name: '계통운영(증가)', baseQuantity: 30, priceWillingness: 'high' as const },
  { name: '예비력확보', baseQuantity: 20, priceWillingness: 'high' as const },
  { name: '급전지시차이', baseQuantity: 15, priceWillingness: 'medium' as const },
  { name: '재생변동보상', baseQuantity: 25, priceWillingness: 'high' as const },
  { name: '수요예측오차', baseQuantity: 10, priceWillingness: 'medium' as const },
];

// RTM has higher price volatility based on time
const RTM_VOLATILITY_MULTIPLIER: Record<number, number> = {
  0: 1.15, 1: 1.10, 2: 1.05, 3: 1.05, 4: 1.10, 5: 1.20,
  6: 1.25, 7: 1.30, 8: 1.25, 9: 1.20, 10: 1.15, 11: 1.15,
  12: 1.20, 13: 1.15, 14: 1.20, 15: 1.25, 16: 1.30, 17: 1.35,
  18: 1.40, 19: 1.35, 20: 1.25, 21: 1.20, 22: 1.15, 23: 1.10,
};

// RTM demand is smaller but more volatile
const RTM_DEMAND_MULTIPLIER: Record<number, number> = {
  0: 0.60, 1: 0.55, 2: 0.50, 3: 0.50, 4: 0.55, 5: 0.70,
  6: 0.85, 7: 1.00, 8: 0.95, 9: 0.90, 10: 0.85, 11: 0.90,
  12: 0.95, 13: 0.90, 14: 1.00, 15: 1.05, 16: 1.10, 17: 1.20,
  18: 1.25, 19: 1.15, 20: 1.00, 21: 0.85, 22: 0.75, 23: 0.65,
};

// Solar availability (same as DAM)
const SOLAR_AVAILABILITY: Record<number, number> = {
  0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0.1,
  6: 0.3, 7: 0.5, 8: 0.7, 9: 0.85, 10: 0.95, 11: 1.00,
  12: 1.00, 13: 0.95, 14: 0.90, 15: 0.80, 16: 0.65, 17: 0.45,
  18: 0.25, 19: 0.1, 20: 0, 21: 0, 22: 0, 23: 0,
};

// ==================== Helper Functions ====================

const phaseLabels: Record<SimulationState['phase'], string> = {
  idle: '대기 중',
  collecting: '입찰 수집',
  sorting: 'Merit Order',
  matching: '수급 매칭',
  clearing: '청산가격',
  complete: '매칭 완료',
};

// Calculate clearing price using supply/demand intersection
function calculateClearingPrice(
  supplyBids: MarketBid[],
  demandBids: MarketBid[]
): ClearingResult | null {
  if (supplyBids.length === 0 || demandBids.length === 0) return null;

  // Sort supply by price ascending (merit order)
  const sortedSupply = [...supplyBids].sort((a, b) => a.price - b.price);
  // Sort demand by price descending (willingness to pay)
  const sortedDemand = [...demandBids].sort((a, b) => b.price - a.price);

  // Build supply curve (cumulative quantity vs price)
  const supplyCurve: { quantity: number; price: number; bidder: string }[] = [];
  let cumSupply = 0;
  sortedSupply.forEach(bid => {
    supplyCurve.push({ quantity: cumSupply, price: bid.price, bidder: bid.bidder });
    cumSupply += bid.quantity;
    supplyCurve.push({ quantity: cumSupply, price: bid.price, bidder: bid.bidder });
  });

  // Build demand curve
  const demandCurve: { quantity: number; price: number; bidder: string }[] = [];
  let cumDemand = 0;
  sortedDemand.forEach(bid => {
    demandCurve.push({ quantity: cumDemand, price: bid.price, bidder: bid.bidder });
    cumDemand += bid.quantity;
    demandCurve.push({ quantity: cumDemand, price: bid.price, bidder: bid.bidder });
  });

  // Find intersection point
  let clearingPrice = 0;
  let clearingQuantity = 0;
  let marginalSupplier: string | null = null;
  let marginalDemander: string | null = null;

  for (let q = 0; q <= Math.max(cumSupply, cumDemand); q += 0.5) {
    const supplyPoint = supplyCurve.find((_s, idx) =>
      idx < supplyCurve.length - 1 &&
      supplyCurve[idx].quantity <= q &&
      supplyCurve[idx + 1].quantity >= q
    );

    const demandPoint = demandCurve.find((_d, idx) =>
      idx < demandCurve.length - 1 &&
      demandCurve[idx].quantity <= q &&
      demandCurve[idx + 1].quantity >= q
    );

    if (supplyPoint && demandPoint) {
      if (supplyPoint.price <= demandPoint.price) {
        clearingQuantity = q;
        clearingPrice = supplyPoint.price;
        marginalSupplier = supplyPoint.bidder;
        marginalDemander = demandPoint.bidder;
      } else {
        break;
      }
    }
  }

  if (clearingQuantity === 0) return null;

  return {
    clearingPrice,
    clearingQuantity,
    intersectionPoint: { x: clearingQuantity, y: clearingPrice },
    marginalSupplier,
    marginalDemander,
  };
}

// Update bid statuses based on clearing
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

// ==================== Component ====================

export default function RTMSimulation() {
  const navigate = useNavigate();
  const location = useLocation();
  const { theme } = useTheme();

  // Get bid data from navigation state
  const bidData = location.state as {
    segments: { id: number; quantity: number; price: number }[];
    selectedHour: number;
    smpForecast: { q10: number; q50: number; q90: number };
  } | null;

  // Chart colors based on theme
  const chartColors = useMemo(() => ({
    grid: theme === 'dark' ? '#374151' : '#e5e7eb',
    axis: theme === 'dark' ? '#9ca3af' : '#6b7280',
    tooltipBg: theme === 'dark' ? '#1f2937' : '#ffffff',
    tooltipBorder: theme === 'dark' ? '#374151' : '#e5e7eb',
  }), [theme]);

  // Generate RTM supply bids
  const generateSupplyBids = useCallback((
    ourBids: { quantity: number; price: number }[],
    _smpForecast: { q10: number; q50: number; q90: number },
    hour: number
  ): MarketBid[] => {
    const bids: MarketBid[] = [];
    const volatility = RTM_VOLATILITY_MULTIPLIER[hour] || 1.0;
    const solarAvail = SOLAR_AVAILABILITY[hour] || 0;

    // Add RTM generator bids with volatility adjustment
    RTM_GENERATORS.forEach((gen, idx) => {
      let effectiveCapacity = gen.capacity;
      if (gen.type === 'solar') {
        effectiveCapacity = Math.round(gen.capacity * solarAvail);
      }
      if (effectiveCapacity <= 0) return;

      // RTM prices are more volatile
      const priceVariation = (Math.random() - 0.5) * 0.3 * volatility;
      const price = Math.round(
        (gen.minPrice + (gen.maxPrice - gen.minPrice) * (0.5 + priceVariation)) * volatility
      );

      bids.push({
        id: `rtm-gen-${idx}`,
        bidder: gen.name,
        type: 'supply',
        resourceType: gen.type,
        quantity: Math.round(effectiveCapacity * (0.6 + Math.random() * 0.4)),
        price: Math.max(0, Math.min(200, price)),
        status: 'pending',
      });
    });

    // Add our bids with RTM premium
    ourBids.forEach((bid, idx) => {
      bids.push({
        id: `rtm-our-${idx}`,
        bidder: '우리회사 (RTM)',
        type: 'supply',
        resourceType: 'ess',
        quantity: bid.quantity * 0.5, // RTM uses smaller quantities
        price: Math.round(bid.price * volatility * 1.1), // RTM premium
        isOurs: true,
        status: 'pending',
      });
    });

    return bids;
  }, []);

  // Generate RTM demand bids
  const generateDemandBids = useCallback((
    smpForecast: { q10: number; q50: number; q90: number },
    hour: number
  ): MarketBid[] => {
    const demandMultiplier = RTM_DEMAND_MULTIPLIER[hour] || 1.0;
    const volatility = RTM_VOLATILITY_MULTIPLIER[hour] || 1.0;

    const demanders = RTM_DEMAND_SOURCES.map((d, idx) => {
      const basePrice = d.priceWillingness === 'high' ? smpForecast.q90 * 1.3 :
                        d.priceWillingness === 'medium' ? smpForecast.q90 * 1.15 :
                        smpForecast.q50;

      return {
        id: `rtm-dem-${idx}`,
        bidder: d.name,
        type: 'demand' as const,
        quantity: Math.round(d.baseQuantity * demandMultiplier * (0.8 + Math.random() * 0.4)),
        price: Math.round(basePrice * volatility * (0.95 + Math.random() * 0.1)),
        status: 'pending' as const,
      };
    });

    return demanders;
  }, []);

  // State
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
    priceVolatility: 0,
  });

  const [supplyBids, setSupplyBids] = useState<MarketBid[]>([]);
  const [demandBids, setDemandBids] = useState<MarketBid[]>([]);
  const [_clearingResult, setClearingResult] = useState<ClearingResult | null>(null);

  // Initialize bids
  useEffect(() => {
    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      const supply = generateSupplyBids(ourBids, bidData.smpForecast, bidData.selectedHour);
      const demand = generateDemandBids(bidData.smpForecast, bidData.selectedHour);
      setSupplyBids(supply);
      setDemandBids(demand);
    }
  }, [bidData, generateSupplyBids, generateDemandBids]);

  // Build curve data for chart
  const round1 = (n: number) => Math.round(n * 10) / 10;

  const supplyCurveData = useMemo(() => {
    const data: { quantity: number; price: number }[] = [];
    let cumSupply = 0;
    [...supplyBids].sort((a, b) => a.price - b.price).forEach(bid => {
      data.push({ quantity: round1(cumSupply), price: bid.price });
      cumSupply += bid.quantity;
      data.push({ quantity: round1(cumSupply), price: bid.price });
    });
    return data;
  }, [supplyBids]);

  const demandCurveData = useMemo(() => {
    const data: { quantity: number; price: number }[] = [];
    let cumDemand = 0;
    [...demandBids].sort((a, b) => b.price - a.price).forEach(bid => {
      data.push({ quantity: round1(cumDemand), price: bid.price });
      cumDemand += bid.quantity;
      data.push({ quantity: round1(cumDemand), price: bid.price });
    });
    return data;
  }, [demandBids]);

  const curveData = useMemo(() => {
    const allQuantities = new Set<number>();
    supplyCurveData.forEach(d => allQuantities.add(d.quantity));
    demandCurveData.forEach(d => allQuantities.add(d.quantity));

    const sortedQuantities = Array.from(allQuantities).sort((a, b) => a - b);

    return sortedQuantities.map(q => {
      let supplyPrice: number | undefined;
      for (let i = supplyCurveData.length - 1; i >= 0; i--) {
        if (supplyCurveData[i].quantity <= q) {
          supplyPrice = supplyCurveData[i].price;
          break;
        }
      }

      let demandPrice: number | undefined;
      for (let i = demandCurveData.length - 1; i >= 0; i--) {
        if (demandCurveData[i].quantity <= q) {
          demandPrice = demandCurveData[i].price;
          break;
        }
      }

      return { quantity: round1(q), supplyPrice, demandPrice };
    });
  }, [supplyCurveData, demandCurveData]);

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
            setClearingResult(result);

            if (result) {
              const updatedSupply = updateBidStatuses(
                supplyBids,
                result.clearingPrice,
                result.clearingQuantity,
                'supply'
              );
              const updatedDemand = updateBidStatuses(
                demandBids,
                result.clearingPrice,
                result.clearingQuantity,
                'demand'
              );

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

              const volatility = bidData ? RTM_VOLATILITY_MULTIPLIER[bidData.selectedHour] || 1.0 : 1.0;

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
                priceVolatility: Math.round((volatility - 1) * 100),
              };
            }

            return { ...prev, phase: 'complete', progress: 100 };
          }
          return { ...prev, phase: phases[currentPhaseIdx], progress: 0 };
        }
        return { ...prev, progress: prev.progress + 8 };
      });
    }, 150);

    setSimulation(prev => ({ ...prev, phase: 'collecting', progress: 0 }));

    return () => clearInterval(interval);
  }, [supplyBids, demandBids, bidData]);

  // Reset simulation
  const resetSimulation = useCallback(() => {
    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      setSupplyBids(generateSupplyBids(ourBids, bidData.smpForecast, bidData.selectedHour));
      setDemandBids(generateDemandBids(bidData.smpForecast, bidData.selectedHour));
    }
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
      priceVolatility: 0,
    });
    setClearingResult(null);
  }, [bidData, generateSupplyBids, generateDemandBids]);

  // No bid data - show message
  if (!bidData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <XCircle className="w-16 h-16 text-danger mx-auto mb-4" />
          <h2 className="text-xl font-bold text-text-primary mb-2">입찰 데이터가 없습니다</h2>
          <p className="text-text-muted mb-4">입찰 관리 페이지에서 RTM 제출을 진행해주세요.</p>
          <button
            onClick={() => navigate('/bidding')}
            className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 inline mr-2" />
            입찰 관리로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  const totalSupplyBids = supplyBids.reduce((sum, b) => sum + b.quantity, 0);
  const totalDemandBids = demandBids.reduce((sum, b) => sum + b.quantity, 0);
  const ourTotalBid = supplyBids.filter(b => b.isOurs).reduce((sum, b) => sum + b.quantity, 0);
  const ourAvgPrice = supplyBids.filter(b => b.isOurs).reduce((sum, b) => sum + b.price * b.quantity, 0) /
    (ourTotalBid || 1);
  const volatility = RTM_VOLATILITY_MULTIPLIER[bidData.selectedHour] || 1.0;

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/bidding')}
            className="p-2 hover:bg-card rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-text-muted" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-text-primary flex items-center gap-3">
              <Activity className="w-7 h-7 text-warning" />
              KPX 실시간시장 시뮬레이션
            </h1>
            <p className="text-text-muted mt-1">
              {String(bidData.selectedHour).padStart(2, '0')}:00 - {String(bidData.selectedHour + 1).padStart(2, '0')}:00 거래시간 | SMP 예측: {bidData.smpForecast.q50.toFixed(0)}원/kWh
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {simulation.phase === 'idle' ? (
            <button
              onClick={runSimulation}
              className="px-6 py-2.5 bg-warning text-white rounded-lg hover:bg-warning/90 transition-colors flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              시뮬레이션 시작
            </button>
          ) : (
            <button
              onClick={resetSimulation}
              className="px-6 py-2.5 bg-card border border-border text-text-primary rounded-lg hover:bg-background transition-colors flex items-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              다시 실행
            </button>
          )}
        </div>
      </div>

      {/* RTM Volatility Alert */}
      <div className={clsx(
        'card p-4 border-l-4',
        volatility >= 1.3 ? 'border-l-danger bg-danger/5' :
        volatility >= 1.2 ? 'border-l-warning bg-warning/5' :
        'border-l-success bg-success/5'
      )}>
        <div className="flex items-center gap-3">
          <Activity className={clsx(
            'w-5 h-5',
            volatility >= 1.3 ? 'text-danger' :
            volatility >= 1.2 ? 'text-warning' :
            'text-success'
          )} />
          <div>
            <span className="font-semibold text-text-primary">RTM 가격 변동성: </span>
            <span className={clsx(
              'font-bold',
              volatility >= 1.3 ? 'text-danger' :
              volatility >= 1.2 ? 'text-warning' :
              'text-success'
            )}>
              {((volatility - 1) * 100).toFixed(0)}%
            </span>
            <span className="text-text-muted ml-2">
              {volatility >= 1.3 ? '(고변동 - 피크시간대)' :
               volatility >= 1.2 ? '(중변동)' :
               '(저변동 - 안정적)'}
            </span>
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <Factory className="w-4 h-4" />
            총 공급
          </div>
          <div className="text-2xl font-bold text-success">
            {totalSupplyBids.toFixed(0)} MW
          </div>
          <div className="text-xs text-text-muted mt-1">
            {supplyBids.length}개 발전사
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <Users className="w-4 h-4" />
            총 수요
          </div>
          <div className="text-2xl font-bold text-warning">
            {totalDemandBids.toFixed(0)} MW
          </div>
          <div className="text-xs text-text-muted mt-1">
            {demandBids.length}개 수요처
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <TrendingUp className="w-4 h-4" />
            우리 입찰
          </div>
          <div className="text-2xl font-bold text-primary">
            {ourTotalBid.toFixed(1)} MW
          </div>
          <div className="text-xs text-text-muted mt-1">
            평균 {ourAvgPrice.toFixed(0)}원/kWh
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <Clock className="w-4 h-4" />
            수요배수
          </div>
          <div className="text-2xl font-bold text-text-primary">
            ×{RTM_DEMAND_MULTIPLIER[bidData.selectedHour]?.toFixed(2) || '1.00'}
          </div>
          <div className="text-xs text-text-muted mt-1">
            {bidData.selectedHour >= 17 && bidData.selectedHour <= 19 ? '피크시간' :
             bidData.selectedHour >= 7 && bidData.selectedHour <= 9 ? '오전피크' : '일반시간'}
          </div>
        </div>
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <Activity className="w-4 h-4" />
            변동성
          </div>
          <div className={clsx(
            'text-2xl font-bold',
            volatility >= 1.3 ? 'text-danger' :
            volatility >= 1.2 ? 'text-warning' :
            'text-success'
          )}>
            +{((volatility - 1) * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-text-muted mt-1">
            RTM 프리미엄
          </div>
        </div>
      </div>

      {/* Phase Progress */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <Clock className="w-5 h-5 text-warning" />
            처리 단계
          </h3>
          <span className={clsx(
            'px-3 py-1 rounded-full text-sm font-medium',
            simulation.phase === 'complete' ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'
          )}>
            {phaseLabels[simulation.phase]}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {(['collecting', 'sorting', 'matching', 'clearing', 'complete'] as const).map((phase, idx) => {
            const phases = ['collecting', 'sorting', 'matching', 'clearing', 'complete'];
            const currentIdx = phases.indexOf(simulation.phase);
            return (
              <div key={phase} className="flex-1 flex items-center">
                <div className={clsx(
                  'flex-1 h-2 rounded-full transition-colors duration-300',
                  currentIdx === idx ? 'bg-warning' : currentIdx > idx ? 'bg-success' : 'bg-background'
                )}>
                  {simulation.phase === phase && (
                    <div
                      className="h-full bg-warning rounded-full transition-all duration-100"
                      style={{ width: `${simulation.progress}%`, opacity: 0.6 }}
                    />
                  )}
                </div>
                {idx < 4 && <div className="w-2" />}
              </div>
            );
          })}
        </div>
        <div className="flex justify-between mt-2 text-xs text-text-muted">
          <span>입찰 수집</span>
          <span>Merit Order</span>
          <span>수급 매칭</span>
          <span>청산가격</span>
          <span>완료</span>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Supply/Demand Curve Chart */}
        <div className="lg:col-span-2 card">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-warning" />
            수요/공급 곡선 (RTM Merit Order)
          </h3>
          <div className="h-[420px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={curveData} margin={{ top: 20, right: 30, left: 10, bottom: 35 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis
                  dataKey="quantity"
                  stroke={chartColors.axis}
                  fontSize={11}
                  tickMargin={5}
                  tickFormatter={(value) => Math.round(value).toString()}
                  label={{ value: '누적 물량 (MW)', position: 'bottom', offset: 0, fill: chartColors.axis, fontSize: 12 }}
                />
                <YAxis
                  stroke={chartColors.axis}
                  fontSize={12}
                  width={50}
                  domain={[0, 200]}
                  label={{ value: '가격 (원/kWh)', angle: -90, position: 'insideLeft', fill: chartColors.axis, dx: -5 }}
                />
                <Tooltip
                  formatter={(value: number, name: string) => [
                    `${value?.toFixed(1) || '-'}원/kWh`,
                    name === 'supplyPrice' ? '공급가격' : '수요가격'
                  ]}
                  labelFormatter={(label) => `물량: ${label} MW`}
                  contentStyle={{
                    backgroundColor: chartColors.tooltipBg,
                    border: `1px solid ${chartColors.tooltipBorder}`,
                    borderRadius: '8px',
                  }}
                />

                {/* Supply curve - green */}
                <Line
                  type="stepAfter"
                  dataKey="supplyPrice"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="supplyPrice"
                  dot={false}
                  connectNulls={true}
                />

                {/* Demand curve - orange */}
                <Line
                  type="stepAfter"
                  dataKey="demandPrice"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  name="demandPrice"
                  dot={false}
                  connectNulls={true}
                />

                {/* Clearing price reference line */}
                {simulation.clearingPrice && (
                  <ReferenceLine
                    y={simulation.clearingPrice}
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="8 4"
                    label={{
                      value: `RTM SMP: ${simulation.clearingPrice.toFixed(1)}원`,
                      position: 'right',
                      fill: '#ef4444',
                      fontSize: 13,
                      fontWeight: 600,
                    }}
                  />
                )}

                {/* Clearing quantity reference line */}
                {simulation.clearingQuantity && (
                  <ReferenceLine
                    x={simulation.clearingQuantity}
                    stroke="#6366f1"
                    strokeWidth={2}
                    strokeDasharray="8 4"
                    label={{
                      value: `${simulation.clearingQuantity.toFixed(0)}MW`,
                      position: 'top',
                      fill: '#6366f1',
                      fontSize: 12,
                    }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Chart Legend */}
          <div className="flex flex-wrap items-center justify-center gap-6 mt-4 pt-4 border-t border-border">
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-success rounded" />
              <span className="text-sm text-text-muted">공급 곡선</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-1 bg-warning rounded" />
              <span className="text-sm text-text-muted">수요 곡선</span>
            </div>
            {simulation.clearingPrice && (
              <>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-0.5 bg-danger rounded" style={{ borderStyle: 'dashed' }} />
                  <span className="text-sm text-text-muted">청산가격 (RTM SMP)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-6 h-0.5 bg-primary rounded" style={{ borderStyle: 'dashed' }} />
                  <span className="text-sm text-text-muted">거래량</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {/* Clearing Results */}
          <div className="card">
            <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-warning" />
              청산 결과
            </h3>
            {simulation.phase === 'complete' && simulation.clearingPrice ? (
              <div className="space-y-4">
                <div className="text-center p-4 bg-warning/10 rounded-lg">
                  <div className="text-3xl font-bold text-warning">
                    {simulation.clearingPrice.toFixed(1)}원
                  </div>
                  <div className="text-sm text-text-muted">청산가격 (RTM SMP)</div>
                  <div className="text-xs text-warning mt-1">
                    예측 대비 +{((simulation.clearingPrice / bidData.smpForecast.q50 - 1) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center p-3 bg-background rounded-lg">
                    <div className="text-xl font-bold text-primary">
                      {simulation.clearingQuantity?.toFixed(0) || 0}
                    </div>
                    <div className="text-xs text-text-muted">거래량 (MW)</div>
                  </div>
                  <div className="text-center p-3 bg-background rounded-lg">
                    <div className="text-xl font-bold text-text-primary">
                      {supplyBids.filter(b => b.status === 'accepted' || b.status === 'partial').length}
                    </div>
                    <div className="text-xs text-text-muted">낙찰 발전사</div>
                  </div>
                </div>
                {/* Market Type with Volatility */}
                <div className={clsx(
                  'p-3 rounded-lg flex items-center gap-2',
                  simulation.marketType === 'normal' ? 'bg-success/10' :
                  simulation.marketType === 'oversupply' ? 'bg-sky-500/10' : 'bg-danger/10'
                )}>
                  {simulation.marketType === 'normal' ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : simulation.marketType === 'oversupply' ? (
                    <Info className="w-4 h-4 text-sky-500" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-danger" />
                  )}
                  <span className={clsx(
                    'text-sm font-medium',
                    simulation.marketType === 'normal' ? 'text-success' :
                    simulation.marketType === 'oversupply' ? 'text-sky-500' : 'text-danger'
                  )}>
                    {simulation.marketType === 'normal' ? '정상 시장' :
                     simulation.marketType === 'oversupply' ? '공급 과잉' : '공급 부족'}
                  </span>
                  <span className="text-xs text-text-muted ml-auto">
                    변동성 +{simulation.priceVolatility}%
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-text-muted">
                <Activity className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>시뮬레이션 완료 후 결과가 표시됩니다</p>
              </div>
            )}
          </div>

          {/* Our Bid Result */}
          <div className={clsx(
            'card border-2',
            simulation.phase === 'complete' && simulation.ourAccepted > 0
              ? 'border-success'
              : 'border-transparent'
          )}>
            <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Award className="w-5 h-5 text-warning" />
              우리 입찰 결과
            </h3>
            {simulation.phase === 'complete' ? (
              <div className="space-y-4">
                {simulation.ourAccepted > 0 ? (
                  <>
                    <div className="flex items-center justify-center gap-2 text-success">
                      <CheckCircle className="w-6 h-6" />
                      <span className="text-lg font-bold">낙찰 성공!</span>
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="text-center p-3 bg-success/10 rounded-lg">
                        <div className="text-xl font-bold text-success">
                          {simulation.ourAccepted.toFixed(1)} MW
                        </div>
                        <div className="text-xs text-text-muted">낙찰 물량</div>
                        <div className="text-xs text-success">
                          ({((simulation.ourAccepted / ourTotalBid) * 100).toFixed(0)}%)
                        </div>
                      </div>
                      <div className="text-center p-3 bg-success/10 rounded-lg">
                        <div className="text-xl font-bold text-success">
                          {(simulation.ourRevenue / 1000).toFixed(1)}천원
                        </div>
                        <div className="text-xs text-text-muted">예상 수익</div>
                        <div className="text-xs text-text-muted">(시간당)</div>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center py-4">
                    <XCircle className="w-12 h-12 text-danger mx-auto mb-2" />
                    <p className="text-danger font-medium">낙찰 실패</p>
                    <p className="text-sm text-text-muted mt-1">
                      입찰가격이 RTM 청산가격보다 높습니다
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-text-muted">
                <Building2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>시뮬레이션 완료 후 결과가 표시됩니다</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RTM Explanation */}
      <div className="card bg-warning/5 border border-warning/20">
        <h3 className="text-lg font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Info className="w-5 h-5 text-warning" />
          KPX 실시간시장(RTM) 입찰 매칭 원리
        </h3>
        <div className="text-sm text-text-muted space-y-2">
          <p>
            한국전력거래소(KPX)의 <strong>실시간시장(RTM)</strong>은 하루전시장(DAM) 거래 이후
            실시간 수급 불균형을 해소하기 위한 보조 시장입니다.
          </p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>거래 시점</strong>: 실제 전력 공급 5~15분 전</li>
            <li><strong>가격 변동성</strong>: DAM 대비 10~40% 높은 변동성</li>
            <li><strong>주요 참여자</strong>: ESS, 빠른 응답 가능한 발전원, 계통운영자</li>
            <li><strong>거래 목적</strong>: 재생에너지 변동 보상, 수요예측 오차 보정, 예비력 확보</li>
            <li><strong>청산 방식</strong>: DAM과 동일한 Merit Order 방식 (한계가격 결정)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
