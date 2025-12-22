/**
 * KPX Market Simulation Page - RE-BMS v6.1.1
 * Realistic KPX electricity market bid matching simulation
 *
 * Improvements:
 * - Dynamic clearing price calculation (supply/demand intersection)
 * - Realistic Jeju power market data
 * - Time-based demand variation
 * - Partial acceptance logic
 * - SMP forecast-based market simulation
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
  Wind,
  Sun,
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
}

interface ClearingResult {
  clearingPrice: number;
  clearingQuantity: number;
  intersectionPoint: { x: number; y: number };
  marginalSupplier: string | null;
  marginalDemander: string | null;
}

// ==================== Jeju Power Market Data ====================

// Realistic Jeju power generators (based on actual Jeju grid data)
const JEJU_GENERATORS = [
  // Renewable Energy (lowest marginal cost - priority dispatch)
  { name: '한라풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 10 },
  { name: '탐라해상풍력', type: 'wind' as const, capacity: 30, minPrice: 0, maxPrice: 12 },
  { name: '김녕풍력', type: 'wind' as const, capacity: 15, minPrice: 0, maxPrice: 8 },
  { name: '가시리풍력', type: 'wind' as const, capacity: 20, minPrice: 0, maxPrice: 11 },
  { name: '행원풍력', type: 'wind' as const, capacity: 10, minPrice: 0, maxPrice: 9 },
  { name: '제주태양광1', type: 'solar' as const, capacity: 25, minPrice: 0, maxPrice: 15 },
  { name: '제주태양광2', type: 'solar' as const, capacity: 20, minPrice: 0, maxPrice: 18 },
  { name: '서귀포태양광', type: 'solar' as const, capacity: 15, minPrice: 0, maxPrice: 16 },
  // ESS (flexible pricing)
  { name: '제주ESS1', type: 'ess' as const, capacity: 40, minPrice: 60, maxPrice: 85 },
  { name: '서귀포ESS', type: 'ess' as const, capacity: 30, minPrice: 65, maxPrice: 90 },
  // Thermal (highest marginal cost)
  { name: '제주화력#1', type: 'thermal' as const, capacity: 75, minPrice: 85, maxPrice: 110 },
  { name: '제주화력#2', type: 'thermal' as const, capacity: 75, minPrice: 88, maxPrice: 115 },
  { name: '남제주화력', type: 'thermal' as const, capacity: 100, minPrice: 90, maxPrice: 120 },
  { name: 'GT(비상)', type: 'thermal' as const, capacity: 40, minPrice: 130, maxPrice: 180 },
];

// Jeju major demand sources
const JEJU_DEMAND_SOURCES = [
  { name: '제주공항', baseQuantity: 25, priceWillingness: 'high' as const },
  { name: '삼성SDI', baseQuantity: 45, priceWillingness: 'high' as const },
  { name: '롯데리조트', baseQuantity: 20, priceWillingness: 'medium' as const },
  { name: '한라시멘트', baseQuantity: 35, priceWillingness: 'medium' as const },
  { name: '제주시', baseQuantity: 80, priceWillingness: 'medium' as const },
  { name: '서귀포시', baseQuantity: 50, priceWillingness: 'medium' as const },
  { name: '제주대학교', baseQuantity: 15, priceWillingness: 'low' as const },
  { name: '농업용전력', baseQuantity: 30, priceWillingness: 'low' as const },
  { name: '일반가정', baseQuantity: 100, priceWillingness: 'low' as const },
];

// Time-based demand multiplier (hourly pattern)
const HOURLY_DEMAND_MULTIPLIER: Record<number, number> = {
  0: 0.65, 1: 0.60, 2: 0.58, 3: 0.55, 4: 0.55, 5: 0.60,
  6: 0.70, 7: 0.85, 8: 0.95, 9: 1.05, 10: 1.10, 11: 1.12,
  12: 1.08, 13: 1.10, 14: 1.15, 15: 1.12, 16: 1.08, 17: 1.05,
  18: 1.10, 19: 1.15, 20: 1.10, 21: 1.00, 22: 0.85, 23: 0.75,
};

// Solar availability by hour
const SOLAR_AVAILABILITY: Record<number, number> = {
  0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0.05,
  6: 0.15, 7: 0.35, 8: 0.55, 9: 0.75, 10: 0.90, 11: 0.98,
  12: 1.00, 13: 0.98, 14: 0.92, 15: 0.80, 16: 0.60, 17: 0.40,
  18: 0.20, 19: 0.05, 20: 0, 21: 0, 22: 0, 23: 0,
};

// ==================== Market Simulation Functions ====================

/**
 * Generate supply bids based on SMP forecast and hour
 */
const generateSupplyBids = (
  ourBids: { quantity: number; price: number }[],
  smpForecast: { q10: number; q50: number; q90: number },
  hour: number
): MarketBid[] => {
  const bids: MarketBid[] = [];
  const solarFactor = SOLAR_AVAILABILITY[hour] || 0;
  const windFactor = 0.3 + Math.random() * 0.5; // 30-80% wind availability

  // Generate bids for each generator
  JEJU_GENERATORS.forEach((gen, idx) => {
    let quantity = gen.capacity;
    let price = gen.minPrice;

    // Adjust based on resource type
    if (gen.type === 'solar') {
      quantity *= solarFactor;
      // Solar bids at very low price (REC incentive)
      price = gen.minPrice + Math.random() * (gen.maxPrice - gen.minPrice) * 0.3;
    } else if (gen.type === 'wind') {
      quantity *= windFactor;
      // Wind bids at very low price
      price = gen.minPrice + Math.random() * (gen.maxPrice - gen.minPrice) * 0.3;
    } else if (gen.type === 'ess') {
      // ESS price based on SMP forecast (arbitrage)
      price = smpForecast.q10 + Math.random() * (smpForecast.q50 - smpForecast.q10) * 0.5;
    } else if (gen.type === 'thermal') {
      // Thermal price based on fuel cost + margin
      const margin = Math.random() * 20;
      price = gen.minPrice + margin;
    }

    if (quantity > 0.5) {  // Only add if meaningful quantity
      bids.push({
        id: `supply-${idx}`,
        bidder: gen.name,
        type: 'supply',
        resourceType: gen.type,
        quantity: Math.round(quantity * 10) / 10,
        price: Math.round(price * 10) / 10,
        status: 'pending',
      });
    }
  });

  // Add our bids
  ourBids.forEach((bid, bidIdx) => {
    bids.push({
      id: `our-${bidIdx}`,
      bidder: 'eXeco (우리)',
      type: 'supply',
      resourceType: 'wind',
      quantity: bid.quantity,
      price: bid.price,
      isOurs: true,
      status: 'pending',
    });
  });

  // Sort by price (Merit Order)
  return bids.sort((a, b) => a.price - b.price);
};

/**
 * Generate demand bids based on hour and SMP forecast
 */
const generateDemandBids = (
  smpForecast: { q10: number; q50: number; q90: number },
  hour: number
): MarketBid[] => {
  const demandMultiplier = HOURLY_DEMAND_MULTIPLIER[hour] || 1.0;

  const priceMap = {
    high: { min: smpForecast.q90 * 1.1, max: smpForecast.q90 * 1.5 },
    medium: { min: smpForecast.q50, max: smpForecast.q90 * 1.1 },
    low: { min: smpForecast.q10 * 0.9, max: smpForecast.q50 },
  };

  return JEJU_DEMAND_SOURCES.map((source, idx): MarketBid => {
    const priceRange = priceMap[source.priceWillingness];
    const quantity = source.baseQuantity * demandMultiplier * (0.9 + Math.random() * 0.2);
    const price = priceRange.min + Math.random() * (priceRange.max - priceRange.min);

    return {
      id: `demand-${idx}`,
      bidder: source.name,
      type: 'demand',
      quantity: Math.round(quantity * 10) / 10,
      price: Math.round(price * 10) / 10,
      status: 'pending',
    };
  }).sort((a, b) => b.price - a.price);  // Sort by price descending
};

/**
 * Calculate clearing price by finding supply/demand intersection
 * This is the core algorithm for realistic market simulation
 */
const calculateClearingPrice = (
  supplyBids: MarketBid[],
  demandBids: MarketBid[]
): ClearingResult | null => {
  if (supplyBids.length === 0 || demandBids.length === 0) return null;

  // Build cumulative supply curve
  const supplyCurve: { quantity: number; price: number; bidder: string }[] = [];
  let cumSupply = 0;
  supplyBids.forEach(bid => {
    supplyCurve.push({ quantity: cumSupply, price: bid.price, bidder: bid.bidder });
    cumSupply += bid.quantity;
    supplyCurve.push({ quantity: cumSupply, price: bid.price, bidder: bid.bidder });
  });

  // Build cumulative demand curve
  const demandCurve: { quantity: number; price: number; bidder: string }[] = [];
  let cumDemand = 0;
  demandBids.forEach(bid => {
    demandCurve.push({ quantity: cumDemand, price: bid.price, bidder: bid.bidder });
    cumDemand += bid.quantity;
    demandCurve.push({ quantity: cumDemand, price: bid.price, bidder: bid.bidder });
  });

  // Find intersection point
  // Supply curve: step function increasing with quantity
  // Demand curve: step function decreasing with quantity

  let clearingPrice = 0;
  let clearingQuantity = 0;
  let marginalSupplier: string | null = null;
  let marginalDemander: string | null = null;

  // Method: Find where supply price >= demand price at same quantity
  for (let q = 0; q <= Math.max(cumSupply, cumDemand); q += 0.5) {
    // Get supply price at quantity q
    const supplyPoint = supplyCurve.find((_s, idx) =>
      idx < supplyCurve.length - 1 &&
      supplyCurve[idx].quantity <= q &&
      supplyCurve[idx + 1].quantity >= q
    );

    // Get demand price at quantity q
    const demandPoint = demandCurve.find((_d, idx) =>
      idx < demandCurve.length - 1 &&
      demandCurve[idx].quantity <= q &&
      demandCurve[idx + 1].quantity >= q
    );

    if (supplyPoint && demandPoint) {
      // At this quantity, if supply price <= demand price, market can clear
      if (supplyPoint.price <= demandPoint.price) {
        clearingQuantity = q;
        // Clearing price is typically set at marginal supply price
        clearingPrice = supplyPoint.price;
        marginalSupplier = supplyPoint.bidder;
        marginalDemander = demandPoint.bidder;
      } else {
        // Found the point where supply exceeds demand
        break;
      }
    }
  }

  // If no intersection found within range, check edge cases
  if (clearingQuantity === 0) {
    // Check if minimum supply price > maximum demand price (no clearing possible)
    const minSupplyPrice = Math.min(...supplyBids.map(b => b.price));
    const maxDemandPrice = Math.max(...demandBids.map(b => b.price));

    if (minSupplyPrice > maxDemandPrice) {
      // Market cannot clear - shortage
      clearingPrice = maxDemandPrice;
      clearingQuantity = 0;
    } else {
      // Take the first intersection approximation
      clearingPrice = (minSupplyPrice + maxDemandPrice) / 2;
      clearingQuantity = Math.min(cumSupply, cumDemand) * 0.8;
    }
  }

  return {
    clearingPrice: Math.round(clearingPrice * 10) / 10,
    clearingQuantity: Math.round(clearingQuantity * 10) / 10,
    intersectionPoint: { x: clearingQuantity, y: clearingPrice },
    marginalSupplier,
    marginalDemander,
  };
};

/**
 * Update bid statuses based on clearing price
 * Handles partial acceptance for marginal units
 */
const updateBidStatuses = (
  bids: MarketBid[],
  clearingPrice: number,
  clearingQuantity: number,
  bidType: 'supply' | 'demand'
): MarketBid[] => {
  const sortedBids = [...bids].sort((a, b) =>
    bidType === 'supply' ? a.price - b.price : b.price - a.price
  );

  let cumQuantity = 0;

  return sortedBids.map(bid => {
    const newBid = { ...bid };

    if (bidType === 'supply') {
      if (bid.price <= clearingPrice && cumQuantity < clearingQuantity) {
        const remainingCapacity = clearingQuantity - cumQuantity;
        if (bid.quantity <= remainingCapacity) {
          newBid.status = 'accepted';
          newBid.acceptedQuantity = bid.quantity;
          cumQuantity += bid.quantity;
        } else {
          // Partial acceptance
          newBid.status = 'partial';
          newBid.acceptedQuantity = remainingCapacity;
          cumQuantity += remainingCapacity;
        }
      } else {
        newBid.status = 'rejected';
        newBid.acceptedQuantity = 0;
      }
    } else {  // demand
      if (bid.price >= clearingPrice && cumQuantity < clearingQuantity) {
        const remainingCapacity = clearingQuantity - cumQuantity;
        if (bid.quantity <= remainingCapacity) {
          newBid.status = 'accepted';
          newBid.acceptedQuantity = bid.quantity;
          cumQuantity += bid.quantity;
        } else {
          newBid.status = 'partial';
          newBid.acceptedQuantity = remainingCapacity;
          cumQuantity += remainingCapacity;
        }
      } else {
        newBid.status = 'rejected';
        newBid.acceptedQuantity = 0;
      }
    }

    return newBid;
  });
};

// ==================== Component ====================

export default function KPXSimulation() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDark } = useTheme();

  const bidData = location.state as {
    segments: { id: number; quantity: number; price: number }[];
    selectedHour: number;
    smpForecast: { q10: number; q50: number; q90: number };
  } | null;

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
  }, [bidData]);

  // Build curve data for chart
  // Generate separate curve data for supply and demand (no nulls, continuous lines)
  const supplyCurveData = useMemo(() => {
    const data: { quantity: number; price: number }[] = [];
    let cumSupply = 0;
    [...supplyBids].sort((a, b) => a.price - b.price).forEach(bid => {
      data.push({ quantity: cumSupply, price: bid.price });
      cumSupply += bid.quantity;
      data.push({ quantity: cumSupply, price: bid.price });
    });
    return data;
  }, [supplyBids]);

  const demandCurveData = useMemo(() => {
    const data: { quantity: number; price: number }[] = [];
    let cumDemand = 0;
    [...demandBids].sort((a, b) => b.price - a.price).forEach(bid => {
      data.push({ quantity: cumDemand, price: bid.price });
      cumDemand += bid.quantity;
      data.push({ quantity: cumDemand, price: bid.price });
    });
    return data;
  }, [demandBids]);

  // Merge curve data for chart (with proper alignment)
  const curveData = useMemo(() => {
    // Get all unique quantities
    const allQuantities = new Set<number>();
    supplyCurveData.forEach(d => allQuantities.add(d.quantity));
    demandCurveData.forEach(d => allQuantities.add(d.quantity));

    const sortedQuantities = Array.from(allQuantities).sort((a, b) => a - b);

    // Build merged data with interpolated values
    return sortedQuantities.map(q => {
      // Find supply price at this quantity (step function)
      let supplyPrice: number | undefined;
      for (let i = supplyCurveData.length - 1; i >= 0; i--) {
        if (supplyCurveData[i].quantity <= q) {
          supplyPrice = supplyCurveData[i].price;
          break;
        }
      }

      // Find demand price at this quantity (step function)
      let demandPrice: number | undefined;
      for (let i = demandCurveData.length - 1; i >= 0; i--) {
        if (demandCurveData[i].quantity <= q) {
          demandPrice = demandCurveData[i].price;
          break;
        }
      }

      return { quantity: q, supplyPrice, demandPrice };
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

            // Calculate clearing price dynamically
            const result = calculateClearingPrice(supplyBids, demandBids);
            setClearingResult(result);

            if (result) {
              // Update bid statuses
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

              // Calculate our results
              const ourAccepted = updatedSupply
                .filter(b => b.isOurs && (b.status === 'accepted' || b.status === 'partial'))
                .reduce((sum, b) => sum + (b.acceptedQuantity || 0), 0);

              const totalSupply = supplyBids.reduce((sum, b) => sum + b.quantity, 0);
              const totalDemand = demandBids.reduce((sum, b) => sum + b.quantity, 0);

              // Determine market type
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
        return { ...prev, progress: prev.progress + 4 };
      });
    }, 80);

    return () => clearInterval(interval);
  }, [supplyBids, demandBids]);

  const handlePlay = () => {
    // Regenerate bids for new simulation
    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      setSupplyBids(generateSupplyBids(ourBids, bidData.smpForecast, bidData.selectedHour));
      setDemandBids(generateDemandBids(bidData.smpForecast, bidData.selectedHour));
    }

    setSimulation({
      phase: 'collecting',
      progress: 0,
      clearingPrice: null,
      clearingQuantity: null,
      totalSupply: 0,
      totalDemand: 0,
      ourAccepted: 0,
      ourRevenue: 0,
      marketType: 'normal',
    });
    setClearingResult(null);

    setTimeout(() => runSimulation(), 100);
  };

  const handleReset = () => {
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
    setClearingResult(null);

    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      setSupplyBids(generateSupplyBids(ourBids, bidData.smpForecast, bidData.selectedHour));
      setDemandBids(generateDemandBids(bidData.smpForecast, bidData.selectedHour));
    }
  };

  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1e2530' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
  };

  const phaseLabels: Record<SimulationState['phase'], string> = {
    idle: '대기 중',
    collecting: '입찰 수집',
    sorting: 'Merit Order 정렬',
    matching: '수급 매칭',
    clearing: '청산가격 결정',
    complete: '매칭 완료',
  };

  // Resource type icon
  const ResourceIcon = ({ type }: { type?: string }) => {
    switch (type) {
      case 'wind': return <Wind className="w-3 h-3 text-sky-500" />;
      case 'solar': return <Sun className="w-3 h-3 text-amber-500" />;
      default: return null;
    }
  };

  if (!bidData) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] space-y-4">
        <XCircle className="w-16 h-16 text-danger" />
        <h2 className="text-xl font-bold text-text-primary">입찰 데이터가 없습니다</h2>
        <p className="text-text-muted">입찰 관리 페이지에서 KPX 제출을 진행해주세요.</p>
        <button
          onClick={() => navigate('/bidding')}
          className="btn-primary flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          입찰 관리로 돌아가기
        </button>
      </div>
    );
  }

  const ourTotalBid = bidData.segments.reduce((sum, s) => sum + s.quantity, 0);
  const ourAvgPrice = bidData.segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / ourTotalBid;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate('/bidding')}
            className="p-2 hover:bg-background rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-text-muted" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-text-primary flex items-center gap-2">
              <Building2 className="w-7 h-7 text-primary" />
              KPX 하루전시장 시뮬레이션
            </h1>
            <p className="text-text-muted mt-1">
              {String(bidData.selectedHour).padStart(2, '0')}:00 - {String(bidData.selectedHour + 1).padStart(2, '0')}:00
              거래시간 | SMP 예측: {bidData.smpForecast.q50.toFixed(0)}원/kWh
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {simulation.phase === 'idle' ? (
            <button onClick={handlePlay} className="btn-primary flex items-center gap-2">
              <Play className="w-4 h-4" />
              시뮬레이션 시작
            </button>
          ) : simulation.phase === 'complete' ? (
            <button onClick={handleReset} className="btn-secondary flex items-center gap-2">
              <RotateCcw className="w-4 h-4" />
              다시 실행
            </button>
          ) : (
            <button disabled className="btn-secondary flex items-center gap-2 opacity-50">
              <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
              진행 중...
            </button>
          )}
        </div>
      </div>

      {/* Market Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card p-4">
          <div className="flex items-center gap-2 text-text-muted text-sm mb-1">
            <Factory className="w-4 h-4" />
            총 공급
          </div>
          <div className="text-2xl font-bold text-success">
            {supplyBids.reduce((s, b) => s + b.quantity, 0).toFixed(0)} MW
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
            {demandBids.reduce((s, b) => s + b.quantity, 0).toFixed(0)} MW
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
            ×{HOURLY_DEMAND_MULTIPLIER[bidData.selectedHour]?.toFixed(2) || '1.00'}
          </div>
          <div className="text-xs text-text-muted mt-1">
            {bidData.selectedHour >= 9 && bidData.selectedHour <= 17 ? '주간 피크' :
             bidData.selectedHour >= 18 && bidData.selectedHour <= 21 ? '저녁 피크' : '오프피크'}
          </div>
        </div>
      </div>

      {/* Phase Progress */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            처리 단계
          </h3>
          <span className={clsx(
            'px-3 py-1 rounded-full text-sm font-medium',
            simulation.phase === 'complete' ? 'bg-success/20 text-success' : 'bg-primary/20 text-primary'
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
                  currentIdx === idx ? 'bg-primary' : currentIdx > idx ? 'bg-success' : 'bg-background'
                )}>
                  {simulation.phase === phase && (
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-100"
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
            <BarChart3 className="w-5 h-5 text-primary" />
            수요/공급 곡선 (Merit Order)
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

                {/* Supply curve - green (step pattern for block bids) */}
                <Line
                  type="stepAfter"
                  dataKey="supplyPrice"
                  stroke="#10b981"
                  strokeWidth={2}
                  name="supplyPrice"
                  dot={false}
                  connectNulls={true}
                />

                {/* Demand curve - orange (step pattern for block bids) */}
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
                      value: `SMP: ${simulation.clearingPrice.toFixed(1)}원`,
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
                  <span className="text-sm text-text-muted">청산가격 (SMP)</span>
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
            <div className="space-y-4">
              <div className="p-4 bg-background rounded-lg text-center">
                <div className="text-3xl font-bold text-smp">
                  {simulation.clearingPrice ? `${simulation.clearingPrice.toFixed(1)}원` : '-'}
                </div>
                <div className="text-sm text-text-muted">청산가격 (SMP)</div>
                {simulation.clearingPrice && bidData.smpForecast.q50 && (
                  <div className={clsx(
                    'text-xs mt-1',
                    simulation.clearingPrice > bidData.smpForecast.q50 ? 'text-danger' : 'text-success'
                  )}>
                    예측 대비 {simulation.clearingPrice > bidData.smpForecast.q50 ? '+' : ''}
                    {((simulation.clearingPrice - bidData.smpForecast.q50) / bidData.smpForecast.q50 * 100).toFixed(1)}%
                  </div>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-background rounded-lg text-center">
                  <div className="text-xl font-bold text-success">
                    {simulation.clearingQuantity ? `${simulation.clearingQuantity.toFixed(0)}` : '-'}
                  </div>
                  <div className="text-xs text-text-muted">거래량 (MW)</div>
                </div>
                <div className="p-3 bg-background rounded-lg text-center">
                  <div className="text-xl font-bold text-primary">
                    {supplyBids.filter(b => b.status === 'accepted' || b.status === 'partial').length}
                  </div>
                  <div className="text-xs text-text-muted">낙찰 발전사</div>
                </div>
              </div>

              {/* Market Type Indicator */}
              {simulation.phase === 'complete' && (
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
                    'text-sm',
                    simulation.marketType === 'normal' ? 'text-success' :
                    simulation.marketType === 'oversupply' ? 'text-sky-500' : 'text-danger'
                  )}>
                    {simulation.marketType === 'normal' ? '정상 시장' :
                     simulation.marketType === 'oversupply' ? '공급 과잉' : '공급 부족'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Our Results */}
          <div className={clsx(
            'card border-2',
            simulation.phase === 'complete' && simulation.ourAccepted > 0
              ? 'border-success'
              : simulation.phase === 'complete'
                ? 'border-danger'
                : 'border-transparent'
          )}>
            <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Award className="w-5 h-5 text-primary" />
              우리 입찰 결과
            </h3>
            {simulation.phase === 'complete' ? (
              <div className="space-y-4">
                <div className="flex items-center justify-center gap-3">
                  {simulation.ourAccepted > 0 ? (
                    <>
                      <CheckCircle className="w-8 h-8 text-success" />
                      <span className="text-xl font-bold text-success">낙찰 성공!</span>
                    </>
                  ) : (
                    <>
                      <XCircle className="w-8 h-8 text-danger" />
                      <span className="text-xl font-bold text-danger">미낙찰</span>
                    </>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 bg-background rounded-lg text-center">
                    <div className="text-xl font-bold text-success">
                      {simulation.ourAccepted.toFixed(1)} MW
                    </div>
                    <div className="text-xs text-text-muted">낙찰 물량</div>
                    <div className="text-xs text-text-muted mt-1">
                      ({((simulation.ourAccepted / ourTotalBid) * 100).toFixed(0)}%)
                    </div>
                  </div>
                  <div className="p-3 bg-background rounded-lg text-center">
                    <div className="text-xl font-bold text-primary">
                      {(simulation.ourRevenue / 1000).toFixed(1)}천원
                    </div>
                    <div className="text-xs text-text-muted">예상 수익</div>
                    <div className="text-xs text-text-muted mt-1">
                      (시간당)
                    </div>
                  </div>
                </div>
                <div className="p-3 bg-success/10 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-text-muted">낙찰 구간</span>
                    <span className="text-success font-medium">
                      {supplyBids.filter(b => b.isOurs && (b.status === 'accepted' || b.status === 'partial')).length} / {supplyBids.filter(b => b.isOurs).length}
                    </span>
                  </div>
                </div>

                {/* Bid Strategy Feedback */}
                {simulation.clearingPrice && (
                  <div className="p-3 bg-background rounded-lg">
                    <div className="text-xs text-text-muted mb-1">입찰 전략 피드백</div>
                    {ourAvgPrice <= simulation.clearingPrice ? (
                      <p className="text-sm text-success">
                        평균 입찰가({ourAvgPrice.toFixed(0)}원)가 SMP({simulation.clearingPrice.toFixed(0)}원) 이하로
                        적절한 입찰 전략입니다.
                      </p>
                    ) : (
                      <p className="text-sm text-warning">
                        평균 입찰가({ourAvgPrice.toFixed(0)}원)가 SMP({simulation.clearingPrice.toFixed(0)}원)보다
                        높습니다. 입찰가 하향 조정을 검토하세요.
                      </p>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-text-muted">
                시뮬레이션 완료 후 결과가 표시됩니다
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bid Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Supply Bids */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Factory className="w-5 h-5 text-success" />
            공급 입찰 (발전사) - Merit Order
          </h3>
          <div className="overflow-x-auto max-h-[400px]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="text-text-muted border-b border-border">
                  <th className="text-left py-2 px-3">발전사</th>
                  <th className="text-right py-2 px-3">물량</th>
                  <th className="text-right py-2 px-3">가격</th>
                  <th className="text-center py-2 px-3">상태</th>
                </tr>
              </thead>
              <tbody>
                {supplyBids.map((bid) => (
                  <tr
                    key={bid.id}
                    className={clsx(
                      'border-b border-border/50 transition-colors',
                      bid.isOurs && 'bg-primary/10',
                      bid.status === 'accepted' && 'bg-success/5',
                      bid.status === 'partial' && 'bg-warning/5',
                      bid.status === 'rejected' && 'opacity-50'
                    )}
                  >
                    <td className="py-2 px-3 font-medium text-text-primary">
                      <div className="flex items-center gap-2">
                        <ResourceIcon type={bid.resourceType} />
                        <span>{bid.bidder}</span>
                        {bid.isOurs && (
                          <span className="px-1.5 py-0.5 text-[10px] bg-primary/20 text-primary rounded">
                            우리
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="text-right py-2 px-3 text-text-muted">
                      {bid.status === 'partial' ? (
                        <span className="text-warning">
                          {bid.acceptedQuantity?.toFixed(1)}/{bid.quantity.toFixed(1)}
                        </span>
                      ) : (
                        `${bid.quantity.toFixed(1)} MW`
                      )}
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-text-primary">
                      {bid.price.toFixed(1)}원
                    </td>
                    <td className="text-center py-2 px-3">
                      {bid.status === 'accepted' && (
                        <span className="text-success text-xs font-medium">✓ 낙찰</span>
                      )}
                      {bid.status === 'partial' && (
                        <span className="text-warning text-xs font-medium">△ 부분</span>
                      )}
                      {bid.status === 'rejected' && (
                        <span className="text-danger text-xs">✗ 탈락</span>
                      )}
                      {bid.status === 'pending' && (
                        <span className="text-text-muted text-xs">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Demand Bids */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
            <Users className="w-5 h-5 text-warning" />
            수요 입찰 (수요처) - 가격 내림차순
          </h3>
          <div className="overflow-x-auto max-h-[400px]">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-card">
                <tr className="text-text-muted border-b border-border">
                  <th className="text-left py-2 px-3">수요처</th>
                  <th className="text-right py-2 px-3">물량</th>
                  <th className="text-right py-2 px-3">희망가격</th>
                  <th className="text-center py-2 px-3">상태</th>
                </tr>
              </thead>
              <tbody>
                {demandBids.map((bid) => (
                  <tr
                    key={bid.id}
                    className={clsx(
                      'border-b border-border/50 transition-colors',
                      bid.status === 'accepted' && 'bg-success/5',
                      bid.status === 'partial' && 'bg-warning/5',
                      bid.status === 'rejected' && 'opacity-50'
                    )}
                  >
                    <td className="py-2 px-3 font-medium text-text-primary">{bid.bidder}</td>
                    <td className="text-right py-2 px-3 text-text-muted">
                      {bid.status === 'partial' ? (
                        <span className="text-warning">
                          {bid.acceptedQuantity?.toFixed(1)}/{bid.quantity.toFixed(1)}
                        </span>
                      ) : (
                        `${bid.quantity.toFixed(1)} MW`
                      )}
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-text-primary">
                      {bid.price.toFixed(1)}원
                    </td>
                    <td className="text-center py-2 px-3">
                      {bid.status === 'accepted' && (
                        <span className="text-success text-xs font-medium">✓ 체결</span>
                      )}
                      {bid.status === 'partial' && (
                        <span className="text-warning text-xs font-medium">△ 부분</span>
                      )}
                      {bid.status === 'rejected' && (
                        <span className="text-text-muted text-xs">미체결</span>
                      )}
                      {bid.status === 'pending' && (
                        <span className="text-text-muted text-xs">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Info Panel */}
      <div className="card bg-primary/5 border-primary/20">
        <h3 className="text-sm font-semibold text-primary mb-2 flex items-center gap-2">
          <Info className="w-4 h-4" />
          KPX 하루전시장(DAM) 입찰 매칭 원리
        </h3>
        <div className="text-sm text-text-muted leading-relaxed space-y-2">
          <p>
            한국전력거래소(KPX)의 하루전시장은 <strong>비용기반 풀(Cost-Based Pool)</strong> 방식으로 운영됩니다.
          </p>
          <ul className="list-disc list-inside space-y-1 ml-2">
            <li><strong>Merit Order</strong>: 발전사는 가격 오름차순으로 정렬되어 낮은 가격부터 우선 급전</li>
            <li><strong>균일가격(SMP)</strong>: 모든 낙찰 발전사는 동일한 청산가격으로 정산 (가격차별 없음)</li>
            <li><strong>마지널 발전기</strong>: 청산가격은 마지막으로 급전된 발전기의 입찰가로 결정</li>
            <li><strong>재생에너지 우선</strong>: 태양광/풍력은 낮은 변동비로 인해 Merit Order 상위에 배치</li>
          </ul>
          <p className="text-xs text-text-muted/80 mt-2">
            ※ 이 시뮬레이션은 교육/발표 목적이며, 실제 KPX 시장과 세부 규칙이 다를 수 있습니다.
          </p>
        </div>
      </div>
    </div>
  );
}
