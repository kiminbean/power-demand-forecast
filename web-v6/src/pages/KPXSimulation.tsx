/**
 * KPX Market Simulation Page - RE-BMS v6.1
 * Simulates KPX electricity market bid matching process
 */

import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Pause,
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
  Legend,
} from 'recharts';
import { useTheme } from '../contexts/ThemeContext';
import clsx from 'clsx';

interface MarketBid {
  id: string;
  bidder: string;
  type: 'supply' | 'demand';
  quantity: number;
  price: number;
  isOurs?: boolean;
  status: 'pending' | 'accepted' | 'rejected' | 'partial';
}

interface SimulationState {
  phase: 'idle' | 'collecting' | 'sorting' | 'matching' | 'clearing' | 'complete';
  progress: number;
  clearingPrice: number | null;
  matchedQuantity: number;
  ourAccepted: number;
  ourRevenue: number;
}

// Generate mock supply bids (sorted by price ascending for merit order)
const generateSupplyBids = (ourBids: { quantity: number; price: number }[]): MarketBid[] => {
  const mockSuppliers = [
    { name: '한라풍력', basePrice: 75, capacity: 30 },
    { name: '제주태양광', basePrice: 78, capacity: 25 },
    { name: '서귀포ESS', basePrice: 82, capacity: 20 },
    { name: '동부바이오', basePrice: 88, capacity: 15 },
    { name: 'eXeco (우리)', basePrice: 0, capacity: 0, isOurs: true },
    { name: '제주화력', basePrice: 95, capacity: 40 },
    { name: '남부GT', basePrice: 105, capacity: 35 },
    { name: '비상LNG', basePrice: 120, capacity: 50 },
  ];

  const bids: MarketBid[] = [];

  mockSuppliers.forEach((supplier, idx) => {
    if (supplier.isOurs) {
      // Add our bids
      ourBids.forEach((bid, bidIdx) => {
        bids.push({
          id: `our-${bidIdx}`,
          bidder: 'eXeco (우리)',
          type: 'supply',
          quantity: bid.quantity,
          price: bid.price,
          isOurs: true,
          status: 'pending',
        });
      });
    } else {
      // Add mock supplier bids
      const priceVariation = Math.random() * 10 - 5;
      bids.push({
        id: `supply-${idx}`,
        bidder: supplier.name,
        type: 'supply',
        quantity: supplier.capacity + Math.random() * 10 - 5,
        price: supplier.basePrice + priceVariation,
        status: 'pending',
      });
    }
  });

  return bids.sort((a, b) => a.price - b.price);
};

// Generate mock demand bids (sorted by price descending)
const generateDemandBids = (): MarketBid[] => {
  const demanders = [
    { name: '제주공항', price: 130, quantity: 25 },
    { name: '삼성SDI', price: 125, quantity: 45 },
    { name: '롯데리조트', price: 115, quantity: 20 },
    { name: '한라시멘트', price: 110, quantity: 35 },
    { name: '제주시청', price: 105, quantity: 30 },
    { name: '서귀포시', price: 100, quantity: 28 },
    { name: '농협', price: 95, quantity: 22 },
    { name: '기타수요', price: 85, quantity: 40 },
  ];

  return demanders.map((d, idx): MarketBid => ({
    id: `demand-${idx}`,
    bidder: d.name,
    type: 'demand' as const,
    quantity: d.quantity + Math.random() * 10 - 5,
    price: d.price + Math.random() * 5,
    status: 'pending' as const,
  })).sort((a, b) => b.price - a.price);
};

export default function KPXSimulation() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDark } = useTheme();

  // Get bid data from navigation state
  const bidData = location.state as {
    segments: { id: number; quantity: number; price: number }[];
    selectedHour: number;
    smpForecast: { q10: number; q50: number; q90: number };
  } | null;

  const [simulation, setSimulation] = useState<SimulationState>({
    phase: 'idle',
    progress: 0,
    clearingPrice: null,
    matchedQuantity: 0,
    ourAccepted: 0,
    ourRevenue: 0,
  });

  const [supplyBids, setSupplyBids] = useState<MarketBid[]>([]);
  const [demandBids, setDemandBids] = useState<MarketBid[]>([]);

  // Initialize bids
  useEffect(() => {
    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      setSupplyBids(generateSupplyBids(ourBids));
      setDemandBids(generateDemandBids());
    }
  }, [bidData]);

  // Build supply/demand curve data
  const buildCurveData = useCallback(() => {
    const data: { quantity: number; supplyPrice: number | null; demandPrice: number | null }[] = [];

    // Supply curve (cumulative quantity, ascending price)
    let cumSupply = 0;
    supplyBids.forEach(bid => {
      data.push({ quantity: cumSupply, supplyPrice: bid.price, demandPrice: null });
      cumSupply += bid.quantity;
      data.push({ quantity: cumSupply, supplyPrice: bid.price, demandPrice: null });
    });

    // Demand curve (cumulative quantity, descending price)
    let cumDemand = 0;
    demandBids.forEach(bid => {
      const existing = data.find(d => Math.abs(d.quantity - cumDemand) < 1);
      if (existing) {
        existing.demandPrice = bid.price;
      } else {
        data.push({ quantity: cumDemand, supplyPrice: null, demandPrice: bid.price });
      }
      cumDemand += bid.quantity;
      const existingEnd = data.find(d => Math.abs(d.quantity - cumDemand) < 1);
      if (existingEnd) {
        existingEnd.demandPrice = bid.price;
      } else {
        data.push({ quantity: cumDemand, supplyPrice: null, demandPrice: bid.price });
      }
    });

    return data.sort((a, b) => a.quantity - b.quantity);
  }, [supplyBids, demandBids]);

  // Simulation phases
  const runSimulation = useCallback(() => {
    const phases: SimulationState['phase'][] = ['collecting', 'sorting', 'matching', 'clearing', 'complete'];
    let currentPhaseIdx = 0;

    const interval = setInterval(() => {
      setSimulation(prev => {
        if (prev.progress >= 100) {
          currentPhaseIdx++;
          if (currentPhaseIdx >= phases.length) {
            clearInterval(interval);

            // Calculate final results
            const clearingPrice = 92; // Simulated clearing price
            const totalDemand = demandBids.reduce((sum, b) => sum + b.quantity, 0);

            // Update bid statuses
            setSupplyBids(bids =>
              bids.map(b => ({
                ...b,
                status: b.price <= clearingPrice ? 'accepted' : 'rejected',
              }))
            );
            setDemandBids(bids =>
              bids.map(b => ({
                ...b,
                status: b.price >= clearingPrice ? 'accepted' : 'rejected',
              }))
            );

            // Calculate our accepted quantity
            const ourAccepted = supplyBids
              .filter(b => b.isOurs && b.price <= clearingPrice)
              .reduce((sum, b) => sum + b.quantity, 0);

            return {
              ...prev,
              phase: 'complete',
              progress: 100,
              clearingPrice,
              matchedQuantity: Math.min(
                supplyBids.filter(b => b.price <= clearingPrice).reduce((sum, b) => sum + b.quantity, 0),
                totalDemand
              ),
              ourAccepted,
              ourRevenue: ourAccepted * clearingPrice,
            };
          }
          return {
            ...prev,
            phase: phases[currentPhaseIdx],
            progress: 0,
          };
        }
        return {
          ...prev,
          progress: prev.progress + 5,
        };
      });
    }, 100);

    return () => clearInterval(interval);
  }, [supplyBids, demandBids]);

  const handlePlay = () => {
    setSimulation({
      phase: 'collecting',
      progress: 0,
      clearingPrice: null,
      matchedQuantity: 0,
      ourAccepted: 0,
      ourRevenue: 0,
    });
    runSimulation();
  };

  const handleReset = () => {
    setSimulation({
      phase: 'idle',
      progress: 0,
      clearingPrice: null,
      matchedQuantity: 0,
      ourAccepted: 0,
      ourRevenue: 0,
    });
    if (bidData) {
      const ourBids = bidData.segments.map(s => ({
        quantity: s.quantity,
        price: s.price,
      }));
      setSupplyBids(generateSupplyBids(ourBids));
      setDemandBids(generateDemandBids());
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
    collecting: '입찰 수집 중',
    sorting: '가격 정렬 중',
    matching: '수급 매칭 중',
    clearing: '청산가격 결정 중',
    complete: '매칭 완료',
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
              KPX 전력시장 시뮬레이션
            </h1>
            <p className="text-text-muted mt-1">
              {String(bidData.selectedHour).padStart(2, '0')}:00 - {String(bidData.selectedHour + 1).padStart(2, '0')}:00 거래시간 입찰 매칭
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
              <Pause className="w-4 h-4" />
              진행 중...
            </button>
          )}
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
          {(['collecting', 'sorting', 'matching', 'clearing', 'complete'] as const).map((phase, idx) => (
            <div key={phase} className="flex-1 flex items-center">
              <div className={clsx(
                'flex-1 h-2 rounded-full transition-colors duration-300',
                simulation.phase === phase
                  ? 'bg-primary'
                  : ['collecting', 'sorting', 'matching', 'clearing', 'complete'].indexOf(simulation.phase) > idx
                    ? 'bg-success'
                    : 'bg-background'
              )}>
                {simulation.phase === phase && (
                  <div
                    className="h-full bg-primary/50 rounded-full transition-all duration-100"
                    style={{ width: `${simulation.progress}%` }}
                  />
                )}
              </div>
              {idx < 4 && <div className="w-2" />}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-2 text-xs text-text-muted">
          <span>입찰 수집</span>
          <span>가격 정렬</span>
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
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={buildCurveData()} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis
                  dataKey="quantity"
                  stroke={chartColors.axis}
                  fontSize={12}
                  label={{ value: '누적 물량 (MW)', position: 'insideBottom', offset: -10, fill: chartColors.axis }}
                />
                <YAxis
                  stroke={chartColors.axis}
                  fontSize={12}
                  width={60}
                  domain={[60, 140]}
                  label={{ value: '가격 (원/kWh)', angle: -90, position: 'insideLeft', fill: chartColors.axis }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltipBg,
                    border: `1px solid ${chartColors.tooltipBorder}`,
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                {/* Supply curve */}
                <Line
                  type="stepAfter"
                  dataKey="supplyPrice"
                  stroke="#10b981"
                  strokeWidth={3}
                  name="공급 곡선"
                  dot={false}
                  connectNulls={false}
                />
                {/* Demand curve */}
                <Line
                  type="stepAfter"
                  dataKey="demandPrice"
                  stroke="#f59e0b"
                  strokeWidth={3}
                  name="수요 곡선"
                  dot={false}
                  connectNulls={false}
                />
                {/* Clearing price reference */}
                {simulation.clearingPrice && (
                  <ReferenceLine
                    y={simulation.clearingPrice}
                    stroke="#ef4444"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{
                      value: `청산가격: ${simulation.clearingPrice}원`,
                      position: 'right',
                      fill: '#ef4444',
                      fontSize: 12,
                    }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
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
                  {simulation.clearingPrice ? `${simulation.clearingPrice}원` : '-'}
                </div>
                <div className="text-sm text-text-muted">청산가격 (SMP)</div>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-background rounded-lg text-center">
                  <div className="text-xl font-bold text-success">
                    {simulation.matchedQuantity ? `${simulation.matchedQuantity.toFixed(1)}` : '-'}
                  </div>
                  <div className="text-xs text-text-muted">총 거래량 (MW)</div>
                </div>
                <div className="p-3 bg-background rounded-lg text-center">
                  <div className="text-xl font-bold text-primary">
                    {supplyBids.filter(b => b.status === 'accepted').length}
                  </div>
                  <div className="text-xs text-text-muted">낙찰 발전사</div>
                </div>
              </div>
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
                  </div>
                  <div className="p-3 bg-background rounded-lg text-center">
                    <div className="text-xl font-bold text-primary">
                      {(simulation.ourRevenue / 1000).toFixed(1)}K
                    </div>
                    <div className="text-xs text-text-muted">예상 수익 (천원)</div>
                  </div>
                </div>
                <div className="p-3 bg-success/10 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-text-muted">낙찰 구간</span>
                    <span className="text-success font-medium">
                      {supplyBids.filter(b => b.isOurs && b.status === 'accepted').length} / {supplyBids.filter(b => b.isOurs).length}
                    </span>
                  </div>
                </div>
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
            공급 입찰 (발전사)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
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
                      bid.status === 'rejected' && 'bg-danger/5 opacity-50'
                    )}
                  >
                    <td className="py-2 px-3 font-medium text-text-primary">
                      {bid.bidder}
                      {bid.isOurs && (
                        <span className="ml-2 px-1.5 py-0.5 text-[10px] bg-primary/20 text-primary rounded">
                          우리
                        </span>
                      )}
                    </td>
                    <td className="text-right py-2 px-3 text-text-muted">
                      {bid.quantity.toFixed(1)} MW
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-text-primary">
                      {bid.price.toFixed(0)}원
                    </td>
                    <td className="text-center py-2 px-3">
                      {bid.status === 'accepted' && (
                        <span className="text-success">✓ 낙찰</span>
                      )}
                      {bid.status === 'rejected' && (
                        <span className="text-danger">✗ 탈락</span>
                      )}
                      {bid.status === 'pending' && (
                        <span className="text-text-muted">-</span>
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
            수요 입찰 (수요처)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-text-muted border-b border-border">
                  <th className="text-left py-2 px-3">수요처</th>
                  <th className="text-right py-2 px-3">물량</th>
                  <th className="text-right py-2 px-3">가격</th>
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
                      bid.status === 'rejected' && 'bg-danger/5 opacity-50'
                    )}
                  >
                    <td className="py-2 px-3 font-medium text-text-primary">{bid.bidder}</td>
                    <td className="text-right py-2 px-3 text-text-muted">
                      {bid.quantity.toFixed(1)} MW
                    </td>
                    <td className="text-right py-2 px-3 font-mono text-text-primary">
                      {bid.price.toFixed(0)}원
                    </td>
                    <td className="text-center py-2 px-3">
                      {bid.status === 'accepted' && (
                        <span className="text-success">✓ 낙찰</span>
                      )}
                      {bid.status === 'rejected' && (
                        <span className="text-warning">△ 미체결</span>
                      )}
                      {bid.status === 'pending' && (
                        <span className="text-text-muted">-</span>
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
        <h3 className="text-sm font-semibold text-primary mb-2">KPX 전력시장 입찰 매칭 원리</h3>
        <p className="text-sm text-text-muted leading-relaxed">
          한국전력거래소(KPX)의 하루전시장(DAM)은 <strong>Merit Order 방식</strong>으로 운영됩니다.
          공급 입찰은 가격이 낮은 순서대로, 수요 입찰은 가격이 높은 순서대로 정렬되어
          수요와 공급이 만나는 지점에서 <strong>균일가격(SMP)</strong>이 결정됩니다.
          SMP 이하로 입찰한 발전사는 모두 낙찰되며, 실제 정산은 SMP로 이루어집니다.
        </p>
      </div>
    </div>
  );
}
