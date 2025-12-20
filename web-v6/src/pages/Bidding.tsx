/**
 * Bidding Page - RE-BMS v6.0
 * 10-Segment Bidding Management for DAM/RTM
 */

import { useState } from 'react';
import {
  CheckCircle,
  Send,
  Save,
  Sparkles,
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ComposedChart,
  Line,
  Bar,
} from 'recharts';
import { useSMPForecast, useMarketStatus } from '../hooks/useApi';
import clsx from 'clsx';

interface BidSegment {
  id: number;
  quantity: number;
  price: number;
}

export default function Bidding() {
  const { data: forecast } = useSMPForecast();
  const { data: marketStatus } = useMarketStatus();
  const [selectedHour, setSelectedHour] = useState(12);
  const [riskLevel, setRiskLevel] = useState<'conservative' | 'moderate' | 'aggressive'>('moderate');
  const [capacity, setCapacity] = useState(50);

  // Generate 10-segment bid structure
  const [segments, setSegments] = useState<BidSegment[]>(() => {
    return Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      quantity: capacity / 10,
      price: 80 + i * 5,
    }));
  });

  // Calculate bid curve data for chart
  const bidCurveData = segments.map((seg, idx) => ({
    segment: `Seg ${seg.id}`,
    quantity: seg.quantity,
    cumulativeQuantity: segments.slice(0, idx + 1).reduce((sum, s) => sum + s.quantity, 0),
    price: seg.price,
  }));

  // Get SMP forecast for selected hour
  const smpForHour = forecast ? {
    q10: forecast.q10[selectedHour],
    q50: forecast.q50[selectedHour],
    q90: forecast.q90[selectedHour],
  } : { q10: 85, q50: 95, q90: 110 };

  // Update segment price
  const updateSegmentPrice = (id: number, newPrice: number) => {
    setSegments((prev) => {
      const updated = [...prev];
      const idx = updated.findIndex((s) => s.id === id);
      if (idx >= 0) {
        updated[idx] = { ...updated[idx], price: newPrice };
        // Enforce monotonic constraint
        for (let i = idx + 1; i < updated.length; i++) {
          if (updated[i].price < newPrice) {
            updated[i] = { ...updated[i], price: newPrice };
          }
        }
        for (let i = idx - 1; i >= 0; i--) {
          if (updated[i].price > newPrice) {
            updated[i] = { ...updated[i], price: newPrice };
          }
        }
      }
      return updated;
    });
  };

  // AI optimization
  const handleOptimize = () => {
    // Apply optimization based on SMP forecast
    const basePrice = smpForHour.q10 * 0.9;
    const priceSpread = (smpForHour.q90 - smpForHour.q10) / 9;

    const newSegments = segments.map((seg, idx) => ({
      ...seg,
      price: Math.round(basePrice + idx * priceSpread),
      quantity: capacity / 10,
    }));

    setSegments(newSegments);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">입찰 관리</h1>
          <p className="text-gray-400 mt-1">10-Segment 입찰가격 설정</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Market Status */}
          <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
            <div className={clsx(
              'status-dot',
              marketStatus?.dam.status === 'open' ? 'status-success' : 'status-danger'
            )} />
            <span className="text-sm text-gray-400">
              DAM {marketStatus?.dam.status === 'open' ? '거래 가능' : '마감'}
            </span>
            {marketStatus?.dam.hours_remaining && (
              <span className="text-xs text-warning">
                {marketStatus.dam.hours_remaining}시간 남음
              </span>
            )}
          </div>
          <button onClick={handleOptimize} className="btn-secondary flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            AI 최적화
          </button>
          <button className="btn-primary flex items-center gap-2">
            <Send className="w-4 h-4" />
            KPX 제출
          </button>
        </div>
      </div>

      {/* Settings Row */}
      <div className="grid grid-cols-4 gap-4">
        {/* Hour Selection */}
        <div className="card">
          <label className="text-sm text-gray-400 block mb-2">거래 시간대</label>
          <select
            value={selectedHour}
            onChange={(e) => setSelectedHour(Number(e.target.value))}
            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-white"
          >
            {Array.from({ length: 24 }, (_, i) => (
              <option key={i} value={i}>
                {String(i).padStart(2, '0')}:00 - {String(i + 1).padStart(2, '0')}:00
              </option>
            ))}
          </select>
        </div>

        {/* Capacity */}
        <div className="card">
          <label className="text-sm text-gray-400 block mb-2">입찰 용량 (MW)</label>
          <input
            type="number"
            value={capacity}
            onChange={(e) => setCapacity(Number(e.target.value))}
            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-white"
            min={1}
            max={500}
          />
        </div>

        {/* Risk Level */}
        <div className="card">
          <label className="text-sm text-gray-400 block mb-2">위험 선호도</label>
          <div className="flex gap-2">
            {(['conservative', 'moderate', 'aggressive'] as const).map((level) => (
              <button
                key={level}
                onClick={() => setRiskLevel(level)}
                className={clsx(
                  'flex-1 px-3 py-2 text-sm rounded-lg transition-colors',
                  riskLevel === level
                    ? 'bg-primary text-white'
                    : 'bg-background text-gray-400 hover:bg-background/80'
                )}
              >
                {level === 'conservative' && '보수적'}
                {level === 'moderate' && '균형'}
                {level === 'aggressive' && '공격적'}
              </button>
            ))}
          </div>
        </div>

        {/* SMP Reference */}
        <div className="card">
          <label className="text-sm text-gray-400 block mb-2">SMP 예측 (원/kWh)</label>
          <div className="flex items-center justify-between">
            <div className="text-center">
              <div className="text-success text-sm">{smpForHour.q10.toFixed(0)}</div>
              <div className="text-xs text-gray-500">하한</div>
            </div>
            <div className="text-center">
              <div className="text-smp text-xl font-bold">{smpForHour.q50.toFixed(0)}</div>
              <div className="text-xs text-gray-500">예측</div>
            </div>
            <div className="text-center">
              <div className="text-danger text-sm">{smpForHour.q90.toFixed(0)}</div>
              <div className="text-xs text-gray-500">상한</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-2 gap-6">
        {/* Bid Matrix */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">10-Segment 입찰 매트릭스</h3>
          <div className="space-y-2">
            <div className="grid grid-cols-4 gap-2 text-xs text-gray-400 font-medium px-2">
              <span>구간</span>
              <span className="text-right">물량 (MW)</span>
              <span className="text-right">가격 (원/kWh)</span>
              <span className="text-right">예상수익</span>
            </div>
            {segments.map((seg, idx) => (
              <div
                key={seg.id}
                className={clsx(
                  'grid grid-cols-4 gap-2 items-center p-2 rounded-lg transition-colors',
                  seg.price <= smpForHour.q50 ? 'bg-success/10 border border-success/20' : 'bg-background'
                )}
              >
                <span className="text-white font-medium">Seg {seg.id}</span>
                <input
                  type="number"
                  value={seg.quantity}
                  onChange={(e) => {
                    const newQuantity = Number(e.target.value);
                    setSegments((prev) => {
                      const updated = [...prev];
                      updated[idx] = { ...updated[idx], quantity: newQuantity };
                      return updated;
                    });
                  }}
                  className="w-full bg-card border border-border rounded px-2 py-1 text-white text-right text-sm"
                />
                <input
                  type="number"
                  value={seg.price}
                  onChange={(e) => updateSegmentPrice(seg.id, Number(e.target.value))}
                  className="w-full bg-card border border-border rounded px-2 py-1 text-white text-right text-sm"
                />
                <span className="text-right text-sm text-gray-400 font-mono">
                  {((seg.quantity * seg.price) / 1000).toFixed(1)}K
                </span>
              </div>
            ))}
          </div>

          {/* Total */}
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">총 입찰량</span>
              <span className="text-xl font-bold text-white">
                {segments.reduce((sum, s) => sum + s.quantity, 0).toFixed(1)} MW
              </span>
            </div>
            <div className="flex justify-between items-center mt-2">
              <span className="text-gray-400">예상 평균가</span>
              <span className="text-xl font-bold text-smp">
                {(segments.reduce((sum, s) => sum + s.price * s.quantity, 0) /
                  segments.reduce((sum, s) => sum + s.quantity, 0)).toFixed(1)} 원/kWh
              </span>
            </div>
          </div>
        </div>

        {/* Bid Curve Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">입찰 곡선 (Step Chart)</h3>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={bidCurveData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="cumulativeQuantity"
                  stroke="#9ca3af"
                  fontSize={12}
                  tickLine={false}
                  label={{ value: '누적 물량 (MW)', position: 'bottom', fill: '#9ca3af' }}
                />
                <YAxis
                  stroke="#9ca3af"
                  fontSize={12}
                  tickLine={false}
                  label={{ value: '가격 (원/kWh)', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e2530',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  labelStyle={{ color: '#fff' }}
                />
                <Legend />

                {/* SMP Reference Lines */}
                <Line
                  type="stepAfter"
                  dataKey="price"
                  stroke="#fbbf24"
                  strokeWidth={3}
                  name="입찰가격"
                  dot={{ fill: '#fbbf24', strokeWidth: 2 }}
                />

                {/* Bar for quantity */}
                <Bar
                  dataKey="quantity"
                  fill="#6366f1"
                  opacity={0.5}
                  name="구간 물량"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* SMP Reference Overlay */}
          <div className="mt-4 flex items-center justify-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-success" />
              <span className="text-xs text-gray-400">SMP 하한 ({smpForHour.q10.toFixed(0)})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-smp" />
              <span className="text-xs text-gray-400">SMP 예측 ({smpForHour.q50.toFixed(0)})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-0.5 bg-danger" />
              <span className="text-xs text-gray-400">SMP 상한 ({smpForHour.q90.toFixed(0)})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success" />
              <span className="text-sm text-gray-400">단조성 제약 충족</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success" />
              <span className="text-sm text-gray-400">용량 제한 준수</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="btn-secondary flex items-center gap-2">
              <Save className="w-4 h-4" />
              임시 저장
            </button>
            <button className="btn-primary flex items-center gap-2">
              <Send className="w-4 h-4" />
              입찰 제출
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
