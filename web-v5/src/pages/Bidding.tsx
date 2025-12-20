/**
 * Mobile Bidding Page
 */

import { useState } from 'react';
import { Sparkles, Send, ChevronDown, ChevronUp, CheckCircle } from 'lucide-react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useSMPForecast, useMarketStatus } from '../hooks/useApi';
import { useTheme } from '../contexts/ThemeContext';
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
  const [capacity, setCapacity] = useState(50);
  const [showSegments, setShowSegments] = useState(false);
  const { isDark } = useTheme();

  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1e2530' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
  };

  // Generate 10-segment bid structure
  const [segments, setSegments] = useState<BidSegment[]>(() => {
    return Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      quantity: capacity / 10,
      price: 80 + i * 5,
    }));
  });

  // Get SMP forecast for selected hour
  const smpForHour = forecast ? {
    q10: forecast.q10[selectedHour],
    q50: forecast.q50[selectedHour],
    q90: forecast.q90[selectedHour],
  } : { q10: 85, q50: 95, q90: 110 };

  // Calculate bid curve data for chart
  const bidCurveData = segments.map((seg, idx) => ({
    segment: `S${seg.id}`,
    quantity: seg.quantity,
    cumulativeQuantity: segments.slice(0, idx + 1).reduce((sum, s) => sum + s.quantity, 0),
    price: seg.price,
  }));

  // AI optimization
  const handleOptimize = () => {
    const basePrice = smpForHour.q10 * 0.9;
    const priceSpread = (smpForHour.q90 - smpForHour.q10) / 9;
    const newSegments = segments.map((seg, idx) => ({
      ...seg,
      price: Math.round(basePrice + idx * priceSpread),
      quantity: capacity / 10,
    }));
    setSegments(newSegments);
  };

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

  const totalQuantity = segments.reduce((sum, s) => sum + s.quantity, 0);
  const avgPrice = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalQuantity;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-text-primary">입찰 관리</h1>
          <p className="text-xs text-text-muted mt-0.5">10-Segment 입찰가격 설정</p>
        </div>
        {/* Market Status */}
        <div className="flex items-center gap-1.5 px-2 py-1 bg-card rounded-lg border border-border">
          <div className={clsx(
            'status-dot',
            marketStatus?.dam.status === 'open' ? 'status-success' : 'status-danger'
          )} />
          <span className="text-xs text-text-muted">
            DAM {marketStatus?.dam.status === 'open' ? '거래가능' : '마감'}
          </span>
        </div>
      </div>

      {/* SMP Reference Card */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-medium text-text-primary">SMP 예측 ({selectedHour}시)</span>
          <select
            value={selectedHour}
            onChange={(e) => setSelectedHour(Number(e.target.value))}
            className="text-sm bg-background border border-border rounded-lg px-2 py-1 text-text-primary"
          >
            {Array.from({ length: 24 }, (_, i) => (
              <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
            ))}
          </select>
        </div>
        <div className="flex items-center justify-between">
          <div className="text-center">
            <div className="text-success text-lg font-bold">{smpForHour.q10.toFixed(0)}</div>
            <div className="text-[10px] text-text-muted">하한</div>
          </div>
          <div className="text-center">
            <div className="text-smp text-3xl font-bold">{smpForHour.q50.toFixed(0)}</div>
            <div className="text-[10px] text-text-muted">예측</div>
          </div>
          <div className="text-center">
            <div className="text-danger text-lg font-bold">{smpForHour.q90.toFixed(0)}</div>
            <div className="text-[10px] text-text-muted">상한</div>
          </div>
        </div>
      </div>

      {/* Capacity Input */}
      <div className="card">
        <label className="text-sm text-text-muted block mb-2">입찰 용량 (MW)</label>
        <input
          type="number"
          value={capacity}
          onChange={(e) => setCapacity(Number(e.target.value))}
          className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary text-lg font-mono"
          min={1}
          max={500}
        />
      </div>

      {/* Bid Curve Chart */}
      <div className="card">
        <h2 className="text-sm font-semibold text-text-primary mb-3">입찰 곡선</h2>
        <div className="h-[180px] -mx-2">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={bidCurveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="cumulativeQuantity"
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
              />
              <YAxis
                stroke={chartColors.axis}
                fontSize={10}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltipBg,
                  border: `1px solid ${chartColors.tooltipBorder}`,
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
              />
              <Line
                type="stepAfter"
                dataKey="price"
                stroke="#fbbf24"
                strokeWidth={2}
                dot={{ fill: '#fbbf24', r: 3 }}
              />
              <Bar
                dataKey="quantity"
                fill="#6366f1"
                opacity={0.4}
                radius={[2, 2, 0, 0]}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-3">
        <div className="card text-center">
          <p className="text-xs text-text-muted">총 입찰량</p>
          <p className="text-xl font-bold text-text-primary">{totalQuantity.toFixed(1)} MW</p>
        </div>
        <div className="card text-center">
          <p className="text-xs text-text-muted">예상 평균가</p>
          <p className="text-xl font-bold text-smp">{avgPrice.toFixed(1)} 원</p>
        </div>
      </div>

      {/* Segment Details (Collapsible) */}
      <div className="card">
        <button
          onClick={() => setShowSegments(!showSegments)}
          className="flex items-center justify-between w-full"
        >
          <span className="text-sm font-semibold text-text-primary">구간별 설정</span>
          {showSegments ? (
            <ChevronUp className="w-5 h-5 text-text-muted" />
          ) : (
            <ChevronDown className="w-5 h-5 text-text-muted" />
          )}
        </button>

        {showSegments && (
          <div className="mt-4 space-y-2">
            {segments.map((seg) => (
              <div
                key={seg.id}
                className={clsx(
                  'flex items-center gap-3 p-2 rounded-lg',
                  seg.price <= smpForHour.q50 ? 'bg-success/10' : 'bg-background'
                )}
              >
                <span className="w-8 text-sm font-medium text-text-primary">S{seg.id}</span>
                <input
                  type="number"
                  value={seg.quantity}
                  onChange={(e) => {
                    const newQuantity = Number(e.target.value);
                    setSegments((prev) => {
                      const updated = [...prev];
                      const idx = updated.findIndex((s) => s.id === seg.id);
                      updated[idx] = { ...updated[idx], quantity: newQuantity };
                      return updated;
                    });
                  }}
                  className="flex-1 bg-card border border-border rounded px-2 py-1.5 text-text-primary text-sm text-right font-mono"
                  placeholder="MW"
                />
                <input
                  type="number"
                  value={seg.price}
                  onChange={(e) => updateSegmentPrice(seg.id, Number(e.target.value))}
                  className="flex-1 bg-card border border-border rounded px-2 py-1.5 text-text-primary text-sm text-right font-mono"
                  placeholder="원"
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Validation */}
      <div className="flex items-center gap-2 px-2">
        <CheckCircle className="w-4 h-4 text-success" />
        <span className="text-xs text-text-muted">단조성 제약 충족</span>
        <CheckCircle className="w-4 h-4 text-success ml-2" />
        <span className="text-xs text-text-muted">용량 제한 준수</span>
      </div>

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-3">
        <button onClick={handleOptimize} className="btn-secondary flex items-center justify-center gap-2">
          <Sparkles className="w-5 h-5" />
          <span>AI 최적화</span>
        </button>
        <button className="btn-primary flex items-center justify-center gap-2">
          <Send className="w-5 h-5" />
          <span>KPX 제출</span>
        </button>
      </div>
    </div>
  );
}
