/**
 * Bidding Page - RE-BMS v6.1
 * 10-Segment Bidding Management for DAM/RTM
 * With internal review workflow and KPX submission
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CheckCircle,
  Save,
  Sparkles,
  FileCheck,
  Building2,
  Zap,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import BidReviewModal from '../components/Modals/BidReviewModal';
import type { BidStatus } from '../types';
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
import { useTheme } from '../contexts/ThemeContext';
import { apiService } from '../services/api';
import clsx from 'clsx';

interface BidSegment {
  id: number;
  quantity: number;
  price: number;
  clearingProbability?: number;  // AI optimization result
  expectedRevenue?: number;      // AI optimization result
}

export default function Bidding() {
  const navigate = useNavigate();
  const { data: forecast } = useSMPForecast();
  const { data: marketStatus } = useMarketStatus();
  const [selectedHour, setSelectedHour] = useState(12);
  const [riskLevel, setRiskLevel] = useState<'conservative' | 'moderate' | 'aggressive'>('moderate');
  const [capacity, setCapacity] = useState(50);
  const { isDark } = useTheme();

  // Bid submission state
  const [bidStatus, setBidStatus] = useState<BidStatus>('draft');
  const [isReviewModalOpen, setIsReviewModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  // AI Optimization state
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationError, setOptimizationError] = useState<string | null>(null);
  const [optimizationInfo, setOptimizationInfo] = useState<{
    modelUsed: string;
    method: string;
    totalExpectedRevenue: number;
  } | null>(null);

  // Theme-aware chart colors
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

  // AI optimization using backend API (BiLSTM+Attention v3.1)
  const handleOptimize = useCallback(async () => {
    setIsOptimizing(true);
    setOptimizationError(null);
    setOptimizationInfo(null);

    try {
      // Call the real AI optimization API
      const result = await apiService.getOptimizedSegments(capacity, riskLevel);

      // Find the hourly bid for the selected hour
      const hourlyBid = result.hourly_bids.find(bid => bid.hour === selectedHour);

      if (hourlyBid && hourlyBid.segments) {
        // Convert API response to local segment format
        const newSegments: BidSegment[] = hourlyBid.segments.map((seg, idx) => ({
          id: seg.segment_id || idx + 1,
          quantity: seg.quantity_mw,
          price: Math.round(seg.price_krw_mwh),
          // Include AI optimization fields if available
          clearingProbability: (seg as any).clearing_probability,
          expectedRevenue: (seg as any).expected_revenue,
        }));

        setSegments(newSegments);

        // Calculate total expected revenue for this hour
        const totalExpectedRevenue = newSegments.reduce(
          (sum, seg) => sum + (seg.expectedRevenue || 0),
          0
        );

        setOptimizationInfo({
          modelUsed: result.model_used || 'AI Optimizer',
          method: (result as any).optimization_method || 'quantile-based',
          totalExpectedRevenue,
        });

        setBidStatus('draft');
      } else {
        throw new Error(`No optimization data for hour ${selectedHour}`);
      }
    } catch (error) {
      console.error('AI optimization failed:', error);
      setOptimizationError(
        error instanceof Error ? error.message : 'AI optimization failed'
      );

      // Fallback to simple client-side optimization
      const basePrice = smpForHour.q10 * 0.9;
      const priceSpread = (smpForHour.q90 - smpForHour.q10) / 9;
      const newSegments = segments.map((seg, idx) => ({
        ...seg,
        price: Math.round(basePrice + idx * priceSpread),
        quantity: capacity / 10,
      }));
      setSegments(newSegments);
    } finally {
      setIsOptimizing(false);
    }
  }, [capacity, riskLevel, selectedHour, smpForHour, segments]);

  // Save draft
  const handleSaveDraft = () => {
    setIsSaving(true);
    setSaveMessage(null);
    // Simulate save
    setTimeout(() => {
      setIsSaving(false);
      setSaveMessage('임시 저장 완료');
      setTimeout(() => setSaveMessage(null), 3000);
    }, 1000);
  };

  // Submit for review (internal approval)
  const handleSubmitForReview = () => {
    setIsReviewModalOpen(true);
  };

  // Handle approval from review modal
  const handleApproved = () => {
    setBidStatus('approved');
    setIsReviewModalOpen(false);
  };

  // Handle rejection from review modal
  const handleRejected = () => {
    setBidStatus('draft');
    setIsReviewModalOpen(false);
  };

  // Submit to KPX (DAM - Day Ahead Market)
  const handleKPXSubmit = () => {
    // Navigate to KPX simulation page with bid data
    navigate('/kpx-simulation', {
      state: {
        segments,
        selectedHour,
        smpForecast: smpForHour,
      },
    });
  };

  // Submit to RTM (Real-Time Market)
  const handleRTMSubmit = () => {
    // Navigate to RTM simulation page with bid data
    navigate('/rtm-simulation', {
      state: {
        segments,
        selectedHour,
        smpForecast: smpForHour,
      },
    });
  };

  // Get status badge
  const getStatusBadge = () => {
    switch (bidStatus) {
      case 'draft':
        return { label: '작성 중', color: 'bg-gray-500/20 text-gray-400' };
      case 'pending_review':
        return { label: '검토 대기', color: 'bg-warning/20 text-warning' };
      case 'approved':
        return { label: 'KPX 제출 가능', color: 'bg-success/20 text-success' };
      case 'submitted_kpx':
        return { label: 'KPX 제출됨', color: 'bg-primary/20 text-primary' };
      default:
        return { label: '작성 중', color: 'bg-gray-500/20 text-gray-400' };
    }
  };

  const statusBadge = getStatusBadge();

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-text-primary">입찰 관리</h1>
            <span className={clsx('px-2.5 py-1 rounded-full text-xs font-medium', statusBadge.color)}>
              {statusBadge.label}
            </span>
          </div>
          <p className="text-text-muted mt-1">10-Segment 입찰가격 설정</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {/* Market Status */}
          <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
            <div className={clsx(
              'status-dot',
              marketStatus?.dam.status === 'open' ? 'status-success' : 'status-danger'
            )} />
            <span className="text-sm text-text-muted whitespace-nowrap">
              DAM {marketStatus?.dam.status === 'open' ? '거래 가능' : '마감'}
            </span>
            {marketStatus?.dam.hours_remaining && (
              <span className="text-xs text-warning whitespace-nowrap">
                {marketStatus.dam.hours_remaining}시간 남음
              </span>
            )}
          </div>
          <button
            onClick={handleOptimize}
            disabled={isOptimizing}
            className={clsx(
              'btn-secondary flex items-center gap-2 whitespace-nowrap',
              isOptimizing && 'opacity-70 cursor-wait'
            )}
          >
            {isOptimizing ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            <span className="hidden sm:inline">{isOptimizing ? 'AI 분석 중...' : 'AI 최적화'}</span>
            <span className="sm:hidden">{isOptimizing ? '...' : '최적화'}</span>
          </button>
          <button
            onClick={handleKPXSubmit}
            disabled={bidStatus !== 'approved'}
            className={clsx(
              'flex items-center gap-2 whitespace-nowrap px-4 py-2 rounded-lg font-medium transition-colors',
              bidStatus === 'approved'
                ? 'bg-success text-white hover:bg-success/90'
                : 'bg-background text-text-muted cursor-not-allowed'
            )}
          >
            <Building2 className="w-4 h-4" />
            <span className="hidden sm:inline">DAM 제출</span>
            <span className="sm:hidden">DAM</span>
          </button>
          <button
            onClick={handleRTMSubmit}
            disabled={bidStatus !== 'approved'}
            className={clsx(
              'flex items-center gap-2 whitespace-nowrap px-4 py-2 rounded-lg font-medium transition-colors',
              bidStatus === 'approved'
                ? 'bg-warning text-white hover:bg-warning/90'
                : 'bg-background text-text-muted cursor-not-allowed'
            )}
          >
            <Zap className="w-4 h-4" />
            <span className="hidden sm:inline">RTM 제출</span>
            <span className="sm:hidden">RTM</span>
          </button>
        </div>
      </div>

      {/* Status Alert */}
      {bidStatus === 'approved' && (
        <div className="flex items-center gap-3 p-4 bg-success/10 border border-success/30 rounded-lg">
          <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-success">입찰이 승인되었습니다</p>
            <p className="text-xs text-success/80 mt-0.5">'DAM 제출' 또는 'RTM 제출' 버튼을 클릭하여 시장 시뮬레이션을 실행하세요.</p>
          </div>
        </div>
      )}

      {/* AI Optimization Info */}
      {optimizationInfo && (
        <div className="flex items-center gap-3 p-4 bg-primary/10 border border-primary/30 rounded-lg">
          <Sparkles className="w-5 h-5 text-primary flex-shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-medium text-primary">AI 최적화 완료</p>
            <p className="text-xs text-primary/80 mt-0.5">
              모델: {optimizationInfo.modelUsed} | 방식: {optimizationInfo.method}
              {optimizationInfo.totalExpectedRevenue > 0 && (
                <> | 예상 수익: {(optimizationInfo.totalExpectedRevenue / 1000000).toFixed(2)}백만원</>
              )}
            </p>
          </div>
        </div>
      )}

      {/* Optimization Error */}
      {optimizationError && (
        <div className="flex items-center gap-3 p-4 bg-warning/10 border border-warning/30 rounded-lg">
          <AlertCircle className="w-5 h-5 text-warning flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-warning">AI 최적화 실패 (대체 알고리즘 사용)</p>
            <p className="text-xs text-warning/80 mt-0.5">{optimizationError}</p>
          </div>
        </div>
      )}

      {/* Settings Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Hour Selection */}
        <div className="card">
          <label className="text-sm text-text-muted block mb-2">거래 시간대</label>
          <select
            value={selectedHour}
            onChange={(e) => setSelectedHour(Number(e.target.value))}
            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
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
          <label className="text-sm text-text-muted block mb-2">입찰 용량 (MW)</label>
          <input
            type="number"
            value={capacity}
            onChange={(e) => setCapacity(Number(e.target.value))}
            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
            min={1}
            max={500}
          />
        </div>

        {/* Risk Level */}
        <div className="card">
          <label className="text-sm text-text-muted block mb-2">위험 선호도</label>
          <div className="flex gap-2">
            {(['conservative', 'moderate', 'aggressive'] as const).map((level) => (
              <button
                key={level}
                onClick={() => setRiskLevel(level)}
                className={clsx(
                  'flex-1 px-3 py-2 text-sm rounded-lg transition-colors',
                  riskLevel === level
                    ? 'bg-primary text-text-primary'
                    : 'bg-background text-text-muted hover:bg-background/80'
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
          <label className="text-sm text-text-muted block mb-2">SMP 예측 (원/kWh)</label>
          <div className="flex items-center justify-between">
            <div className="text-center">
              <div className="text-success text-sm">{smpForHour.q10.toFixed(0)}</div>
              <div className="text-xs text-text-muted">하한</div>
            </div>
            <div className="text-center">
              <div className="text-smp text-xl font-bold">{smpForHour.q50.toFixed(0)}</div>
              <div className="text-xs text-text-muted">예측</div>
            </div>
            <div className="text-center">
              <div className="text-danger text-sm">{smpForHour.q90.toFixed(0)}</div>
              <div className="text-xs text-text-muted">상한</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bid Matrix */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">10-Segment 입찰 매트릭스</h3>
          <div className="space-y-2">
            <div className="grid grid-cols-5 gap-2 text-xs text-text-muted font-medium px-2">
              <span>구간</span>
              <span className="text-right">물량 (MW)</span>
              <span className="text-right">가격 (원/kWh)</span>
              <span className="text-right">낙찰확률</span>
              <span className="text-right">예상수익</span>
            </div>
            {segments.map((seg, idx) => (
              <div
                key={seg.id}
                className={clsx(
                  'grid grid-cols-5 gap-2 items-center p-2 rounded-lg transition-colors',
                  seg.price <= smpForHour.q50 ? 'bg-success/10 border border-success/20' : 'bg-background'
                )}
              >
                <span className="text-text-primary font-medium">Seg {seg.id}</span>
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
                  className="w-full bg-card border border-border rounded px-2 py-1 text-text-primary text-right text-sm"
                />
                <input
                  type="number"
                  value={seg.price}
                  onChange={(e) => updateSegmentPrice(seg.id, Number(e.target.value))}
                  className="w-full bg-card border border-border rounded px-2 py-1 text-text-primary text-right text-sm"
                />
                <span className={clsx(
                  'text-right text-sm font-mono',
                  seg.clearingProbability !== undefined
                    ? seg.clearingProbability >= 0.7 ? 'text-success' : seg.clearingProbability >= 0.4 ? 'text-warning' : 'text-danger'
                    : 'text-text-muted'
                )}>
                  {seg.clearingProbability !== undefined
                    ? `${(seg.clearingProbability * 100).toFixed(0)}%`
                    : '-'}
                </span>
                <span className="text-right text-sm text-text-muted font-mono">
                  {seg.expectedRevenue !== undefined
                    ? `${(seg.expectedRevenue / 1000).toFixed(0)}K`
                    : `${((seg.quantity * seg.price) / 1000).toFixed(1)}K`}
                </span>
              </div>
            ))}
          </div>

          {/* Total */}
          <div className="mt-4 pt-4 border-t border-border">
            <div className="flex justify-between items-center">
              <span className="text-text-muted">총 입찰량</span>
              <span className="text-xl font-bold text-text-primary">
                {segments.reduce((sum, s) => sum + s.quantity, 0).toFixed(1)} MW
              </span>
            </div>
            <div className="flex justify-between items-center mt-2">
              <span className="text-text-muted">예상 평균가</span>
              <span className="text-xl font-bold text-smp">
                {(segments.reduce((sum, s) => sum + s.price * s.quantity, 0) /
                  segments.reduce((sum, s) => sum + s.quantity, 0)).toFixed(1)} 원/kWh
              </span>
            </div>
          </div>
        </div>

        {/* Bid Curve Chart */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">입찰 곡선 (Step Chart)</h3>
          <div className="h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={bidCurveData} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                <XAxis
                  dataKey="cumulativeQuantity"
                  stroke={chartColors.axis}
                  fontSize={14}
                  tickLine={false}
                  label={{ value: '누적 물량 (MW)', position: 'insideBottom', offset: -10, fill: chartColors.axis, fontSize: 14, fontWeight: 500 }}
                />
                <YAxis
                  stroke={chartColors.axis}
                  fontSize={14}
                  tickLine={false}
                  width={60}
                  label={{ value: '가격 (원/kWh)', angle: -90, position: 'insideLeft', fill: chartColors.axis, fontSize: 14, fontWeight: 500 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: chartColors.tooltipBg,
                    border: `1px solid ${chartColors.tooltipBorder}`,
                    borderRadius: '8px',
                    fontSize: '15px',
                  }}
                  labelStyle={{ color: isDark ? '#fff' : '#000', fontSize: '15px', fontWeight: 600 }}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  wrapperStyle={{ paddingBottom: 10, fontSize: 15 }}
                  formatter={(value) => <span className="text-text-muted font-medium">{value}</span>}
                />

                {/* SMP Reference Lines */}
                <Line
                  type="stepAfter"
                  dataKey="price"
                  stroke="#fbbf24"
                  strokeWidth={3}
                  name="입찰가격"
                  dot={{ fill: '#fbbf24', strokeWidth: 2, r: 4 }}
                />

                {/* Bar for quantity */}
                <Bar
                  dataKey="quantity"
                  fill="#6366f1"
                  opacity={0.5}
                  name="구간 물량"
                  radius={[2, 2, 0, 0]}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* SMP Reference Overlay */}
          <div className="mt-4 pt-4 border-t border-border flex flex-wrap items-center justify-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-6 h-2 bg-success rounded" />
              <span className="text-base text-text-muted whitespace-nowrap">SMP 하한 ({smpForHour.q10.toFixed(0)})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-2 bg-smp rounded" />
              <span className="text-base text-text-muted whitespace-nowrap">SMP 예측 ({smpForHour.q50.toFixed(0)})</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-6 h-2 bg-danger rounded" />
              <span className="text-base text-text-muted whitespace-nowrap">SMP 상한 ({smpForHour.q90.toFixed(0)})</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Actions */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex flex-wrap items-center gap-3 sm:gap-4">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
              <span className="text-sm text-text-muted whitespace-nowrap">단조성 제약 충족</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success flex-shrink-0" />
              <span className="text-sm text-text-muted whitespace-nowrap">용량 제한 준수</span>
            </div>
            {saveMessage && (
              <div className="flex items-center gap-2 text-success">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm">{saveMessage}</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 sm:gap-3">
            <button
              onClick={handleSaveDraft}
              disabled={isSaving}
              className="btn-secondary flex items-center gap-2 whitespace-nowrap"
            >
              <Save className={clsx('w-4 h-4', isSaving && 'animate-spin')} />
              <span className="hidden sm:inline">{isSaving ? '저장 중...' : '임시 저장'}</span>
              <span className="sm:hidden">{isSaving ? '...' : '저장'}</span>
            </button>
            <button
              onClick={handleSubmitForReview}
              disabled={bidStatus === 'approved'}
              className={clsx(
                'flex items-center gap-2 whitespace-nowrap px-4 py-2 rounded-lg font-medium transition-colors',
                bidStatus === 'approved'
                  ? 'bg-background text-text-muted cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary/90'
              )}
            >
              <FileCheck className="w-4 h-4" />
              <span className="hidden sm:inline">
                {bidStatus === 'approved' ? '승인됨' : '입찰 제출'}
              </span>
              <span className="sm:hidden">
                {bidStatus === 'approved' ? '승인' : '제출'}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Review Modal */}
      <BidReviewModal
        isOpen={isReviewModalOpen}
        onClose={() => setIsReviewModalOpen(false)}
        onApprove={handleApproved}
        onReject={handleRejected}
        segments={segments}
        selectedHour={selectedHour}
        smpForecast={smpForHour}
        capacity={capacity}
      />
    </div>
  );
}
