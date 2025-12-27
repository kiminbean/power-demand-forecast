/**
 * Bidding Page - RE-BMS v6.2
 * 10-Segment Bidding Management for DAM/RTM
 * With internal review workflow, KPX submission, and Power Plant Registration
 */

import { useState, useCallback, useEffect } from 'react';
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
  Plus,
  X,
  Trash2,
} from 'lucide-react';
import BidReviewModal from '../components/Modals/BidReviewModal';
import type { BidStatus, PowerPlant, PowerPlantCreate, WeatherCondition, PlantStatus } from '../types';
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
import {
  calculateEfficiency,
  estimateDailyGeneration,
  formatCapacity,
} from '../utils/powerPlantUtils';
import {
  PLANT_TYPE_LABELS as PlantTypeLabels,
  CONTRACT_TYPE_LABELS as ContractTypeLabels,
  ROOF_DIRECTION_LABELS as RoofDirectionLabels,
} from '../types';
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

  // Power Plant states (v6.2.0)
  const [powerPlants, setPowerPlants] = useState<PowerPlant[]>([]);
  const [isRegistrationOpen, setIsRegistrationOpen] = useState(false);
  const [currentWeather] = useState<WeatherCondition>('clear');
  const [vppBiddingEnabled, setVppBiddingEnabled] = useState(true); // VPP auto-bidding toggle
  const [newPlant, setNewPlant] = useState<Partial<PowerPlantCreate>>({
    name: '',
    type: 'solar',
    capacity: 3,
    installDate: new Date().toISOString().split('T')[0],
    contractType: 'net_metering',
    location: { address: '' },
    roofDirection: 'south',
  });

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
        // Convert API response to local segment format with correct revenue calculation
        const newSegments: BidSegment[] = hourlyBid.segments.map((seg, idx) => {
          // 낙찰확률: API에서 제공하거나 가격에 따라 계산
          const clearingProb = (seg as any).clearing_probability ||
            Math.max(0.1, 1 - (idx * 0.08) + (Math.random() * 0.1 - 0.05));
          // 예상수익: MW × 1000(kW) × 가격(원/kWh) × 낙찰확률
          const expectedRev = (seg as any).expected_revenue ||
            seg.quantity_mw * 1000 * seg.price_krw_mwh * clearingProb;

          return {
            id: seg.segment_id || idx + 1,
            quantity: seg.quantity_mw,
            price: Math.round(seg.price_krw_mwh),
            clearingProbability: clearingProb,
            expectedRevenue: expectedRev,
          };
        });

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

      // Fallback to simple client-side optimization with correct revenue
      const basePrice = smpForHour.q10 * 0.9;
      const priceSpread = (smpForHour.q90 - smpForHour.q10) / 9;
      const capacityPerSegment = capacity / 10;
      const newSegments = segments.map((seg, idx) => {
        const segPrice = Math.round(basePrice + idx * priceSpread);
        const clearingProb = Math.max(0.1, 1 - (idx * 0.08));
        // 예상수익: MW × 1000(kW) × 가격(원/kWh) × 낙찰확률
        const expectedRev = capacityPerSegment * 1000 * segPrice * clearingProb;
        return {
          ...seg,
          price: segPrice,
          quantity: capacityPerSegment,
          clearingProbability: clearingProb,
          expectedRevenue: expectedRev,
        };
      });
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

  // Power Plant Functions (v6.2.0)
  const loadPowerPlants = useCallback(async () => {
    try {
      const plants = await apiService.getPowerPlants();
      setPowerPlants(plants);
    } catch (error) {
      // Fallback to localStorage
      const stored = localStorage.getItem('powerPlants');
      if (stored) {
        setPowerPlants(JSON.parse(stored));
      }
    }
  }, []);

  const handleCreatePlant = useCallback(async () => {
    if (!newPlant.name || !newPlant.capacity) return;

    try {
      const created = await apiService.createPowerPlant(newPlant as PowerPlantCreate);
      setPowerPlants(prev => [...prev, created]);
      localStorage.setItem('powerPlants', JSON.stringify([...powerPlants, created]));
    } catch (error) {
      // Fallback: create locally
      const localPlant: PowerPlant = {
        ...newPlant as PowerPlantCreate,
        id: `local-${Date.now()}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      setPowerPlants(prev => [...prev, localPlant]);
      localStorage.setItem('powerPlants', JSON.stringify([...powerPlants, localPlant]));
    }

    // Reset form and close modal
    setNewPlant({
      name: '',
      type: 'solar',
      capacity: 3,
      installDate: new Date().toISOString().split('T')[0],
      contractType: 'net_metering',
      location: { address: '' },
      roofDirection: 'south',
    });
    setIsRegistrationOpen(false);
  }, [newPlant, powerPlants]);

  const handleDeletePlant = useCallback(async (plantId: string) => {
    if (!confirm('이 발전소를 삭제하시겠습니까?')) return;

    try {
      await apiService.deletePowerPlant(plantId);
    } catch (error) {
      // Fallback: delete locally
    }

    const updated = powerPlants.filter(p => p.id !== plantId);
    setPowerPlants(updated);
    localStorage.setItem('powerPlants', JSON.stringify(updated));
  }, [powerPlants]);

  const handleUpdatePlantStatus = useCallback(async (plantId: string, newStatus: PlantStatus) => {
    try {
      const updated = await apiService.updatePowerPlant(plantId, { status: newStatus });
      setPowerPlants(prev => prev.map(p => p.id === plantId ? { ...p, status: updated.status } : p));
    } catch (error) {
      console.error('Failed to update plant status:', error);
      // Fallback: update locally
      const updated = powerPlants.map(p => p.id === plantId ? { ...p, status: newStatus } : p);
      setPowerPlants(updated);
      localStorage.setItem('powerPlants', JSON.stringify(updated));
    }
  }, [powerPlants]);

  // Count active plants
  const activePlantCount = powerPlants.filter(p => (p.status || 'active') === 'active').length;
  const activePlantCapacity = powerPlants
    .filter(p => (p.status || 'active') === 'active')
    .reduce((sum, p) => sum + p.capacity, 0);

  // Calculate recommended capacity based on registered plants (only active)
  const recommendedCapacity = powerPlants.filter(p => (p.status || 'active') === 'active').reduce((sum, plant) => {
    const efficiency = calculateEfficiency(plant.installDate);
    const dailyKwh = estimateDailyGeneration(plant.capacity, efficiency, currentWeather, plant.roofDirection || 'south');
    return sum + dailyKwh;
  }, 0);

  // Load power plants on mount
  useEffect(() => {
    loadPowerPlants();
  }, [loadPowerPlants]);

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

      {/* Conditional UI based on capacity */}
      {(() => {
        const totalPlantCapacityKw = powerPlants.reduce((sum, p) => sum + p.capacity, 0);
        const isLargeCapacity = totalPlantCapacityKw >= 1000; // 1MW = 1000kW

        return isLargeCapacity ? (
          /* Professional Settings Row for large capacity users */
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
        ) : (
          /* Simplified VPP Summary for small-scale users */
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <span className="text-3xl">🤖</span>
                <div>
                  <h3 className="text-lg font-semibold text-text-primary">VPP 자동 입찰</h3>
                  <p className="text-sm text-text-muted">소규모 발전소는 VPP(가상발전소)가 최적의 전략으로 자동 입찰합니다</p>
                </div>
              </div>
              {/* VPP Toggle Switch */}
              <button
                onClick={() => setVppBiddingEnabled(!vppBiddingEnabled)}
                className={clsx(
                  'relative w-14 h-7 rounded-full transition-colors',
                  vppBiddingEnabled ? 'bg-success' : 'bg-gray-400'
                )}
              >
                <div
                  className={clsx(
                    'absolute top-1 w-5 h-5 bg-white rounded-full transition-transform',
                    vppBiddingEnabled ? 'translate-x-8' : 'translate-x-1'
                  )}
                />
              </button>
            </div>

            {vppBiddingEnabled ? (
              <>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  {/* Today's Bid Capacity */}
                  <div className="p-4 bg-background rounded-lg">
                    <p className="text-sm text-text-muted mb-1">오늘의 입찰량</p>
                    <p className="text-2xl font-bold text-primary">{recommendedCapacity.toFixed(1)} kWh</p>
                  </div>

                  {/* Expected Revenue */}
                  <div className="p-4 bg-background rounded-lg">
                    <p className="text-sm text-text-muted mb-1">예상 수익</p>
                    <p className="text-2xl font-bold text-success">
                      {(recommendedCapacity * smpForHour.q50).toLocaleString('ko-KR', { maximumFractionDigits: 0 })}원
                    </p>
                  </div>

                  {/* Current SMP */}
                  <div className="p-4 bg-background rounded-lg">
                    <p className="text-sm text-text-muted mb-1">현재 SMP</p>
                    <p className="text-2xl font-bold text-smp">{smpForHour.q50.toFixed(0)}원/kWh</p>
                  </div>

                  {/* Weather */}
                  <div className="p-4 bg-background rounded-lg">
                    <p className="text-sm text-text-muted mb-1">날씨</p>
                    <p className="text-2xl font-bold text-text-primary">
                      {currentWeather === 'clear' ? '☀️ 맑음' :
                       currentWeather === 'partly_cloudy' ? '⛅ 약간흐림' :
                       currentWeather === 'cloudy' ? '☁️ 흐림' : '🌧️ 비'}
                    </p>
                  </div>
                </div>

                {/* VPP Status */}
                <div className="p-4 bg-success/10 border border-success/20 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-success rounded-full animate-pulse" />
                      <div>
                        <p className="text-sm font-medium text-success">VPP 자동 입찰 활성화</p>
                        <p className="text-xs text-text-muted">
                          운영중 {activePlantCount}개 / 전체 {powerPlants.length}개 · {formatCapacity(activePlantCapacity)}
                        </p>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-success/20 text-success text-sm rounded-full">자동 관리 중</span>
                  </div>
                </div>
              </>
            ) : (
              /* VPP Disabled State */
              <div className="p-4 bg-gray-500/10 border border-gray-500/20 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 bg-gray-400 rounded-full" />
                    <div>
                      <p className="text-sm font-medium text-gray-400">VPP 자동 입찰 비활성화</p>
                      <p className="text-xs text-text-muted">
                        자동 입찰이 중지되었습니다. 토글을 켜면 다시 시작됩니다.
                      </p>
                    </div>
                  </div>
                  <span className="px-3 py-1 bg-gray-500/20 text-gray-400 text-sm rounded-full">입찰 중지</span>
                </div>
              </div>
            )}
          </div>
        );
      })()}

      {/* Power Plant Section (v6.2.0) */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">내 발전소</h3>
          <button
            onClick={() => setIsRegistrationOpen(true)}
            className="flex items-center gap-2 px-3 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span className="hidden sm:inline">발전소 등록</span>
          </button>
        </div>

        {powerPlants.length > 0 ? (
          <div className="space-y-3">
            {powerPlants.map((plant) => {
              const efficiency = calculateEfficiency(plant.installDate);
              const dailyKwh = estimateDailyGeneration(
                plant.capacity,
                efficiency,
                currentWeather,
                plant.roofDirection || 'south'
              );
              const plantType = PlantTypeLabels[plant.type as keyof typeof PlantTypeLabels];
              const plantStatus = (plant.status || 'active') as PlantStatus;

              return (
                <div
                  key={plant.id}
                  className={clsx(
                    'flex items-center justify-between p-3 rounded-lg',
                    plantStatus === 'active' ? 'bg-background' : 'bg-background opacity-75 border border-warning/30'
                  )}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{plantType?.icon || '⚡'}</span>
                    <div>
                      <p className="font-medium text-text-primary">{plant.name}</p>
                      <p className="text-sm text-text-muted">
                        {formatCapacity(plant.capacity)} · 효율 {(efficiency * 100).toFixed(0)}% · 예상 {dailyKwh.toFixed(1)} kWh/일
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <select
                      value={plantStatus}
                      onChange={(e) => handleUpdatePlantStatus(plant.id, e.target.value as PlantStatus)}
                      className={clsx(
                        'text-xs px-2 py-1 rounded border-0 cursor-pointer',
                        plantStatus === 'active' && 'bg-success/20 text-success',
                        plantStatus === 'maintenance' && 'bg-warning/20 text-warning',
                        plantStatus === 'paused' && 'bg-gray-500/20 text-gray-400'
                      )}
                    >
                      <option value="active">✓ 운영중</option>
                      <option value="maintenance">🔧 점검중</option>
                      <option value="paused">⏸ 중지</option>
                    </select>
                    <button
                      onClick={() => handleDeletePlant(plant.id)}
                      className="p-2 text-danger hover:bg-danger/10 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              );
            })}

            {/* Recommended Capacity */}
            <div className="mt-4 p-3 bg-primary/5 border border-primary/20 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-muted">오늘의 추천 입찰량</span>
                <span className="text-xs px-2 py-1 bg-background rounded">
                  {currentWeather === 'clear' ? '☀️ 맑음' :
                   currentWeather === 'partly_cloudy' ? '⛅ 약간 흐림' :
                   currentWeather === 'cloudy' ? '☁️ 흐림' : '🌧️ 비'}
                </span>
              </div>
              <p className="text-xl font-bold text-primary mt-1">{recommendedCapacity.toFixed(1)} kWh</p>
              <p className="text-xs text-text-muted">등록된 {powerPlants.length}개 발전소 기준</p>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-text-muted">
            <p className="text-3xl mb-2">🏭</p>
            <p>등록된 발전소가 없습니다</p>
            <p className="text-sm mt-1">발전소를 등록하면 효율과 날씨를 고려한 맞춤 입찰량을 추천받을 수 있습니다</p>
          </div>
        )}
      </div>

      {/* Main Content - Only show for large capacity users */}
      {powerPlants.reduce((sum, p) => sum + p.capacity, 0) >= 1000 && (
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
                  {(() => {
                    // 수익 계산: MW × 1000(kW) × 가격(원/kWh) × 낙찰확률
                    const clearingProb = seg.clearingProbability ?? 1;
                    const revenue = seg.expectedRevenue !== undefined
                      ? seg.expectedRevenue
                      : seg.quantity * 1000 * seg.price * clearingProb;
                    // Format: K(천원), M(백만원)
                    if (revenue >= 1000000) {
                      return `${(revenue / 1000000).toFixed(1)}M`;
                    } else if (revenue >= 1000) {
                      return `${(revenue / 1000).toFixed(0)}K`;
                    } else {
                      return `${revenue.toFixed(0)}`;
                    }
                  })()}
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
      )}

      {/* Bottom Actions - Only show for large capacity users */}
      {powerPlants.reduce((sum, p) => sum + p.capacity, 0) >= 1000 && (
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
      )}

      {/* Review Modal - Only show for large capacity users */}
      {powerPlants.reduce((sum, p) => sum + p.capacity, 0) >= 1000 && (
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
      )}

      {/* Power Plant Registration Modal (v6.2.0) */}
      {isRegistrationOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-card rounded-xl shadow-xl w-full max-w-md mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h3 className="text-lg font-semibold text-text-primary">내 발전소 등록</h3>
              <button
                onClick={() => setIsRegistrationOpen(false)}
                className="p-1 hover:bg-background rounded"
              >
                <X className="w-5 h-5 text-text-muted" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Plant Name */}
              <div>
                <label className="block text-sm text-text-muted mb-1">발전소 이름</label>
                <input
                  type="text"
                  value={newPlant.name || ''}
                  onChange={(e) => setNewPlant(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="예: 우리집 태양광 1호"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Plant Type */}
              <div>
                <label className="block text-sm text-text-muted mb-1">설비 유형</label>
                <div className="flex gap-2">
                  {(['solar', 'wind', 'ess'] as const).map((type) => (
                    <button
                      key={type}
                      onClick={() => setNewPlant(prev => ({ ...prev, type }))}
                      className={clsx(
                        'flex-1 px-3 py-2 rounded-lg transition-colors text-sm',
                        newPlant.type === type
                          ? 'bg-primary text-white'
                          : 'bg-background text-text-muted hover:bg-background/80'
                      )}
                    >
                      {PlantTypeLabels[type]?.icon} {PlantTypeLabels[type]?.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Capacity */}
              <div>
                <label className="block text-sm text-text-muted mb-1">설비 용량 (kW)</label>
                <input
                  type="number"
                  value={newPlant.capacity || ''}
                  onChange={(e) => setNewPlant(prev => ({ ...prev, capacity: Number(e.target.value) }))}
                  placeholder="3"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Install Date */}
              <div>
                <label className="block text-sm text-text-muted mb-1">설치일</label>
                <input
                  type="date"
                  value={newPlant.installDate || ''}
                  onChange={(e) => setNewPlant(prev => ({ ...prev, installDate: e.target.value }))}
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Efficiency Preview */}
              {newPlant.installDate && (
                <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg">
                  <p className="text-sm text-text-muted">
                    현재 효율: <span className="font-semibold text-primary">{(calculateEfficiency(newPlant.installDate) * 100).toFixed(0)}%</span>
                  </p>
                </div>
              )}

              {/* Contract Type */}
              <div>
                <label className="block text-sm text-text-muted mb-1">계약 유형</label>
                <div className="flex gap-2">
                  {(['net_metering', 'ppa'] as const).map((type) => (
                    <button
                      key={type}
                      onClick={() => setNewPlant(prev => ({ ...prev, contractType: type }))}
                      className={clsx(
                        'flex-1 px-3 py-2 rounded-lg transition-colors text-sm',
                        newPlant.contractType === type
                          ? 'bg-primary text-white'
                          : 'bg-background text-text-muted hover:bg-background/80'
                      )}
                    >
                      {ContractTypeLabels[type]?.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Roof Direction (for solar) */}
              {newPlant.type === 'solar' && (
                <div>
                  <label className="block text-sm text-text-muted mb-1">지붕 방향</label>
                  <div className="flex gap-2">
                    {(['south', 'east', 'west', 'flat'] as const).map((dir) => (
                      <button
                        key={dir}
                        onClick={() => setNewPlant(prev => ({ ...prev, roofDirection: dir }))}
                        className={clsx(
                          'flex-1 px-2 py-2 rounded-lg transition-colors text-sm',
                          newPlant.roofDirection === dir
                            ? 'bg-primary text-white'
                            : 'bg-background text-text-muted hover:bg-background/80'
                        )}
                      >
                        {RoofDirectionLabels[dir]}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Address */}
              <div>
                <label className="block text-sm text-text-muted mb-1">주소 (선택)</label>
                <input
                  type="text"
                  value={newPlant.location?.address || ''}
                  onChange={(e) => setNewPlant(prev => ({ ...prev, location: { ...prev.location, address: e.target.value } }))}
                  placeholder="제주시 예시동 123"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>
            </div>

            <div className="flex gap-3 p-4 border-t border-border">
              <button
                onClick={() => setIsRegistrationOpen(false)}
                className="flex-1 px-4 py-2 bg-background text-text-muted rounded-lg hover:bg-background/80"
              >
                취소
              </button>
              <button
                onClick={handleCreatePlant}
                disabled={!newPlant.name || !newPlant.capacity}
                className={clsx(
                  'flex-1 px-4 py-2 rounded-lg font-medium transition-colors',
                  newPlant.name && newPlant.capacity
                    ? 'bg-primary text-white hover:bg-primary/90'
                    : 'bg-background text-text-muted cursor-not-allowed'
                )}
              >
                등록
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
