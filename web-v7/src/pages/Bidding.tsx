/**
 * Bidding Page - RE-BMS v6.2
 * 10-Segment Bidding Management for DAM/RTM
 * With internal review workflow, KPX submission, and Power Plant Registration
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CheckCircle,
  Save,
  Sparkles,
  FileCheck,
  Building2,
  Zap,
  Plus,
  X,
  Trash2,
} from 'lucide-react';
import BidReviewModal from '../components/Modals/BidReviewModal';
import type { BidStatus, PowerPlant, PowerPlantCreate, PlantType, ContractType, RoofDirection, WeatherCondition, PlantStatus } from '../types';
import { PLANT_TYPE_LABELS, CONTRACT_TYPE_LABELS, ROOF_DIRECTION_LABELS, PLANT_STATUS_LABELS } from '../types';
import { apiService } from '../services/api';
import {
  calculateEfficiency,
  estimateDailyGeneration,
  getEfficiencyStatus,
  formatCapacity,
  formatRevenue,

} from '../utils/powerPlantUtils';
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
import clsx from 'clsx';

interface BidSegment {
  id: number;
  quantity: number;
  price: number;
  clearingProbability?: number;
  expectedRevenue?: number;
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

  // Power Plant state (v6.2.0)
  const [powerPlants, setPowerPlants] = useState<PowerPlant[]>([]);
  const [showRegistration, setShowRegistration] = useState(false);
  const [currentWeather] = useState<WeatherCondition>('clear');
  const [isLoadingPlants, setIsLoadingPlants] = useState(false);
  const [vppBiddingEnabled, setVppBiddingEnabled] = useState(true); // VPP auto-bidding toggle

  // Registration form state
  const [plantName, setPlantName] = useState('');
  const [plantType, setPlantType] = useState<PlantType>('solar');
  const [plantCapacity, setPlantCapacity] = useState('3');
  const [installYear, setInstallYear] = useState(new Date().getFullYear());
  const [installMonth, setInstallMonth] = useState(1);
  const [contractType, setContractType] = useState<ContractType>('net_metering');
  const [roofDirection, setRoofDirection] = useState<RoofDirection>('south');
  const [plantAddress, setPlantAddress] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Load power plants on mount
  useEffect(() => {
    loadPowerPlants();
  }, []);

  const loadPowerPlants = async () => {
    setIsLoadingPlants(true);
    try {
      const plants = await apiService.getPowerPlants();
      setPowerPlants(plants);
    } catch (error) {
      console.error('Failed to load power plants:', error);
      // Load from localStorage as fallback
      const saved = localStorage.getItem('powerPlants');
      if (saved) {
        setPowerPlants(JSON.parse(saved));
      }
    } finally {
      setIsLoadingPlants(false);
    }
  };

  const handleRegisterPlant = async () => {
    if (!plantName.trim() || !plantCapacity) return;

    setIsSubmitting(true);
    const newPlant: PowerPlantCreate = {
      name: plantName.trim(),
      type: plantType,
      capacity: parseFloat(plantCapacity),
      installDate: `${installYear}-${String(installMonth).padStart(2, '0')}-01`,
      contractType,
      location: { address: plantAddress },
      roofDirection,
    };

    try {
      const created = await apiService.createPowerPlant(newPlant);
      setPowerPlants((prev) => [...prev, created]);
      resetRegistrationForm();
      setShowRegistration(false);
    } catch (error) {
      console.error('Failed to create power plant:', error);
      // Fallback: save locally
      const localPlant: PowerPlant = {
        ...newPlant,
        id: `local-${Date.now()}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      const updated = [...powerPlants, localPlant];
      setPowerPlants(updated);
      localStorage.setItem('powerPlants', JSON.stringify(updated));
      resetRegistrationForm();
      setShowRegistration(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeletePlant = async (plantId: string) => {
    try {
      await apiService.deletePowerPlant(plantId);
      setPowerPlants((prev) => prev.filter((p) => p.id !== plantId));
    } catch (error) {
      console.error('Failed to delete power plant:', error);
      // Fallback: delete locally
      const updated = powerPlants.filter((p) => p.id !== plantId);
      setPowerPlants(updated);
      localStorage.setItem('powerPlants', JSON.stringify(updated));
    }
  };

  const handleUpdatePlantStatus = async (plantId: string, newStatus: PlantStatus) => {
    try {
      const updated = await apiService.updatePowerPlant(plantId, { status: newStatus });
      setPowerPlants((prev) => prev.map((p) => p.id === plantId ? { ...p, status: updated.status } : p));
    } catch (error) {
      console.error('Failed to update plant status:', error);
      // Fallback: update locally
      const updated = powerPlants.map((p) => p.id === plantId ? { ...p, status: newStatus } : p);
      setPowerPlants(updated);
      localStorage.setItem('powerPlants', JSON.stringify(updated));
    }
  };

  const resetRegistrationForm = () => {
    setPlantName('');
    setPlantType('solar');
    setPlantCapacity('3');
    setInstallYear(new Date().getFullYear());
    setInstallMonth(1);
    setContractType('net_metering');
    setRoofDirection('south');
    setPlantAddress('');
  };

  // Calculate total recommended capacity based on registered plants (only active plants)
  const getRecommendedCapacity = (): number => {
    return powerPlants
      .filter((plant) => (plant.status || 'active') === 'active')
      .reduce((total, plant) => {
        const efficiency = calculateEfficiency(plant.installDate);
        const dailyKwh = estimateDailyGeneration(
          plant.capacity,
          efficiency,
          currentWeather,
          plant.roofDirection || 'south'
        );
        return total + dailyKwh;
      }, 0);
  };

  // Count active plants for display
  const activePlantCount = powerPlants.filter((p) => (p.status || 'active') === 'active').length;
  const activePlantCapacity = powerPlants
    .filter((p) => (p.status || 'active') === 'active')
    .reduce((sum, p) => sum + p.capacity, 0);

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

  // AI optimization
  const handleOptimize = () => {
    // Apply optimization based on SMP forecast
    const basePrice = smpForHour.q10 * 0.9;
    const priceSpread = (smpForHour.q90 - smpForHour.q10) / 9;
    const capacityPerSegment = capacity / 10;

    const newSegments = segments.map((seg, idx) => {
      const segPrice = Math.round(basePrice + idx * priceSpread);
      // Higher segments have lower clearing probability
      const clearingProb = Math.max(0.1, 1 - (idx * 0.08));
      // Revenue: MW × 1000(kW) × price(원/kWh) × clearingProbability
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
    setBidStatus('draft');
  };

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
          <button onClick={handleOptimize} className="btn-secondary flex items-center gap-2 whitespace-nowrap">
            <Sparkles className="w-4 h-4" />
            <span className="hidden sm:inline">AI 최적화</span>
            <span className="sm:hidden">최적화</span>
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

      {/* Power Plant Section (v6.2.0) */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-text-primary">내 발전소</h3>
          <button
            onClick={() => setShowRegistration(true)}
            className="btn-primary flex items-center gap-2 text-sm"
          >
            <Plus className="w-4 h-4" />
            발전소 등록
          </button>
        </div>

        {isLoadingPlants ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
          </div>
        ) : powerPlants.length === 0 ? (
          <div className="text-center py-8 text-text-muted">
            <p>등록된 발전소가 없습니다.</p>
            <p className="text-sm mt-1">발전소를 등록하면 최적 입찰량을 추천받을 수 있습니다.</p>
          </div>
        ) : (
          <>
            {/* Plant Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
              {powerPlants.map((plant) => {
                const efficiency = calculateEfficiency(plant.installDate);
                const dailyKwh = estimateDailyGeneration(
                  plant.capacity,
                  efficiency,
                  currentWeather,
                  plant.roofDirection || 'south'
                );
                const installYear = new Date(plant.installDate).getFullYear();
                const yearsOld = new Date().getFullYear() - installYear;
                const plantStatus = (plant.status || 'active') as PlantStatus;
                const statusInfo = PLANT_STATUS_LABELS[plantStatus];

                return (
                  <div
                    key={plant.id}
                    className={clsx(
                      'bg-background border rounded-lg p-4 relative group',
                      plantStatus === 'active' ? 'border-border' : 'border-warning/50 opacity-75'
                    )}
                  >
                    <button
                      onClick={() => handleDeletePlant(plant.id)}
                      className="absolute top-2 right-2 p-1 rounded-full hover:bg-danger/20 text-text-muted hover:text-danger opacity-0 group-hover:opacity-100 transition-opacity"
                      title="삭제"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-xl">{PLANT_TYPE_LABELS[plant.type].icon}</span>
                      <span className="font-medium text-text-primary">{plant.name}</span>
                    </div>
                    <div className="space-y-1 text-sm">
                      <div className="flex justify-between">
                        <span className="text-text-muted">용량</span>
                        <span className="text-text-primary">{formatCapacity(plant.capacity)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">효율</span>
                        <span className="text-text-primary">
                          {(efficiency * 100).toFixed(0)}% ({yearsOld}년차)
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-muted">예상 발전</span>
                        <span className="text-text-primary">{dailyKwh.toFixed(1)} kWh/일</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-text-muted">상태</span>
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
                          <option value="active">{statusInfo.icon} 운영중</option>
                          <option value="maintenance">🔧 점검중</option>
                          <option value="paused">⏸ 중지</option>
                        </select>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Recommended Capacity */}
            <div className="bg-primary/10 border border-primary/30 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-muted">오늘의 추천 입찰량</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    날씨: {currentWeather === 'clear' ? '맑음 ☀️' :
                           currentWeather === 'partly_cloudy' ? '구름많음 ⛅' :
                           currentWeather === 'cloudy' ? '흐림 ☁️' : '비 🌧️'}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-2xl font-bold text-primary">
                    {getRecommendedCapacity().toFixed(1)} kWh
                  </p>
                  <p className="text-xs text-text-muted">
                    {powerPlants.length}개 발전소 합산
                  </p>
                </div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Conditional UI based on capacity */}
      {(() => {
        const totalPlantCapacityKw = powerPlants.reduce((sum, p) => sum + p.capacity, 0);
        const isLargeCapacity = totalPlantCapacityKw >= 1000; // 1MW = 1000kW
        const recommendedCapacity = getRecommendedCapacity();

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

      {/* Main Content - Only show for large capacity users */}
      {powerPlants.reduce((sum, p) => sum + p.capacity, 0) >= 1000 && (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bid Matrix */}
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">10-Segment 입찰 매트릭스</h3>
          <div className="space-y-2">
            <div className="grid grid-cols-4 gap-2 text-xs text-text-muted font-medium px-2">
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
                <span className="text-right text-sm text-text-muted font-mono">
                  {(() => {
                    // 수익 계산: MW × 1000(kW) × 가격(원/kWh)
                    const revenue = seg.quantity * 1000 * seg.price;
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
      {showRegistration && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-card border border-border rounded-xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-border">
              <h2 className="text-lg font-semibold text-text-primary">내 발전소 등록</h2>
              <button
                onClick={() => setShowRegistration(false)}
                className="p-2 rounded-full hover:bg-background text-text-muted"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-4 space-y-4">
              {/* Plant Name */}
              <div>
                <label className="block text-sm text-text-muted mb-1">발전소 이름</label>
                <input
                  type="text"
                  value={plantName}
                  onChange={(e) => setPlantName(e.target.value)}
                  placeholder="예: 우리집 태양광 1호"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Plant Type */}
              <div>
                <label className="block text-sm text-text-muted mb-2">설비 유형</label>
                <div className="grid grid-cols-3 gap-2">
                  {(Object.keys(PLANT_TYPE_LABELS) as PlantType[]).map((type) => (
                    <button
                      key={type}
                      onClick={() => setPlantType(type)}
                      className={clsx(
                        'flex flex-col items-center gap-1 p-3 rounded-lg border transition-colors',
                        plantType === type
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      )}
                    >
                      <span className="text-xl">{PLANT_TYPE_LABELS[type].icon}</span>
                      <span className="text-sm text-text-primary">{PLANT_TYPE_LABELS[type].label}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Capacity */}
              <div>
                <label className="block text-sm text-text-muted mb-1">설비 용량 (kW)</label>
                <input
                  type="number"
                  value={plantCapacity}
                  onChange={(e) => setPlantCapacity(e.target.value)}
                  min="0.1"
                  step="0.1"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Install Date */}
              <div>
                <label className="block text-sm text-text-muted mb-1">설치일</label>
                <div className="grid grid-cols-2 gap-2">
                  <select
                    value={installYear}
                    onChange={(e) => setInstallYear(Number(e.target.value))}
                    className="bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                  >
                    {Array.from({ length: 30 }, (_, i) => new Date().getFullYear() - i).map((year) => (
                      <option key={year} value={year}>{year}년</option>
                    ))}
                  </select>
                  <select
                    value={installMonth}
                    onChange={(e) => setInstallMonth(Number(e.target.value))}
                    className="bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                  >
                    {Array.from({ length: 12 }, (_, i) => i + 1).map((month) => (
                      <option key={month} value={month}>{month}월</option>
                    ))}
                  </select>
                </div>
                {/* Efficiency Preview */}
                {(() => {
                  const installDate = `${installYear}-${String(installMonth).padStart(2, '0')}-01`;
                  const efficiency = calculateEfficiency(installDate);
                  const effStatus = getEfficiencyStatus(efficiency);
                  return (
                    <div className="mt-2 p-2 bg-background rounded-lg flex justify-between items-center">
                      <span className="text-sm text-text-muted">현재 효율</span>
                      <span className="font-medium" style={{ color: effStatus.color }}>
                        {(efficiency * 100).toFixed(0)}% ({effStatus.status})
                      </span>
                    </div>
                  );
                })()}
              </div>

              {/* Contract Type */}
              <div>
                <label className="block text-sm text-text-muted mb-2">계약 유형</label>
                <div className="grid grid-cols-2 gap-2">
                  {(Object.keys(CONTRACT_TYPE_LABELS) as ContractType[]).map((type) => (
                    <button
                      key={type}
                      onClick={() => setContractType(type)}
                      className={clsx(
                        'p-3 rounded-lg border text-left transition-colors',
                        contractType === type
                          ? 'border-primary bg-primary/10'
                          : 'border-border hover:border-primary/50'
                      )}
                    >
                      <div className="font-medium text-text-primary">{CONTRACT_TYPE_LABELS[type].label}</div>
                      <div className="text-xs text-text-muted">{CONTRACT_TYPE_LABELS[type].description}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Roof Direction (for solar) */}
              {plantType === 'solar' && (
                <div>
                  <label className="block text-sm text-text-muted mb-2">지붕 방향</label>
                  <div className="grid grid-cols-4 gap-2">
                    {(Object.keys(ROOF_DIRECTION_LABELS) as RoofDirection[]).map((dir) => (
                      <button
                        key={dir}
                        onClick={() => setRoofDirection(dir)}
                        className={clsx(
                          'p-2 rounded-lg border text-sm transition-colors',
                          roofDirection === dir
                            ? 'border-primary bg-primary/10 text-text-primary'
                            : 'border-border text-text-muted hover:border-primary/50'
                        )}
                      >
                        {ROOF_DIRECTION_LABELS[dir]}
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
                  value={plantAddress}
                  onChange={(e) => setPlantAddress(e.target.value)}
                  placeholder="예: 제주시 구좌읍"
                  className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary"
                />
              </div>

              {/* Generation Estimate */}
              {(() => {
                const installDate = `${installYear}-${String(installMonth).padStart(2, '0')}-01`;
                const efficiency = calculateEfficiency(installDate);
                const cap = parseFloat(plantCapacity) || 0;
                const dailyKwh = estimateDailyGeneration(cap, efficiency, 'clear', roofDirection);
                const monthlyKwh = dailyKwh * 30;
                const smp = smpForHour.q50 || 100;
                const monthlyRevenue = monthlyKwh * smp;

                return (
                  <div className="bg-success/10 border border-success/30 rounded-lg p-4 space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-text-muted">예상 일일 발전량</span>
                      <span className="font-medium text-success">{dailyKwh.toFixed(1)} kWh</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-text-muted">예상 월간 수익</span>
                      <span className="font-medium text-success">
                        약 {formatRevenue(monthlyRevenue)}
                      </span>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Modal Footer */}
            <div className="flex gap-3 p-4 border-t border-border">
              <button
                onClick={() => setShowRegistration(false)}
                className="flex-1 btn-secondary"
              >
                취소
              </button>
              <button
                onClick={handleRegisterPlant}
                disabled={!plantName.trim() || !plantCapacity || isSubmitting}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
                  !plantName.trim() || !plantCapacity || isSubmitting
                    ? 'bg-background text-text-muted cursor-not-allowed'
                    : 'bg-primary text-white hover:bg-primary/90'
                )}
              >
                {isSubmitting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    등록 중...
                  </>
                ) : (
                  '등록'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
