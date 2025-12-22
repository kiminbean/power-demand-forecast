/**
 * Analysis Page - RE-BMS v6.0
 * XAI and Model Performance Analysis
 */

import { useState } from 'react';
import {
  Brain,
  TrendingUp,
  BarChart3,
  Info,
  Download,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  Legend,
} from 'recharts';
import { useModelInfo, useSMPForecast } from '../hooks/useApi';
import { useTheme } from '../contexts/ThemeContext';
import clsx from 'clsx';

// Feature importance data (XAI)
const featureImportance = [
  { feature: '이전 시간 SMP', importance: 0.32, category: 'temporal' },
  { feature: '기온', importance: 0.18, category: 'weather' },
  { feature: '태양광 발전량', importance: 0.15, category: 'generation' },
  { feature: '전력 수요', importance: 0.12, category: 'demand' },
  { feature: '풍속', importance: 0.09, category: 'weather' },
  { feature: '시간대', importance: 0.08, category: 'temporal' },
  { feature: '요일', importance: 0.04, category: 'temporal' },
  { feature: '습도', importance: 0.02, category: 'weather' },
];

// Model performance metrics
const performanceMetrics = [
  { metric: 'MAPE', value: 4.23, unit: '%', target: 5, status: 'good' },
  { metric: 'RMSE', value: 8.56, unit: '원', target: 10, status: 'good' },
  { metric: 'MAE', value: 6.12, unit: '원', target: 8, status: 'good' },
  { metric: 'R²', value: 0.92, unit: '', target: 0.9, status: 'good' },
  { metric: '신뢰구간 커버리지', value: 94.5, unit: '%', target: 90, status: 'good' },
];

// Daily performance trend
const performanceTrend = Array.from({ length: 14 }, (_, i) => ({
  day: `${12 - 13 + i}일`,
  mape: 3.5 + Math.random() * 2,
  coverage: 92 + Math.random() * 6,
}));

// Model radar data
const radarData = [
  { subject: '정확도', A: 92, fullMark: 100 },
  { subject: '안정성', A: 88, fullMark: 100 },
  { subject: '반응성', A: 85, fullMark: 100 },
  { subject: '일반화', A: 90, fullMark: 100 },
  { subject: '신뢰구간', A: 95, fullMark: 100 },
  { subject: '극단값 처리', A: 78, fullMark: 100 },
];

export default function Analysis() {
  const { data: modelInfo } = useModelInfo();
  useSMPForecast(); // Load forecast data
  const [selectedTab, setSelectedTab] = useState<'xai' | 'performance' | 'comparison'>('xai');
  const { isDark } = useTheme();

  // Theme-aware chart colors
  const chartColors = {
    grid: isDark ? '#374151' : '#e5e7eb',
    axis: isDark ? '#9ca3af' : '#6b7280',
    tooltipBg: isDark ? '#1e2530' : '#ffffff',
    tooltipBorder: isDark ? '#374151' : '#e5e7eb',
  };

  const categoryColors: Record<string, string> = {
    temporal: '#6366f1',
    weather: '#06b6d4',
    generation: '#fbbf24',
    demand: '#22c55e',
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">분석</h1>
          <p className="text-text-muted mt-1">모델 성능 및 예측 근거 분석</p>
        </div>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {modelInfo && (
            <div className="flex items-center gap-2 px-3 py-2 bg-card rounded-lg border border-border">
              <Brain className="w-4 h-4 text-primary flex-shrink-0" />
              <span className="text-sm text-text-primary whitespace-nowrap">{modelInfo.type} {modelInfo.version}</span>
              <span className="px-2 py-0.5 text-xs bg-success/20 text-success rounded whitespace-nowrap">Active</span>
            </div>
          )}
          <button className="btn-secondary flex items-center gap-2 whitespace-nowrap">
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">리포트 다운로드</span>
            <span className="sm:hidden">다운로드</span>
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-border pb-2">
        {[
          { key: 'xai', label: 'XAI 분석', icon: Brain },
          { key: 'performance', label: '모델 성능', icon: BarChart3 },
          { key: 'comparison', label: '예측 비교', icon: TrendingUp },
        ].map(({ key, label, icon: Icon }) => (
          <button
            key={key}
            onClick={() => setSelectedTab(key as any)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
              selectedTab === key
                ? 'bg-primary text-text-primary'
                : 'text-text-muted hover:text-text-primary hover:bg-background'
            )}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* XAI Tab */}
      {selectedTab === 'xai' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Feature Importance */}
          <div className="card">
            <h3 className="text-lg font-semibold text-text-primary mb-4">피처 중요도</h3>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={featureImportance}
                  layout="vertical"
                  margin={{ left: 100, right: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} horizontal={false} />
                  <XAxis
                    type="number"
                    stroke={chartColors.axis}
                    fontSize={12}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    domain={[0, 0.35]}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    stroke={chartColors.axis}
                    fontSize={12}
                    width={100}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltipBg,
                      border: `1px solid ${chartColors.tooltipBorder}`,
                      borderRadius: '8px',
                    }}
                    formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, '중요도']}
                  />
                  <Bar
                    dataKey="importance"
                    radius={[0, 4, 4, 0]}
                    fill="#6366f1"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-4 flex flex-wrap gap-2">
              {Object.entries(categoryColors).map(([cat, color]) => (
                <div key={cat} className="flex items-center gap-2 px-2 py-1 bg-background rounded">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: color }} />
                  <span className="text-xs text-text-muted">
                    {cat === 'temporal' && '시간적'}
                    {cat === 'weather' && '기상'}
                    {cat === 'generation' && '발전'}
                    {cat === 'demand' && '수요'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Model Radar */}
          <div className="card">
            <h3 className="text-lg font-semibold text-text-primary mb-4">모델 특성 분석</h3>
            <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                  <PolarGrid stroke={chartColors.grid} />
                  <PolarAngleAxis dataKey="subject" tick={{ fill: chartColors.axis, fontSize: 12 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: chartColors.axis, fontSize: 10 }} />
                  <Radar
                    name="모델 성능"
                    dataKey="A"
                    stroke="#6366f1"
                    fill="#6366f1"
                    fillOpacity={0.3}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Explanation Cards */}
          <div className="card col-span-2">
            <h3 className="text-lg font-semibold text-text-primary mb-4">예측 근거 설명</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="p-4 bg-background rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-4 h-4 text-primary" />
                  </div>
                  <span className="font-medium text-text-primary">시계열 패턴</span>
                </div>
                <p className="text-sm text-text-muted">
                  이전 시간대의 SMP 값이 가장 강한 예측 인자로, 자기상관 패턴을 학습하여
                  단기 예측에 높은 정확도를 보입니다.
                </p>
              </div>

              <div className="p-4 bg-background rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-8 bg-wind/10 rounded-lg flex items-center justify-center">
                    <Info className="w-4 h-4 text-wind" />
                  </div>
                  <span className="font-medium text-text-primary">기상 영향</span>
                </div>
                <p className="text-sm text-text-muted">
                  기온과 풍속이 SMP에 유의미한 영향을 미치며, 특히 태양광/풍력 발전량
                  변화를 통해 간접적으로 가격에 반영됩니다.
                </p>
              </div>

              <div className="p-4 bg-background rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-8 h-8 bg-success/10 rounded-lg flex items-center justify-center">
                    <BarChart3 className="w-4 h-4 text-success" />
                  </div>
                  <span className="font-medium text-text-primary">수급 균형</span>
                </div>
                <p className="text-sm text-text-muted">
                  재생에너지 발전량과 전력 수요의 균형이 SMP 결정에 핵심 역할을 하며,
                  피크 시간대에 특히 중요합니다.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Performance Tab */}
      {selectedTab === 'performance' && (
        <div className="space-y-6">
          {/* Metrics Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
            {performanceMetrics.map((m) => (
              <div key={m.metric} className="card">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-muted">{m.metric}</span>
                  <span className={clsx(
                    'px-2 py-0.5 text-xs rounded',
                    m.status === 'good' ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'
                  )}>
                    목표 달성
                  </span>
                </div>
                <div className="text-2xl font-bold text-text-primary">
                  {m.value}
                  <span className="text-sm text-text-muted ml-1">{m.unit}</span>
                </div>
                <div className="text-xs text-text-muted mt-1">
                  목표: {m.target}{m.unit}
                </div>
              </div>
            ))}
          </div>

          {/* Performance Trend */}
          <div className="card">
            <h3 className="text-lg font-semibold text-text-primary mb-4">일별 성능 추이</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={performanceTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
                  <XAxis dataKey="day" stroke={chartColors.axis} fontSize={12} />
                  <YAxis yAxisId="left" stroke={chartColors.axis} fontSize={12} domain={[0, 10]} />
                  <YAxis yAxisId="right" orientation="right" stroke={chartColors.axis} fontSize={12} domain={[80, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: chartColors.tooltipBg,
                      border: `1px solid ${chartColors.tooltipBorder}`,
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="mape"
                    stroke="#6366f1"
                    strokeWidth={2}
                    name="MAPE (%)"
                    dot={{ fill: '#6366f1' }}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="coverage"
                    stroke="#22c55e"
                    strokeWidth={2}
                    name="커버리지 (%)"
                    dot={{ fill: '#22c55e' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Tab */}
      {selectedTab === 'comparison' && (
        <div className="card">
          <h3 className="text-lg font-semibold text-text-primary mb-4">예측 vs 실제 비교</h3>
          <p className="text-text-muted mb-4">
            최근 예측 결과와 실제 SMP 값을 비교하여 모델 성능을 검증합니다.
          </p>
          <div className="h-[400px] flex items-center justify-center text-text-muted">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>실시간 데이터 연동 후 비교 분석이 표시됩니다.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
