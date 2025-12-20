/**
 * Portfolio Page - RE-BMS v6.0
 * Jeju Power Plant Portfolio Management
 */

import { useState } from 'react';
import {
  Sun,
  Wind,
  Battery,
  MapPin,
  Activity,
  Zap,
  MoreVertical,
  Search,
} from 'lucide-react';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { useResources } from '../hooks/useApi';
import clsx from 'clsx';

export default function Portfolio() {
  const { data: resources } = useResources();
  const [filter, setFilter] = useState<'all' | 'solar' | 'wind'>('all');
  const [searchTerm, setSearchTerm] = useState('');

  // Filter resources
  const filteredResources = resources?.filter((r) => {
    const matchesFilter = filter === 'all' || r.type === filter;
    const matchesSearch = r.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          r.location.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  }) ?? [];

  // Calculate stats
  const stats = resources ? {
    totalCapacity: resources.reduce((sum, r) => sum + r.capacity, 0),
    currentOutput: resources.reduce((sum, r) => sum + r.current_output, 0),
    solarCapacity: resources.filter(r => r.type === 'solar').reduce((sum, r) => sum + r.capacity, 0),
    windCapacity: resources.filter(r => r.type === 'wind').reduce((sum, r) => sum + r.capacity, 0),
    solarOutput: resources.filter(r => r.type === 'solar').reduce((sum, r) => sum + r.current_output, 0),
    windOutput: resources.filter(r => r.type === 'wind').reduce((sum, r) => sum + r.current_output, 0),
    activeCount: resources.filter(r => r.status === 'active').length,
  } : null;

  // Pie chart data
  const capacityByType = [
    { name: '풍력', value: stats?.windCapacity ?? 0, color: '#06b6d4' },
    { name: '태양광', value: stats?.solarCapacity ?? 0, color: '#fbbf24' },
  ];

  // Bar chart data - top 10 by output
  const topResources = [...(resources ?? [])]
    .sort((a, b) => b.current_output - a.current_output)
    .slice(0, 10)
    .map((r) => ({
      name: r.name.replace(/풍력|태양광/g, ''),
      output: r.current_output,
      capacity: r.capacity,
      utilization: r.utilization,
      type: r.type,
    }));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">포트폴리오</h1>
          <p className="text-gray-400 mt-1">제주 재생에너지 발전소 현황</p>
        </div>
        <div className="flex items-center gap-3">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="발전소 검색..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 bg-card border border-border rounded-lg text-white text-sm w-64"
            />
          </div>

          {/* Filter */}
          <div className="flex items-center bg-card rounded-lg border border-border p-1">
            {(['all', 'solar', 'wind'] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={clsx(
                  'px-3 py-1.5 text-sm rounded-md transition-colors',
                  filter === f
                    ? 'bg-primary text-white'
                    : 'text-gray-400 hover:text-white'
                )}
              >
                {f === 'all' && '전체'}
                {f === 'solar' && '태양광'}
                {f === 'wind' && '풍력'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-5 gap-4">
        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-primary" />
            <span className="text-sm text-gray-400">총 설비용량</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {stats?.totalCapacity.toFixed(1) ?? '-'} <span className="text-sm text-gray-400">MW</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-success" />
            <span className="text-sm text-gray-400">현재 발전량</span>
          </div>
          <div className="text-2xl font-bold text-success">
            {stats?.currentOutput.toFixed(1) ?? '-'} <span className="text-sm text-gray-400">MW</span>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Wind className="w-4 h-4 text-wind" />
            <span className="text-sm text-gray-400">풍력 발전</span>
          </div>
          <div className="text-2xl font-bold text-wind">
            {stats?.windOutput.toFixed(1) ?? '-'} <span className="text-sm text-gray-400">MW</span>
          </div>
          <div className="text-xs text-gray-500">{stats?.windCapacity.toFixed(1)} MW 용량</div>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Sun className="w-4 h-4 text-solar" />
            <span className="text-sm text-gray-400">태양광 발전</span>
          </div>
          <div className="text-2xl font-bold text-solar">
            {stats?.solarOutput.toFixed(1) ?? '-'} <span className="text-sm text-gray-400">MW</span>
          </div>
          <div className="text-xs text-gray-500">{stats?.solarCapacity.toFixed(1)} MW 용량</div>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Battery className="w-4 h-4 text-primary" />
            <span className="text-sm text-gray-400">가동률</span>
          </div>
          <div className="text-2xl font-bold text-primary">
            {stats ? ((stats.currentOutput / stats.totalCapacity) * 100).toFixed(1) : '-'}
            <span className="text-sm text-gray-400">%</span>
          </div>
          <div className="text-xs text-gray-500">{stats?.activeCount ?? '-'}개 운영중</div>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-6">
        {/* Capacity Distribution Pie */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">설비용량 구성</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={capacityByType}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false}
                >
                  {capacityByType.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e2530',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)} MW`, '']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Producers Bar Chart */}
        <div className="card col-span-2">
          <h3 className="text-lg font-semibold text-white mb-4">발전량 Top 10</h3>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topResources} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
                <XAxis type="number" stroke="#9ca3af" fontSize={12} />
                <YAxis
                  type="category"
                  dataKey="name"
                  stroke="#9ca3af"
                  fontSize={11}
                  width={80}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e2530',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                  }}
                  formatter={(value: number, name: string) => [
                    `${value.toFixed(1)} MW`,
                    name === 'output' ? '발전량' : '용량',
                  ]}
                />
                <Bar
                  dataKey="output"
                  fill="#22c55e"
                  radius={[0, 4, 4, 0]}
                  name="발전량"
                />
                <Bar
                  dataKey="capacity"
                  fill="#6366f1"
                  radius={[0, 4, 4, 0]}
                  opacity={0.3}
                  name="용량"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Resource Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">발전소 목록</h3>
          <span className="text-sm text-gray-400">{filteredResources.length}개 발전소</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-border">
                <th className="text-left py-3 px-4">발전소명</th>
                <th className="text-left py-3 px-4">유형</th>
                <th className="text-left py-3 px-4">위치</th>
                <th className="text-right py-3 px-4">설비용량</th>
                <th className="text-right py-3 px-4">현재발전</th>
                <th className="text-right py-3 px-4">가동률</th>
                <th className="text-center py-3 px-4">상태</th>
                <th className="text-center py-3 px-4"></th>
              </tr>
            </thead>
            <tbody>
              {filteredResources.map((resource) => (
                <tr
                  key={resource.id}
                  className="border-b border-border/50 hover:bg-background/50 transition-colors"
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div className={clsx(
                        'w-8 h-8 rounded-lg flex items-center justify-center',
                        resource.type === 'wind' ? 'bg-wind/10' : 'bg-solar/10'
                      )}>
                        {resource.type === 'wind' ? (
                          <Wind className="w-4 h-4 text-wind" />
                        ) : (
                          <Sun className="w-4 h-4 text-solar" />
                        )}
                      </div>
                      <span className="text-white font-medium">{resource.name}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <span className={clsx(
                      'px-2 py-1 text-xs rounded',
                      resource.type === 'wind'
                        ? 'bg-wind/20 text-wind'
                        : 'bg-solar/20 text-solar'
                    )}>
                      {resource.type === 'wind' ? '풍력' : '태양광'}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1 text-gray-400">
                      <MapPin className="w-3 h-3" />
                      {resource.location}
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-white">
                    {resource.capacity.toFixed(1)} MW
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-success">
                    {resource.current_output.toFixed(1)} MW
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="w-16 h-2 bg-background rounded-full overflow-hidden">
                        <div
                          className={clsx(
                            'h-full rounded-full',
                            resource.utilization >= 80 ? 'bg-success' :
                            resource.utilization >= 50 ? 'bg-primary' : 'bg-warning'
                          )}
                          style={{ width: `${resource.utilization}%` }}
                        />
                      </div>
                      <span className="font-mono text-white w-12 text-right">
                        {resource.utilization.toFixed(0)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <span className={clsx(
                      'px-2 py-1 text-xs rounded',
                      resource.status === 'active'
                        ? 'bg-success/20 text-success'
                        : 'bg-danger/20 text-danger'
                    )}>
                      {resource.status === 'active' ? '운영중' : '정지'}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-center">
                    <button className="p-1 hover:bg-background rounded transition-colors">
                      <MoreVertical className="w-4 h-4 text-gray-400" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
