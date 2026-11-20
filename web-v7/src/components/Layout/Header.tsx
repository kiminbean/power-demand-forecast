/**
 * Header Component - eXeco DS v1
 */

import { useState, useEffect } from 'react';
import { Bell, Settings, User, Activity, Sun, Moon } from 'lucide-react';
import { useApiStatus, useModelInfo } from '../../hooks/useApi';
import { useTheme } from '../../contexts/ThemeContext';

export default function Header() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const apiStatus = useApiStatus();
  const { data: modelInfo } = useModelInfo();
  const { toggleTheme, isDark } = useTheme();

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('ko-KR', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'short',
    });
  };

  return (
    // eXeco DS v1: 헤더 그라데이션 #04265E → #0A3A7D, 항상 밝은 로고 사용
    <header
      className="h-16 px-6 flex items-center justify-between border-b"
      style={{
        background: 'linear-gradient(90deg, #04265E 0%, #0A3A7D 100%)',
        borderBottomColor: 'rgba(148, 163, 184, 0.16)',
      }}
    >
      {/* Logo & Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <img
            src="/logo-light.png"
            alt="eXeco"
            className="h-8 w-auto"
          />
          <div className="border-l border-white/25 pl-3">
            <h1 className="text-sm font-bold text-white">eXeco</h1>
            <p className="text-xs text-white/80">v7.0</p>
          </div>
        </div>
      </div>

      {/* Center - Time Display */}
      <div className="flex items-center gap-6">
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-white tabular-nums">
            {formatTime(currentTime)}
          </div>
          <div className="text-xs text-white/85">{formatDate(currentTime)}</div>
        </div>
      </div>

      {/* Right - Status & Actions */}
      <div className="flex items-center gap-4">
        {/* API Status */}
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg border"
          style={{
            background: 'rgba(255, 255, 255, 0.10)',
            borderColor: 'rgba(148, 163, 184, 0.24)',
          }}
        >
          <div className={`status-dot ${apiStatus === true ? 'status-success' : apiStatus === false ? 'status-danger' : 'status-warning'}`} />
          <span className="text-sm font-medium text-white/90">
            {apiStatus === true ? 'API Connected' : apiStatus === false ? 'Offline' : 'Checking...'}
          </span>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border"
            style={{
              background: 'rgba(255, 255, 255, 0.10)',
              borderColor: 'rgba(148, 163, 184, 0.24)',
            }}
          >
            <Activity className="w-4 h-4 text-accent" />
            <span className="text-sm font-medium text-white/90">
              {modelInfo.version} | {typeof modelInfo.mape === 'number' ? `MAPE: ${modelInfo.mape}%` : 'MAPE 재검증 전'}
            </span>
          </div>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {isDark ? (
            <Sun className="w-5 h-5 text-white/85 hover:text-warning" />
          ) : (
            <Moon className="w-5 h-5 text-white/85 hover:text-accent" />
          )}
        </button>

        {/* Notifications */}
        <button className="relative p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Bell className="w-5 h-5 text-white/85" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-danger rounded-full" />
        </button>

        {/* Settings */}
        <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-white/85" />
        </button>

        {/* User */}
        <button className="flex items-center gap-2 p-2 hover:bg-white/10 rounded-lg transition-colors">
          <div className="w-8 h-8 bg-white/15 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
        </button>
      </div>
    </header>
  );
}
