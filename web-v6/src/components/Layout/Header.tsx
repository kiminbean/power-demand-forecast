/**
 * Header Component - RE-BMS v6.0
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
    <header className="h-16 bg-secondary border-b border-border px-6 flex items-center justify-between transition-colors duration-200">
      {/* Logo & Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <img
            src={isDark ? '/logo-light.png' : '/logo-dark.png'}
            alt="eXeco"
            className="h-8 w-auto"
          />
          <div className="border-l border-border pl-3">
            <h1 className="text-sm font-bold text-brand">RE-BMS</h1>
            <p className="text-xs text-text-muted">v6.0</p>
          </div>
        </div>
      </div>

      {/* Center - Time Display */}
      <div className="flex items-center gap-6">
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-text-primary tabular-nums">
            {formatTime(currentTime)}
          </div>
          <div className="text-xs text-text-muted">{formatDate(currentTime)}</div>
        </div>
      </div>

      {/* Right - Status & Actions */}
      <div className="flex items-center gap-4">
        {/* API Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
          <div className={`status-dot ${apiStatus === true ? 'status-success' : apiStatus === false ? 'status-danger' : 'status-warning'}`} />
          <span className="text-sm text-text-muted">
            {apiStatus === true ? 'API Connected' : apiStatus === false ? 'Offline' : 'Checking...'}
          </span>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
            <Activity className="w-4 h-4 text-primary" />
            <span className="text-sm text-text-muted">
              {modelInfo.version} | MAPE: {modelInfo.mape}%
            </span>
          </div>
        )}

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 hover:bg-background rounded-lg transition-colors"
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {isDark ? (
            <Sun className="w-5 h-5 text-text-muted hover:text-warning" />
          ) : (
            <Moon className="w-5 h-5 text-text-muted hover:text-primary" />
          )}
        </button>

        {/* Notifications */}
        <button className="relative p-2 hover:bg-background rounded-lg transition-colors">
          <Bell className="w-5 h-5 text-text-muted" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-danger rounded-full" />
        </button>

        {/* Settings */}
        <button className="p-2 hover:bg-background rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-text-muted" />
        </button>

        {/* User */}
        <button className="flex items-center gap-2 p-2 hover:bg-background rounded-lg transition-colors">
          <div className="w-8 h-8 bg-primary/20 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-primary" />
          </div>
        </button>
      </div>
    </header>
  );
}
