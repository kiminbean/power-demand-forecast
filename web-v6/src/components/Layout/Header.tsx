/**
 * Header Component - RE-BMS v6.0
 */

import { useState, useEffect } from 'react';
import { Bell, Settings, User, Zap, Activity } from 'lucide-react';
import { useApiStatus, useModelInfo } from '../../hooks/useApi';

export default function Header() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const apiStatus = useApiStatus();
  const { data: modelInfo } = useModelInfo();

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
    <header className="h-16 bg-secondary border-b border-border px-6 flex items-center justify-between">
      {/* Logo & Title */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">RE-BMS</h1>
            <p className="text-xs text-gray-400">v6.0 Desktop</p>
          </div>
        </div>
      </div>

      {/* Center - Time Display */}
      <div className="flex items-center gap-6">
        <div className="text-center">
          <div className="text-2xl font-mono font-bold text-white tabular-nums">
            {formatTime(currentTime)}
          </div>
          <div className="text-xs text-gray-400">{formatDate(currentTime)}</div>
        </div>
      </div>

      {/* Right - Status & Actions */}
      <div className="flex items-center gap-4">
        {/* API Status */}
        <div className="flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
          <div className={`status-dot ${apiStatus === true ? 'status-success' : apiStatus === false ? 'status-danger' : 'status-warning'}`} />
          <span className="text-sm text-gray-400">
            {apiStatus === true ? 'API Connected' : apiStatus === false ? 'Offline' : 'Checking...'}
          </span>
        </div>

        {/* Model Info */}
        {modelInfo && (
          <div className="flex items-center gap-2 px-3 py-1.5 bg-background rounded-lg">
            <Activity className="w-4 h-4 text-primary" />
            <span className="text-sm text-gray-400">
              {modelInfo.version} | MAPE: {modelInfo.mape}%
            </span>
          </div>
        )}

        {/* Notifications */}
        <button className="relative p-2 hover:bg-background rounded-lg transition-colors">
          <Bell className="w-5 h-5 text-gray-400" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-danger rounded-full" />
        </button>

        {/* Settings */}
        <button className="p-2 hover:bg-background rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-gray-400" />
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
