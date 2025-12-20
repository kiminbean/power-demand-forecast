/**
 * Mobile Layout with Bottom Navigation
 */

import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { TrendingUp, Gavel, Receipt, Sun, Moon } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { useApiStatus } from '../hooks/useApi';
import clsx from 'clsx';

const navItems = [
  { path: '/smp', label: 'SMP 예측', icon: TrendingUp },
  { path: '/bidding', label: '입찰', icon: Gavel },
  { path: '/settlement', label: '정산', icon: Receipt },
];

export default function MobileLayout() {
  const { isDark, toggleTheme } = useTheme();
  const apiStatus = useApiStatus();
  const location = useLocation();

  // Get current page title
  const currentPage = navItems.find(item => item.path === location.pathname);
  const pageTitle = currentPage?.label || 'RE-BMS';

  return (
    <div className="min-h-screen min-h-[100dvh] flex flex-col bg-background">
      {/* Header */}
      <header className="safe-area-top bg-secondary border-b border-border sticky top-0 z-50">
        <div className="flex items-center justify-between px-4 h-14">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <img
              src={isDark ? '/logo-light.png' : '/logo-dark.png'}
              alt="eXeco"
              className="h-6 w-auto"
            />
            <span className="text-sm font-bold text-brand">|</span>
            <span className="text-sm font-semibold text-text-primary">{pageTitle}</span>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-3">
            {/* API Status */}
            <div className="flex items-center gap-1.5">
              <div className={clsx(
                'status-dot',
                apiStatus === true ? 'status-success' : apiStatus === false ? 'status-danger' : 'status-warning'
              )} />
              <span className="text-xs text-text-muted">
                {apiStatus === true ? 'Live' : apiStatus === false ? 'Off' : '...'}
              </span>
            </div>

            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg active:bg-background transition-colors"
            >
              {isDark ? (
                <Sun className="w-5 h-5 text-text-muted" />
              ) : (
                <Moon className="w-5 h-5 text-text-muted" />
              )}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto pb-20">
        <div className="p-4 animate-fade-in">
          <Outlet />
        </div>
      </main>

      {/* Bottom Navigation */}
      <nav className="safe-area-bottom fixed bottom-0 left-0 right-0 bg-secondary border-t border-border z-50">
        <div className="flex items-center justify-around h-16">
          {navItems.map(({ path, label, icon: Icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) => clsx('nav-item flex-1', isActive && 'active')}
            >
              <Icon className="w-6 h-6" />
              <span className="text-xs font-medium">{label}</span>
            </NavLink>
          ))}
        </div>
      </nav>
    </div>
  );
}
