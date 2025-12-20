/**
 * Sidebar Navigation - RE-BMS v6.0
 */

import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  Gavel,
  FolderKanban,
  Receipt,
  Map,
  BarChart3,
  HelpCircle,
} from 'lucide-react';
import clsx from 'clsx';

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactNode;
  badge?: string;
}

const navItems: NavItem[] = [
  { path: '/', label: '대시보드', icon: <LayoutDashboard className="w-5 h-5" /> },
  { path: '/smp', label: 'SMP 예측', icon: <TrendingUp className="w-5 h-5" />, badge: 'AI' },
  { path: '/bidding', label: '입찰 관리', icon: <Gavel className="w-5 h-5" /> },
  { path: '/portfolio', label: '포트폴리오', icon: <FolderKanban className="w-5 h-5" /> },
  { path: '/settlement', label: '정산', icon: <Receipt className="w-5 h-5" /> },
  { path: '/map', label: '제주 지도', icon: <Map className="w-5 h-5" /> },
  { path: '/analysis', label: '분석', icon: <BarChart3 className="w-5 h-5" /> },
];

export default function Sidebar() {
  return (
    <aside className="w-64 bg-secondary border-r border-border flex flex-col transition-colors duration-200">
      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        <div className="text-xs font-semibold text-text-muted uppercase tracking-wider px-4 py-2">
          메인
        </div>
        {navItems.slice(0, 5).map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx('nav-link', isActive && 'active')
            }
          >
            {item.icon}
            <span className="flex-1">{item.label}</span>
            {item.badge && (
              <span className="px-1.5 py-0.5 text-[10px] font-bold bg-primary/20 text-primary rounded">
                {item.badge}
              </span>
            )}
          </NavLink>
        ))}

        <div className="text-xs font-semibold text-text-muted uppercase tracking-wider px-4 py-2 mt-4">
          분석
        </div>
        {navItems.slice(5).map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx('nav-link', isActive && 'active')
            }
          >
            {item.icon}
            <span className="flex-1">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-border">
        <div className="card p-3 bg-background">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
              <HelpCircle className="w-5 h-5 text-primary" />
            </div>
            <div>
              <div className="text-sm font-medium text-text-primary">도움이 필요하신가요?</div>
              <div className="text-xs text-text-muted">문서 및 지원</div>
            </div>
          </div>
          <button className="w-full py-2 text-sm text-primary hover:bg-primary/10 rounded-lg transition-colors">
            가이드 보기
          </button>
        </div>

        <div className="mt-4 text-center">
          <p className="text-xs text-text-muted">RE-BMS v6.0.0</p>
          <p className="text-xs text-text-muted">© 2025 Power Demand Forecast</p>
        </div>
      </div>
    </aside>
  );
}
