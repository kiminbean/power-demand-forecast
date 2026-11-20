/**
 * KPI Card Component - eXeco DS v1
 */

import { LucideIcon } from 'lucide-react';
import clsx from 'clsx';

interface KPICardProps {
  title: string;
  value: string | number;
  unit?: string;
  change?: number;
  changeLabel?: string;
  icon: LucideIcon;
  color?: 'primary' | 'success' | 'warning' | 'danger' | 'smp' | 'solar' | 'wind';
  size?: 'sm' | 'md' | 'lg';
}

// 아이콘 칩 배경 + 값 텍스트는 다크/라이트 양쪽에서 4.5:1 이상 되도록 테마 변수 사용
const colorClasses = {
  primary: 'text-[color:var(--color-info-text)] bg-primary/10',
  success: 'text-[color:var(--color-success-text)] bg-success/10',
  warning: 'text-[color:var(--color-warning-text)] bg-warning/10',
  danger: 'text-[color:var(--color-danger-text)] bg-danger/10',
  smp: 'text-[color:var(--color-warning-text)] bg-smp/10',
  solar: 'text-[color:var(--color-warning-text)] bg-solar/10',
  wind: 'text-[color:var(--color-wind-text)] bg-wind/10',
};

const valueColorClasses = {
  primary: 'text-[color:var(--color-info-text)]',
  success: 'text-[color:var(--color-success-text)]',
  warning: 'text-[color:var(--color-warning-text)]',
  danger: 'text-[color:var(--color-danger-text)]',
  smp: 'text-[color:var(--color-warning-text)]',
  solar: 'text-[color:var(--color-warning-text)]',
  wind: 'text-[color:var(--color-wind-text)]',
};

export default function KPICard({
  title,
  value,
  unit,
  change,
  changeLabel,
  icon: Icon,
  color = 'primary',
  size = 'md',
}: KPICardProps) {
  const isPositive = change !== undefined && change > 0;
  const isNegative = change !== undefined && change < 0;

  return (
    <div className="card card-hover">
      <div className="flex items-start justify-between mb-4">
        <div className={clsx('p-2 rounded-lg', colorClasses[color])}>
          <Icon className={clsx('w-5 h-5')} />
        </div>
        {change !== undefined && (
          <div
            className={clsx(
              'flex items-center gap-1 text-sm font-medium',
              isPositive && 'text-[color:var(--color-success-text)]',
              isNegative && 'text-[color:var(--color-danger-text)]',
              !isPositive && !isNegative && 'text-text-muted'
            )}
          >
            <span>
              {isPositive && '↑'}
              {isNegative && '↓'}
              {!isPositive && !isNegative && '–'}
            </span>
            <span>{Math.abs(change).toFixed(1)}%</span>
          </div>
        )}
      </div>

      <div>
        <p className="text-sm text-text-muted mb-1">{title}</p>
        <div className="flex items-baseline gap-1">
          <span
            className={clsx(
              'font-bold tabular-nums',
              valueColorClasses[color],
              size === 'sm' && 'text-xl',
              size === 'md' && 'text-3xl',
              size === 'lg' && 'text-4xl'
            )}
          >
            {typeof value === 'number' ? value.toLocaleString() : value}
          </span>
          {unit && <span className="text-sm text-text-muted">{unit}</span>}
        </div>
        {changeLabel && (
          <p className="text-xs text-text-muted mt-1">{changeLabel}</p>
        )}
      </div>
    </div>
  );
}
