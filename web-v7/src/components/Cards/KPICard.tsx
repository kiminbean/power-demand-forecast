/**
 * KPI Card Component - RE-BMS v6.0
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

const colorClasses = {
  primary: 'text-primary bg-primary/10',
  success: 'text-success bg-success/10',
  warning: 'text-warning bg-warning/10',
  danger: 'text-danger bg-danger/10',
  smp: 'text-smp bg-smp/10',
  solar: 'text-solar bg-solar/10',
  wind: 'text-wind bg-wind/10',
};

const valueColorClasses = {
  primary: 'text-primary',
  success: 'text-success',
  warning: 'text-warning',
  danger: 'text-danger',
  smp: 'text-smp',
  solar: 'text-solar',
  wind: 'text-wind',
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
              isPositive && 'text-success',
              isNegative && 'text-danger',
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
