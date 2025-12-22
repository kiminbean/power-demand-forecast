/**
 * Bid Review Modal - RE-BMS v6.1
 * Internal review and approval workflow before KPX submission
 */

import { useState, useEffect } from 'react';
import {
  X,
  CheckCircle,
  XCircle,
  AlertTriangle,
  User,
  Clock,
  Shield,
  ArrowRight,
  Loader2,
} from 'lucide-react';
import clsx from 'clsx';

interface BidSegment {
  id: number;
  quantity: number;
  price: number;
}

interface BidReviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApprove: () => void;
  onReject: () => void;
  segments: BidSegment[];
  selectedHour: number;
  smpForecast: { q10: number; q50: number; q90: number };
  capacity: number;
}

type ReviewStep = 'review' | 'approving' | 'approved' | 'rejected';

export default function BidReviewModal({
  isOpen,
  onClose,
  onApprove,
  onReject,
  segments,
  selectedHour,
  smpForecast,
  capacity,
}: BidReviewModalProps) {
  const [step, setStep] = useState<ReviewStep>('review');
  const [remarks, setRemarks] = useState('');
  const [checklist, setChecklist] = useState({
    priceValid: false,
    quantityValid: false,
    monotonic: false,
    riskAssessed: false,
  });

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setStep('review');
      setRemarks('');
      setChecklist({
        priceValid: false,
        quantityValid: false,
        monotonic: false,
        riskAssessed: false,
      });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const totalQuantity = segments.reduce((sum, s) => sum + s.quantity, 0);
  const avgPrice = segments.reduce((sum, s) => sum + s.price * s.quantity, 0) / totalQuantity;
  const expectedRevenue = (totalQuantity * avgPrice) / 1000;

  // Check monotonic constraint
  const isMonotonic = segments.every((seg, idx) =>
    idx === 0 || seg.price >= segments[idx - 1].price
  );

  // Validation checks
  const validations = [
    {
      key: 'priceValid',
      label: '가격 범위 적정성',
      passed: segments.every(s => s.price >= 50 && s.price <= 200),
      description: '모든 구간 가격이 50~200원/kWh 범위 내',
    },
    {
      key: 'quantityValid',
      label: '물량 제한 준수',
      passed: totalQuantity <= capacity,
      description: `총 물량 ${totalQuantity.toFixed(1)}MW ≤ 허용 용량 ${capacity}MW`,
    },
    {
      key: 'monotonic',
      label: '단조성 제약 충족',
      passed: isMonotonic,
      description: '입찰가격이 물량 증가에 따라 단조 증가',
    },
    {
      key: 'riskAssessed',
      label: 'SMP 예측 대비 적정성',
      passed: avgPrice <= smpForecast.q90 * 1.1,
      description: `평균 입찰가 ${avgPrice.toFixed(0)}원 vs SMP 상한 ${smpForecast.q90.toFixed(0)}원`,
    },
  ];

  const allChecked = Object.values(checklist).every(v => v);

  const handleApprove = () => {
    setStep('approving');
    // Simulate approval process
    setTimeout(() => {
      setStep('approved');
      setTimeout(() => {
        onApprove();
      }, 1500);
    }, 2000);
  };

  const handleReject = () => {
    setStep('rejected');
    setTimeout(() => {
      onReject();
    }, 1500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-3xl max-h-[90vh] overflow-auto bg-card border border-border rounded-2xl shadow-2xl m-4">
        {/* Header */}
        <div className="sticky top-0 bg-card border-b border-border px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-primary/20 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-text-primary">입찰 검토 및 승인</h2>
              <p className="text-sm text-text-muted">
                {String(selectedHour).padStart(2, '0')}:00 - {String(selectedHour + 1).padStart(2, '0')}:00 거래시간
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-background rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-text-muted" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {step === 'review' && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-background rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-text-primary">{totalQuantity.toFixed(1)}</div>
                  <div className="text-sm text-text-muted">총 물량 (MW)</div>
                </div>
                <div className="bg-background rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-smp">{avgPrice.toFixed(1)}</div>
                  <div className="text-sm text-text-muted">평균가격 (원/kWh)</div>
                </div>
                <div className="bg-background rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-success">{expectedRevenue.toFixed(1)}K</div>
                  <div className="text-sm text-text-muted">예상 수익 (천원)</div>
                </div>
                <div className="bg-background rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-primary">{segments.length}</div>
                  <div className="text-sm text-text-muted">입찰 구간</div>
                </div>
              </div>

              {/* Bid Segments Preview */}
              <div className="bg-background rounded-lg p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">입찰 구간 상세</h3>
                <div className="grid grid-cols-5 gap-2 text-xs text-text-muted font-medium mb-2">
                  <span>구간</span>
                  <span className="text-right">물량</span>
                  <span className="text-right">가격</span>
                  <span className="text-right">예상수익</span>
                  <span className="text-center">SMP대비</span>
                </div>
                <div className="space-y-1 max-h-40 overflow-y-auto">
                  {segments.map((seg) => {
                    const belowSMP = seg.price <= smpForecast.q50;
                    return (
                      <div
                        key={seg.id}
                        className={clsx(
                          'grid grid-cols-5 gap-2 p-2 rounded text-sm',
                          belowSMP ? 'bg-success/10' : 'bg-card'
                        )}
                      >
                        <span className="text-text-primary">Seg {seg.id}</span>
                        <span className="text-right text-text-muted">{seg.quantity} MW</span>
                        <span className="text-right text-text-primary font-mono">{seg.price}원</span>
                        <span className="text-right text-text-muted">{((seg.quantity * seg.price) / 1000).toFixed(1)}K</span>
                        <span className="text-center">
                          {belowSMP ? (
                            <span className="text-success">✓</span>
                          ) : (
                            <span className="text-warning">△</span>
                          )}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Validation Checklist */}
              <div className="bg-background rounded-lg p-4">
                <h3 className="text-sm font-semibold text-text-primary mb-3">검증 체크리스트</h3>
                <div className="space-y-3">
                  {validations.map((v) => (
                    <label
                      key={v.key}
                      className="flex items-start gap-3 cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={checklist[v.key as keyof typeof checklist]}
                        onChange={(e) => setChecklist(prev => ({
                          ...prev,
                          [v.key]: e.target.checked,
                        }))}
                        className="mt-1 w-4 h-4 rounded border-border text-primary focus:ring-primary"
                      />
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-text-primary font-medium">{v.label}</span>
                          {v.passed ? (
                            <CheckCircle className="w-4 h-4 text-success" />
                          ) : (
                            <AlertTriangle className="w-4 h-4 text-warning" />
                          )}
                        </div>
                        <p className="text-xs text-text-muted mt-0.5">{v.description}</p>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              {/* Remarks */}
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  검토 의견 (선택)
                </label>
                <textarea
                  value={remarks}
                  onChange={(e) => setRemarks(e.target.value)}
                  placeholder="승인 또는 반려 사유를 입력하세요..."
                  className="w-full bg-background border border-border rounded-lg px-4 py-3 text-text-primary placeholder:text-text-muted resize-none"
                  rows={3}
                />
              </div>

              {/* Approver Info */}
              <div className="flex items-center gap-4 p-4 bg-background rounded-lg">
                <div className="w-12 h-12 bg-primary/20 rounded-full flex items-center justify-center">
                  <User className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <div className="text-sm font-medium text-text-primary">승인자: 관리자</div>
                  <div className="text-xs text-text-muted flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date().toLocaleString('ko-KR')}
                  </div>
                </div>
              </div>
            </>
          )}

          {step === 'approving' && (
            <div className="py-12 text-center">
              <Loader2 className="w-16 h-16 text-primary mx-auto animate-spin" />
              <h3 className="text-xl font-bold text-text-primary mt-6">입찰 승인 처리 중...</h3>
              <p className="text-text-muted mt-2">잠시만 기다려주세요</p>
              <div className="mt-6 space-y-2 text-sm text-text-muted">
                <p>✓ 입찰 데이터 검증 완료</p>
                <p>✓ 전자서명 적용 중...</p>
                <p className="animate-pulse">○ 승인 처리 중...</p>
              </div>
            </div>
          )}

          {step === 'approved' && (
            <div className="py-12 text-center">
              <div className="w-20 h-20 bg-success/20 rounded-full flex items-center justify-center mx-auto">
                <CheckCircle className="w-10 h-10 text-success" />
              </div>
              <h3 className="text-xl font-bold text-success mt-6">승인 완료!</h3>
              <p className="text-text-muted mt-2">입찰이 승인되었습니다. KPX 제출이 가능합니다.</p>
              <div className="mt-6 flex items-center justify-center gap-2 text-primary">
                <span>KPX 제출 대기</span>
                <ArrowRight className="w-4 h-4" />
              </div>
            </div>
          )}

          {step === 'rejected' && (
            <div className="py-12 text-center">
              <div className="w-20 h-20 bg-danger/20 rounded-full flex items-center justify-center mx-auto">
                <XCircle className="w-10 h-10 text-danger" />
              </div>
              <h3 className="text-xl font-bold text-danger mt-6">반려됨</h3>
              <p className="text-text-muted mt-2">입찰이 반려되었습니다. 수정 후 다시 제출해주세요.</p>
              {remarks && (
                <div className="mt-4 p-4 bg-background rounded-lg text-left max-w-md mx-auto">
                  <p className="text-sm text-text-muted">반려 사유:</p>
                  <p className="text-sm text-text-primary mt-1">{remarks}</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        {step === 'review' && (
          <div className="sticky bottom-0 bg-card border-t border-border px-6 py-4 flex items-center justify-between">
            <button
              onClick={handleReject}
              className="px-6 py-2.5 text-danger border border-danger rounded-lg hover:bg-danger/10 transition-colors flex items-center gap-2"
            >
              <XCircle className="w-4 h-4" />
              반려
            </button>
            <div className="flex items-center gap-3">
              <button
                onClick={onClose}
                className="px-6 py-2.5 text-text-muted hover:bg-background rounded-lg transition-colors"
              >
                취소
              </button>
              <button
                onClick={handleApprove}
                disabled={!allChecked}
                className={clsx(
                  'px-6 py-2.5 rounded-lg flex items-center gap-2 transition-colors',
                  allChecked
                    ? 'bg-success text-white hover:bg-success/90'
                    : 'bg-background text-text-muted cursor-not-allowed'
                )}
              >
                <CheckCircle className="w-4 h-4" />
                승인
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
