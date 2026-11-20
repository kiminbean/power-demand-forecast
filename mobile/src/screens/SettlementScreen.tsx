/**
 * Settlement Screen - Page 4
 * Figma: iPhone 16 Pro - 14 (id: 2:374)
 * Design: 정산 with daily revenue chart and stats
 * Uses real API data from backend
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Platform,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';

import { colors as themeColors } from '../theme/colors';
import { apiService, SettlementRecord, SettlementStats } from '../services/api';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Design colors from Figma
const colors = {
  primary: themeColors.navy,
  secondary: themeColors.info,
  background: themeColors.bg,
  cardBg: themeColors.surface,
  smpCardBg: themeColors.info,
  text: themeColors.text.primary,
  textSecondary: themeColors.text.secondary,
  textMuted: themeColors.text.secondary,
  orange: themeColors.warning,
  blue: themeColors.info,
  green: themeColors.accent,
  border: themeColors.border.primary,
  chartBg: themeColors.surfaceAlt
};

// Daily revenue data type
interface DailyRevenue {
  date: string;
  revenue: number;
  imbalance: number;
}

// Signed values avoid presenting losses as positive bars.
function RevenueChart({ data }: { data: DailyRevenue[] }) {
  return (
    <View style={styles.chartContainer}>
      {data.map((item, index) => (
        <View key={`${item.date}-${index}`} style={styles.revenueRow}>
          <Text style={styles.revenueDate}>{item.date}</Text>
          <Text style={styles.revenueAmount}>{item.revenue > 0 ? '+' : ''}{item.revenue} M</Text>
        </View>
      ))}
    </View>
  );
}

export default function SettlementScreen() {
  const [isTransactionsExpanded, setIsTransactionsExpanded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<'unavailable' | 'error' | null>(null);


  // API data states
  const [settlements, setSettlements] = useState<SettlementRecord[]>([]);
  const [summary, setSummary] = useState<SettlementStats | null>(null);

  const validNumber = (value: unknown): value is number =>
    typeof value === 'number' && Number.isFinite(value);
  const validDate = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
  const dateLabel = (date: string) => date.slice(-5).replace('-', '/');
  const revenueData = settlements.filter(s => s && validDate(s.date) && validNumber(s.net_revenue_million))
    .slice(0, 7).map(s => ({ date: dateLabel(s.date), revenue: s.net_revenue_million, imbalance: 0 }));
  const transactions = settlements.flatMap(s => {
    if (!s || !validDate(s.date)) return [];
    const date = dateLabel(s.date);
    return [
      ...(validNumber(s.revenue_million) ? [{ date, amount: s.revenue_million, description: '발전 수익' }] : []),
      ...(validNumber(s.imbalance_million) ? [{ date, amount: s.imbalance_million, description: '불균형 정산' }] : []),
    ];
  }).slice(0, 6);

  // Fetch settlement data from API
  const fetchSettlementData = useCallback(async (isRefresh = false) => {
    setError(null);
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      // Fetch all data in parallel
      const [settlementsData, summaryData] = await Promise.all([
        apiService.getRecentSettlements(7),
        apiService.getSettlementSummary(),
      ]);

      setSettlements(Array.isArray(settlementsData) ? settlementsData : []);
      setSummary(summaryData && typeof summaryData === 'object' && !Array.isArray(summaryData) ? summaryData : null);
    } catch (error) {
      setSettlements([]);
      setSummary(null);
      setError(error instanceof Error && error.message === 'API Error: 404' ? 'unavailable' : 'error');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchSettlementData();
  }, [fetchSettlementData]);

  const onRefresh = () => fetchSettlementData(true);

  // A missing field is unavailable, not zero or a sample value.
  const totalRevenue = summary?.generation_revenue_million;
  const totalImbalance = summary?.imbalance_charges_million;
  const accuracy = summary?.forecast_accuracy_pct;
  const completeGeneration = settlements.length > 0 && settlements.every(s => s && validNumber(s.generation_mwh));
  const completeAccuracy = settlements.length > 0 && settlements.every(s => s && validNumber(s.accuracy_pct));
  const totalGeneration = completeGeneration
    ? Math.round(settlements.reduce((sum, s) => sum + s.generation_mwh, 0) * 10) / 10 : null;
  const avgAccuracy = completeAccuracy
    ? Math.round(settlements.reduce((sum, s) => sum + s.accuracy_pct, 0) / settlements.length * 10) / 10 : null;
  const smpSummary = summary as (SettlementStats & { avg_da_smp?: unknown; avg_rt_smp?: unknown }) | null;
  const avgDaSmp = smpSummary?.avg_da_smp;
  const avgRtSmp = smpSummary?.avg_rt_smp;
  const display = (value: unknown) => validNumber(value) ? String(value) : '—';

  return (
    <ScrollView
      style={styles.container}
      showsVerticalScrollIndicator={false}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      {/* Title Section */}
      <View style={styles.titleSection}>
        <View style={styles.titleRow}>
          <Text style={styles.pageTitle}>정산</Text>
          {(summary?.is_simulated ?? true) && (
            <View style={styles.mockBadge}>
              <Text style={styles.mockBadgeText}>모의 결과</Text>
            </View>
          )}
        </View>
        <Text style={styles.subtitle}>KPX 제주 시범사업 이중 정산</Text>
      </View>

      {loading && <View><ActivityIndicator /><Text>정산 정보를 불러오는 중입니다.</Text></View>}
      {error && <View><Text>{error === 'unavailable' ? '정산 정보를 현재 이용할 수 없습니다.' : '정산 정보를 불러오지 못했습니다.'}</Text>
        <TouchableOpacity onPress={() => fetchSettlementData()}><Text>다시 시도</Text></TouchableOpacity>
      </View>}
      {!loading && !error && settlements.length === 0 && <Text>정산 내역이 없습니다.</Text>}
      {!loading && !error && <>
      <Text style={styles.periodLabel}>최근 7일 정산 내역</Text>

      {/* Average DA-SMP Card */}
      <View style={styles.smpCard}>
        <View style={styles.smpCardContent}>
          <Text style={styles.smpCardLabel}>평균 DA-SMP (7일)</Text>
          <Text style={styles.smpCardValue}>{validNumber(avgDaSmp) ? Math.round(avgDaSmp) : '—'}</Text>
          <Text style={styles.smpCardUnit}>원/kWh</Text>
        </View>
      </View>

      {/* Stats Row */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>총 발전량</Text>
          <View style={styles.statValueRow}>
            <Text style={styles.statValue}>{display(totalGeneration)}</Text>
            <Text style={styles.statUnit}> MWh</Text>
          </View>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>예측 정확도</Text>
          <View style={styles.statValueRow}>
            <Text style={[styles.statValue, { color: (avgAccuracy ?? 0) >= 95 ? colors.green : colors.orange }]}>
              {avgAccuracy !== null ? `${avgAccuracy.toFixed(1)}%` : '—'}
            </Text>
            <Text style={styles.statUnit}> 평균</Text>
          </View>
        </View>
      </View>

      {/* DA/RT SMP Info */}
      <View style={styles.smpInfoRow}>
        <View style={styles.smpInfoCard}>
          <Text style={styles.smpInfoLabel}>평균 DA-SMP</Text>
          <Text style={[styles.smpInfoValue, { color: themeColors.info }]}>
            {validNumber(avgDaSmp) ? avgDaSmp.toFixed(1) : '—'}
            <Text style={styles.smpInfoUnit}> 원/kWh</Text>
          </Text>
        </View>
        <View style={styles.smpInfoCard}>
          <Text style={styles.smpInfoLabel}>평균 RT-SMP</Text>
          <Text style={[styles.smpInfoValue, { color: '#8b5cf6' }]}>
            {validNumber(avgRtSmp) ? avgRtSmp.toFixed(1) : '—'}
            <Text style={styles.smpInfoUnit}> 원/kWh</Text>
          </Text>
        </View>
      </View>

      {/* Daily Revenue Section */}
      <View style={styles.chartSection}>
        <Text style={styles.sectionTitle}>일별 순수익</Text>
        <View style={styles.chartCard}>
          {revenueData.length > 0 ? <RevenueChart data={revenueData} /> : <Text>수익 데이터가 없습니다.</Text>}
        </View>
      </View>

      {/* Revenue Stats */}
      <View style={styles.revenueStatsRow}>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>발전수익</Text>
          <Text style={styles.revenueStatValue}>{display(totalRevenue)} M</Text>
        </View>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>불균형</Text>
          <Text style={[styles.revenueStatValue, { color: colors.orange }]}>
            {display(totalImbalance)} M
          </Text>
        </View>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>정확도</Text>
          <Text style={[styles.revenueStatValue, { color: colors.blue }]}>
            {validNumber(accuracy) ? `${accuracy}%` : '—'}
          </Text>
        </View>
      </View>

      {/* Transaction History */}
      <TouchableOpacity
        style={styles.transactionSection}
        onPress={() => setIsTransactionsExpanded(!isTransactionsExpanded)}
      >
        <View style={styles.transactionHeader}>
          <Text style={styles.sectionTitle}>거래 내역</Text>
          <Text style={styles.expandIcon}>{isTransactionsExpanded ? '∨' : '>'}</Text>
        </View>
      </TouchableOpacity>

      {isTransactionsExpanded && (
        <View style={styles.transactionList}>
          {transactions.length === 0 && <Text>거래 내역이 없습니다.</Text>}
          {transactions.map((transaction, index) => (
            <View key={index} style={styles.transactionRow}>
              <View style={styles.transactionLeft}>
                <Text style={styles.transactionDate}>{transaction.date}</Text>
                <Text style={styles.transactionDesc}>{transaction.description}</Text>
              </View>
              <Text
                style={[
                  styles.transactionAmount,
                  transaction.amount < 0 && { color: colors.orange },
                ]}
              >
                {transaction.amount > 0 ? '+' : ''}{transaction.amount.toFixed(1)} M
              </Text>
            </View>
          ))}
        </View>
      )}

      </>}
      {/* Bottom padding */}
      <View style={{ height: 100 }} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
    paddingHorizontal: 16,
  },

  // Title Section
  titleSection: {
    marginTop: 16,
    marginBottom: 16,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  pageTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text,
  },
  mockBadge: {
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.4)',
  },
  mockBadgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: themeColors.warning,
  },
  periodBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.cardBg,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
  },
  periodIcon: {
    fontSize: 14,
    marginRight: 4,
  },
  periodText: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.text,
  },
  subtitle: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: 4,
  },

  periodLabel: {
    fontSize: 13,
    fontWeight: '500',
    color: colors.textSecondary,
    marginBottom: 16,
  },

  // SMP Card
  smpCard: {
    backgroundColor: colors.smpCardBg,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  smpCardContent: {
    alignItems: 'center',
  },
  smpCardLabel: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 4,
  },
  smpCardValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: themeColors.text.primary,
    lineHeight: 56,
  },
  smpCardUnit: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
  },

  // Stats Row
  statsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
  },
  statLabel: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  statValueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text,
  },
  statUnit: {
    fontSize: 14,
    color: colors.textSecondary,
  },

  // DA/RT SMP Info
  smpInfoRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  smpInfoCard: {
    flex: 1,
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 14,
  },
  smpInfoLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  smpInfoValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  smpInfoUnit: {
    fontSize: 12,
    color: colors.textSecondary,
    fontWeight: 'normal',
  },

  // Chart Section
  chartSection: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text,
    marginBottom: 12,
  },
  chartCard: {
    backgroundColor: colors.chartBg,
    borderRadius: 12,
    padding: 16,
  },
  chartContainer: {
  },
  revenueRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  revenueDate: {
    fontSize: 13,
    color: colors.textSecondary,
  },
  revenueAmount: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
  },

  // Revenue Stats Row
  revenueStatsRow: {
    flexDirection: 'row',
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  revenueStat: {
    flex: 1,
    alignItems: 'center',
  },
  revenueStatLabel: {
    fontSize: 12,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  revenueStatValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text,
  },

  // Transaction Section
  transactionSection: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
  },
  transactionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  expandIcon: {
    fontSize: 18,
    color: colors.textSecondary,
  },
  transactionList: {
    backgroundColor: colors.cardBg,
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  transactionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  transactionLeft: {
    flex: 1,
  },
  transactionDate: {
    fontSize: 12,
    color: colors.textMuted,
  },
  transactionDesc: {
    fontSize: 14,
    color: colors.text,
    marginTop: 2,
  },
  transactionAmount: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.green,
  },
});
