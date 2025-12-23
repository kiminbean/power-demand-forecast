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
import { apiService, SettlementRecord, SettlementStats, CurrentSMP, SMPForecast } from '../services/api';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Design colors from Figma
const colors = {
  primary: '#04265e',
  secondary: '#0048ff',
  background: '#ffffff',
  cardBg: '#f8f9fa',
  smpCardBg: '#2563eb',
  text: '#000000',
  textSecondary: '#666666',
  textMuted: '#999999',
  orange: '#f59e0b',
  blue: '#2563eb',
  green: '#10b981',
  border: '#e5e7eb',
  chartBg: '#f1f5f9',
};

// Daily revenue data type
interface DailyRevenue {
  date: string;
  revenue: number;
  imbalance: number;
}

// Mock revenue data
const mockRevenueData: DailyRevenue[] = [
  { date: '12/17', revenue: 180, imbalance: -5.2 },
  { date: '12/18', revenue: 195, imbalance: -8.1 },
  { date: '12/19', revenue: 175, imbalance: -3.5 },
  { date: '12/20', revenue: 210, imbalance: -10.2 },
  { date: '12/21', revenue: 188, imbalance: -6.8 },
  { date: '12/22', revenue: 165, imbalance: -4.5 },
  { date: '12/23', revenue: 138, imbalance: -7.0 },
];

// Transaction data type
interface Transaction {
  date: string;
  type: 'generation' | 'imbalance' | 'settlement';
  amount: number;
  description: string;
}

// Mock transactions
const mockTransactions: Transaction[] = [
  { date: '12/23', type: 'generation', amount: 45.2, description: '발전 수익' },
  { date: '12/23', type: 'imbalance', amount: -3.5, description: '불균형 정산' },
  { date: '12/22', type: 'generation', amount: 52.1, description: '발전 수익' },
  { date: '12/22', type: 'imbalance', amount: -4.2, description: '불균형 정산' },
  { date: '12/21', type: 'settlement', amount: 158.3, description: '주간 정산' },
];

// Simple Revenue Chart Component
function RevenueChart({ data }: { data: DailyRevenue[] }) {
  const maxRevenue = Math.max(...data.map(d => Math.abs(d.revenue)), 1);
  const chartHeight = 100; // Fixed height for bars area

  return (
    <View style={styles.chartContainer}>
      <View style={styles.chartArea}>
        {data.map((item, index) => {
          const barHeight = Math.max((Math.abs(item.revenue) / maxRevenue) * chartHeight, 4);
          return (
            <View key={`${item.date}-${index}`} style={styles.chartBarWrapper}>
              <View style={[styles.chartBar, { height: barHeight }]} />
              <Text style={styles.chartBarLabel}>{item.date.split('/')[1] || item.date}</Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

export default function SettlementScreen() {
  const [isTransactionsExpanded, setIsTransactionsExpanded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // API data states
  const [settlements, setSettlements] = useState<SettlementRecord[]>([]);
  const [summary, setSummary] = useState<SettlementStats | null>(null);
  const [currentSMPData, setCurrentSMPData] = useState<CurrentSMP | null>(null);
  const [smpForecast, setSmpForecast] = useState<SMPForecast | null>(null);

  // Transform settlements to chart data format
  const revenueData = settlements.length > 0 ? settlements.slice(0, 7).map(s => ({
    date: s.date.slice(-5).replace('-', '/'), // Format: MM/DD
    revenue: s.net_revenue_million,
    imbalance: s.imbalance_million,
  })) : mockRevenueData;

  // Transform settlements to transaction format
  const transactions = settlements.length > 0 ? settlements.flatMap(s => [
    { date: s.date.slice(-5).replace('-', '/'), type: 'generation' as const, amount: s.revenue_million, description: '발전 수익' },
    { date: s.date.slice(-5).replace('-', '/'), type: 'imbalance' as const, amount: s.imbalance_million, description: '불균형 정산' },
  ]).slice(0, 6) : mockTransactions;

  // Fetch settlement data from API
  const fetchSettlementData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      // Fetch all data in parallel
      const [settlementsData, summaryData, smpData, forecastData] = await Promise.all([
        apiService.getRecentSettlements(7),
        apiService.getSettlementSummary(),
        apiService.getCurrentSMP('jeju'),
        apiService.getSMPForecast(),
      ]);

      setSettlements(settlementsData);
      setSummary(summaryData);
      setCurrentSMPData(smpData);
      setSmpForecast(forecastData);
    } catch (error) {
      console.log('API unavailable, using mock data:', error);
      // Keep mock data on error
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchSettlementData();
  }, [fetchSettlementData]);

  const onRefresh = () => fetchSettlementData(true);

  // Calculate display values from API or fallback to defaults
  const totalRevenue = summary?.generation_revenue_million ?? 1251;
  const totalImbalance = summary?.imbalance_charges_million ?? -45.3;
  const accuracy = summary?.forecast_accuracy_pct ?? 94.5;
  const totalGeneration = Math.round(settlements.reduce((sum, s) => sum + s.generation_mwh, 0) * 10) / 10;
  const avgAccuracy = settlements.length > 0
    ? Math.round(settlements.reduce((sum, s) => sum + s.accuracy_pct, 0) / settlements.length * 10) / 10
    : 0.0;

  // Current SMP values
  const currentSMP = currentSMPData?.current_smp ?? 71.2;
  const currentHour = currentSMPData?.hour ?? new Date().getHours();

  // SMP high/low from forecast
  const smpHigh = smpForecast ? {
    value: Math.round(Math.max(...smpForecast.q50)),
    hour: smpForecast.q50.indexOf(Math.max(...smpForecast.q50)),
  } : { value: 102, hour: 18 };
  const smpLow = smpForecast ? {
    value: Math.round(Math.min(...smpForecast.q50)),
    hour: smpForecast.q50.indexOf(Math.min(...smpForecast.q50)),
  } : { value: 71, hour: 12 };

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
          <TouchableOpacity style={styles.periodBadge}>
            <Text style={styles.periodIcon}>📅</Text>
            <Text style={styles.periodText}>7일</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.subtitle}>최근 7일 정산 현황</Text>
      </View>

      {/* Current SMP Card */}
      <View style={styles.smpCard}>
        <View style={styles.smpCardLeft}>
          <Text style={styles.smpCardLabel}>현재 SMP ({currentHour}시)</Text>
          <Text style={styles.smpCardValue}>{Math.round(currentSMP)}</Text>
          <Text style={styles.smpCardUnit}>원/kWh</Text>
        </View>
        <View style={styles.smpCardRight}>
          <View style={styles.smpHighLow}>
            <Text style={styles.highLowIcon}>↗</Text>
            <Text style={styles.highLowText}>최고 {smpHigh.value} ({smpHigh.hour}시)</Text>
          </View>
          <View style={styles.smpHighLow}>
            <Text style={styles.highLowIcon}>↘</Text>
            <Text style={styles.highLowText}>최저 {smpLow.value} ({smpLow.hour}시)</Text>
          </View>
        </View>
      </View>

      {/* Stats Row */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>총 발전량</Text>
          <View style={styles.statValueRow}>
            <Text style={styles.statValue}>{totalGeneration}</Text>
            <Text style={styles.statUnit}> MWh</Text>
          </View>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>예측 정확도</Text>
          <View style={styles.statValueRow}>
            <Text style={styles.statValue}>{avgAccuracy.toFixed(1)}%</Text>
            <Text style={styles.statUnit}> 평균</Text>
          </View>
        </View>
      </View>

      {/* Daily Revenue Section */}
      <View style={styles.chartSection}>
        <Text style={styles.sectionTitle}>일별 순수익</Text>
        <View style={styles.chartCard}>
          <RevenueChart data={revenueData} />
        </View>
      </View>

      {/* Revenue Stats */}
      <View style={styles.revenueStatsRow}>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>발전수익</Text>
          <Text style={styles.revenueStatValue}>{totalRevenue} M</Text>
        </View>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>불균형</Text>
          <Text style={[styles.revenueStatValue, { color: colors.orange }]}>
            {totalImbalance} M
          </Text>
        </View>
        <View style={styles.revenueStat}>
          <Text style={styles.revenueStatLabel}>정확도</Text>
          <Text style={[styles.revenueStatValue, { color: colors.blue }]}>
            {accuracy}%
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

  // SMP Card
  smpCard: {
    backgroundColor: colors.smpCardBg,
    borderRadius: 16,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  smpCardLeft: {
    flex: 1,
  },
  smpCardLabel: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.8)',
    marginBottom: 4,
  },
  smpCardValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#ffffff',
    lineHeight: 56,
  },
  smpCardUnit: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
  },
  smpCardRight: {
    justifyContent: 'center',
    alignItems: 'flex-end',
  },
  smpHighLow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  highLowIcon: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.9)',
    marginRight: 4,
  },
  highLowText: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.9)',
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
    height: 160,
  },
  chartContainer: {
    flex: 1,
  },
  chartArea: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'space-around',
    paddingBottom: 20,
  },
  chartBarWrapper: {
    alignItems: 'center',
    flex: 1,
  },
  chartBar: {
    width: 24,
    backgroundColor: '#94a3b8',
    borderRadius: 4,
    marginBottom: 8,
  },
  chartBarLabel: {
    fontSize: 10,
    color: colors.textMuted,
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
