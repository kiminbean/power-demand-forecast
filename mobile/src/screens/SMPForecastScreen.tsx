/**
 * SMP Forecast Screen - Page 1
 * Figma: iPhone 16 Pro - 11 (id: 2:100)
 * Design: SMP 예측 with 24-hour chart and hourly table
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
  Image,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { apiService, SMPForecast, ModelInfo } from '../services/api';

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
  green: '#10b981',
  border: '#e5e7eb',
};

// SMP hourly data type
interface SMPHourlyData {
  hour: string;
  price: number;
  rangeMin: number;
  rangeMax: number;
}

// Transform API data to hourly format
const transformSMPData = (forecast: SMPForecast): SMPHourlyData[] => {
  const { q10, q50, q90 } = forecast;
  return q50.map((price, i) => ({
    hour: `${String(i).padStart(2, '0')}시`,
    price: Math.round(price * 10) / 10,
    rangeMin: Math.round(q10[i]),
    rangeMax: Math.round(q90[i]),
  }));
};

// Fallback mock data when API is unavailable
const generateMockSMPData = (): SMPHourlyData[] => {
  const basePrice = 71;
  return Array.from({ length: 24 }, (_, i) => {
    const variation = Math.sin(i * 0.5) * 15 + Math.random() * 10;
    const price = Math.round((basePrice + variation) * 10) / 10;
    return {
      hour: `${String(i).padStart(2, '0')}시`,
      price,
      rangeMin: Math.round(price * 0.75),
      rangeMax: Math.round(price * 1.5),
    };
  });
};

// Simple Line Chart Component
function SMPChart({ data }: { data: SMPHourlyData[] }) {
  const maxPrice = Math.max(...data.map(d => d.price));
  const minPrice = Math.min(...data.map(d => d.price));
  const range = maxPrice - minPrice || 1;

  return (
    <View style={styles.chartContainer}>
      {/* Y-axis and Chart Area Row */}
      <View style={styles.chartRow}>
        <View style={styles.chartYAxis}>
          <Text style={styles.chartAxisLabel}>{Math.round(maxPrice + 10)}</Text>
          <Text style={styles.chartAxisLabel}>{Math.round((maxPrice + minPrice) / 2)}</Text>
          <Text style={styles.chartAxisLabel}>{Math.round(minPrice - 10)}</Text>
        </View>
        <View style={styles.chartArea}>
          {/* Chart background grid */}
          <View style={styles.chartGrid}>
            <View style={styles.chartGridLine} />
            <View style={styles.chartGridLine} />
            <View style={styles.chartGridLine} />
          </View>
          {/* Area fill */}
          <View style={styles.chartFill}>
            {data.map((item, index) => {
              const height = ((item.price - minPrice + 10) / (range + 20)) * 100;
              return (
                <View
                  key={index}
                  style={[
                    styles.chartBar,
                    { height: `${height}%` },
                  ]}
                />
              );
            })}
          </View>
          {/* Current time indicator */}
          <View style={[styles.currentTimeIndicator, { left: '75%' }]}>
            <View style={styles.currentTimeLine} />
          </View>
        </View>
      </View>
      {/* X-axis aligned with chart area */}
      <View style={styles.chartXAxisRow}>
        <View style={styles.chartYAxisSpacer} />
        <View style={styles.chartXAxis}>
          <Text style={styles.chartAxisLabel}>00시</Text>
          <Text style={styles.chartAxisLabel}>06시</Text>
          <Text style={styles.chartAxisLabel}>12시</Text>
          <Text style={styles.chartAxisLabel}>18시</Text>
        </View>
      </View>
    </View>
  );
}


export default function SMPForecastScreen() {
  const [smpData, setSmpData] = useState<SMPHourlyData[]>([]);
  const [currentSMP, setCurrentSMP] = useState(71.2);
  const [highSMP, setHighSMP] = useState({ price: 102, hour: 18 });
  const [lowSMP, setLowSMP] = useState({ price: 71, hour: 12 });
  const [lastUpdate, setLastUpdate] = useState('오전 10:48');
  const [confidence, setConfidence] = useState(92);
  const [avgPrice, setAvgPrice] = useState(73);
  const [modelInfo, setModelInfo] = useState('LSTM-Attention v3.1 | MAPE 4.23%');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [currentHour, setCurrentHour] = useState(new Date().getHours());

  // Fetch SMP forecast data from API
  const fetchSMPData = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      // Fetch forecast and model info in parallel
      const [forecast, model] = await Promise.all([
        apiService.getSMPForecast(),
        apiService.getModelInfo(),
      ]);

      // Transform data
      const transformedData = transformSMPData(forecast);
      setSmpData(transformedData);

      // Calculate current hour SMP
      const now = new Date();
      const hour = now.getHours();
      setCurrentHour(hour);
      setCurrentSMP(Math.round(forecast.q50[hour] * 10) / 10);

      // Find high/low
      const prices = forecast.q50;
      const maxPrice = Math.max(...prices);
      const minPrice = Math.min(...prices);
      const maxHour = prices.indexOf(maxPrice);
      const minHour = prices.indexOf(minPrice);
      setHighSMP({ price: Math.round(maxPrice), hour: maxHour });
      setLowSMP({ price: Math.round(minPrice), hour: minHour });

      // Set confidence and average
      setConfidence(Math.round(forecast.confidence * 100));
      setAvgPrice(Math.round(prices.reduce((a, b) => a + b, 0) / prices.length));

      // Set model info
      if (model.version && model.mape) {
        setModelInfo(`${model.type || 'LSTM-Attention'} ${model.version} | MAPE ${model.mape}%`);
      }

      // Update time
      const minutes = now.getMinutes();
      const ampm = hour < 12 ? '오전' : '오후';
      const hour12 = hour % 12 || 12;
      setLastUpdate(`${ampm} ${hour12}:${String(minutes).padStart(2, '0')}`);

    } catch (error) {
      console.log('API unavailable, using mock data:', error);
      // Fallback to mock data
      const mockData = generateMockSMPData();
      setSmpData(mockData);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchSMPData();

    // Auto refresh every 5 minutes
    const interval = setInterval(() => fetchSMPData(), 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchSMPData]);

  const onRefresh = () => fetchSMPData(true);

  // Show loading spinner on initial load
  if (loading && smpData.length === 0) {
    return (
      <View style={[styles.container, styles.loadingContainer]}>
        <ActivityIndicator size="large" color={colors.secondary} />
        <Text style={styles.loadingText}>SMP 데이터 로딩중...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
      >
        {/* Title Section */}
        <View style={styles.titleSection}>
          <View style={styles.titleRow}>
            <Text style={styles.pageTitle}>SMP 예측</Text>
            <TouchableOpacity style={styles.refreshButton} onPress={onRefresh}>
              <Text style={styles.refreshIcon}>↻</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.subtitle}>{modelInfo}</Text>
        </View>

        {/* Current SMP Card */}
        <View style={styles.smpCard}>
          <View style={styles.smpCardLeft}>
            <Text style={styles.smpCardLabel}>현재 SMP ({currentHour}시)</Text>
            <Text style={styles.smpCardValue}>{currentSMP}</Text>
            <Text style={styles.smpCardUnit}>원/kWh</Text>
          </View>
          <View style={styles.smpCardRight}>
            <View style={styles.smpHighLow}>
              <Text style={styles.highLowIcon}>↗</Text>
              <Text style={styles.highLowText}>최고 {highSMP.price} ({highSMP.hour}시)</Text>
            </View>
            <View style={styles.smpHighLow}>
              <Text style={styles.highLowIcon}>↘</Text>
              <Text style={styles.highLowText}>최저 {lowSMP.price} ({lowSMP.hour}시)</Text>
            </View>
          </View>
        </View>

        {/* 24 Hour Chart */}
        <View style={styles.chartSection}>
          <Text style={styles.sectionTitle}>24시간 예측</Text>
          <SMPChart data={smpData} />
        </View>

        {/* Update Time & Stats */}
        <Text style={styles.updateTime}>업데이트 {lastUpdate}</Text>

        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statLabel}>신뢰도</Text>
            <Text style={[styles.statValue, { color: colors.orange }]}>{confidence}%</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statLabel}>평균</Text>
            <Text style={[styles.statValue, { color: colors.orange }]}>{avgPrice}</Text>
          </View>
        </View>

        {/* Hourly Detail Table */}
        <View style={styles.tableSection}>
          <Text style={styles.sectionTitle}>시간대별 상세</Text>

          {/* Table Header */}
          <View style={styles.tableHeader}>
            <Text style={[styles.tableHeaderText, styles.colHour]}>시간</Text>
            <Text style={[styles.tableHeaderText, styles.colPrice]}>예측 입찰가</Text>
            <Text style={[styles.tableHeaderText, styles.colRange]}>범위</Text>
          </View>

          {/* Table Rows */}
          {smpData.slice(0, 6).map((item, index) => (
            <View key={index} style={styles.tableRow}>
              <Text style={[styles.tableCell, styles.colHour]}>{item.hour}</Text>
              <Text style={[styles.tableCell, styles.colPrice, styles.priceCell]}>
                {item.price.toFixed(1)}
              </Text>
              <Text style={[styles.tableCell, styles.colRange]}>
                {item.rangeMin}~{item.rangeMax}
              </Text>
            </View>
          ))}
        </View>

        {/* Bottom padding for scroll */}
        <View style={{ height: 100 }} />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  loadingContainer: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 12,
    fontSize: 14,
    color: colors.textSecondary,
  },
  scrollView: {
    flex: 1,
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
  refreshButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  refreshIcon: {
    fontSize: 18,
    color: colors.textSecondary,
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
    marginBottom: 24,
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
  chartContainer: {
    height: 180,
  },
  chartRow: {
    flex: 1,
    flexDirection: 'row',
  },
  chartYAxis: {
    width: 35,
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    paddingRight: 8,
    paddingBottom: 5,
  },
  chartYAxisSpacer: {
    width: 35,
  },
  chartArea: {
    flex: 1,
    position: 'relative',
  },
  chartGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  chartGridLine: {
    height: 1,
    backgroundColor: colors.border,
  },
  chartFill: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: '100%',
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  chartBar: {
    flex: 1,
    backgroundColor: 'rgba(245, 158, 11, 0.3)',
    marginHorizontal: 0.5,
    borderTopLeftRadius: 2,
    borderTopRightRadius: 2,
  },
  currentTimeIndicator: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
  },
  currentTimeLine: {
    flex: 1,
    backgroundColor: '#ef4444',
    width: 2,
  },
  chartXAxisRow: {
    flexDirection: 'row',
    marginTop: 8,
  },
  chartXAxis: {
    flex: 1,
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
  },
  chartAxisLabel: {
    fontSize: 10,
    color: colors.textMuted,
  },

  // Update Time
  updateTime: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: 16,
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
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 13,
    color: colors.textSecondary,
    marginBottom: 4,
  },
  statValue: {
    fontSize: 28,
    fontWeight: 'bold',
  },

  // Table Section
  tableSection: {
    marginBottom: 24,
  },
  tableHeader: {
    flexDirection: 'row',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.cardBg,
    borderTopLeftRadius: 8,
    borderTopRightRadius: 8,
    paddingHorizontal: 16,
  },
  tableHeaderText: {
    fontSize: 13,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  tableRow: {
    flexDirection: 'row',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
    backgroundColor: colors.background,
  },
  tableCell: {
    fontSize: 14,
    color: colors.text,
  },
  colHour: {
    width: 60,
  },
  colPrice: {
    flex: 1,
    textAlign: 'center',
  },
  colRange: {
    width: 80,
    textAlign: 'right',
  },
  priceCell: {
    color: colors.orange,
    fontWeight: '600',
  },
});
