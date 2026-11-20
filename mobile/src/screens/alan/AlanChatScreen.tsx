/**
 * Alan Chat Screen - Page 2
 * Figma: iPhone 16 Pro - 12 (id: 2:219)
 * Design: Alan AI main screen with quick actions and chat input
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Dimensions,
  Animated,
} from 'react-native';

import { colors as themeColors } from '../../theme/colors';
import { alanApi, AlanMessage, AlanResponse } from '../../services/alan/alanApi';
import { apiService } from '../../services/api';
import { describeCurrentSmp, formatNullableNumber, publishedSmpPrice } from '../../utils/smpPresentation';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Design colors from Figma (Page 2: Alan Chat)
const colors = {
  primary: themeColors.navy,
  secondary: themeColors.info,
  background: themeColors.bg,
  gradientStart: themeColors.surface,
  gradientEnd: themeColors.surfaceAlt,
  white: themeColors.text.primary,
  text: themeColors.text.primary,
  textSecondary: themeColors.text.secondary,
  textMuted: themeColors.text.secondary,
  border: themeColors.border.primary,
  inputBg: themeColors.surface,
  buttonBlue: themeColors.info,
  robotFace: themeColors.navy,
  robotEyes: '#00d4ff',       // Cyan eyes
};

// ============================================
// Hierarchical Keyword Navigation System
// ============================================

interface KeywordItem {
  id: string;
  label: string;
  icon: string;
  children?: KeywordItem[];
  action?: string;
}

// 4 Main Categories with 4 sub-items each (Gemini recommended structure)
const keywordTree: KeywordItem[] = [
  {
    id: 'revenue',
    label: '내 돈 관리',
    icon: '💰',
    children: [
      { id: 'revenue_month', label: '이번 달 수익', icon: '📊', action: 'show_monthly_revenue' },
      { id: 'revenue_detail', label: '정산 상세', icon: '📋', action: 'show_settlement_detail' },
      { id: 'revenue_penalty', label: '페널티 조회', icon: '⚠️', action: 'show_penalty' },
      { id: 'revenue_simulate', label: '수익 시뮬레이션', icon: '🔮', action: 'simulate_revenue' },
    ],
  },
  {
    id: 'bidding',
    label: '스마트 입찰',
    icon: '⚡',
    children: [
      { id: 'bidding_recommend', label: '내일 입찰 추천', icon: '🎯', action: 'show_bidding_recommend' },
      { id: 'bidding_generation', label: '발전량 예측', icon: '📈', action: 'show_generation_forecast' },
      { id: 'bidding_smp', label: 'SMP 예측', icon: '💹', action: 'show_smp_forecast' },
      { id: 'bidding_curtailment', label: '출력제어 확률', icon: '🚨', action: 'show_curtailment_prob' },
    ],
  },
  {
    id: 'plant',
    label: '발전소 상태',
    icon: '🏭',
    children: [
      { id: 'plant_realtime', label: '실시간 현황', icon: '⚡', action: 'show_realtime_status' },
      { id: 'plant_list', label: '발전소 목록', icon: '📍', action: 'show_plant_list' },
      { id: 'plant_alert', label: '설비 알림', icon: '🔔', action: 'show_equipment_alerts' },
      { id: 'plant_ess', label: 'ESS 상태', icon: '🔋', action: 'show_ess_status' },
    ],
  },
  {
    id: 'market',
    label: '시장 리포트',
    icon: '📈',
    children: [
      { id: 'market_smp', label: 'SMP 시세', icon: '💹', action: 'show_smp_price' },
      { id: 'market_weather', label: '기상 정보', icon: '🌤️', action: 'show_weather_info' },
      { id: 'market_news', label: '시장 뉴스', icon: '📰', action: 'show_market_news' },
      { id: 'market_faq', label: '자주 묻는 질문', icon: '❓', action: 'show_faq' },
    ],
  },
];

interface Props {
  onNavigate?: (screen: string, params?: any) => void;
}

// Alan Robot Avatar Component (matching Figma design - cat ears pointing up)
function AlanRobot() {
  const bounceAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Subtle floating animation
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(bounceAnim, {
          toValue: -8,
          duration: 2000,
          useNativeDriver: Platform.OS !== 'web',
        }),
        Animated.timing(bounceAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: Platform.OS !== 'web',
        }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, []);

  return (
    <Animated.View style={[styles.alanRobot, { transform: [{ translateY: bounceAnim }] }]}>
      {/* Cat Ears (pointing up) */}
      <View style={styles.earsContainer}>
        <View style={styles.catEarLeft}>
          <View style={styles.catEarInnerLeft} />
        </View>
        <View style={styles.catEarRight}>
          <View style={styles.catEarInnerRight} />
        </View>
      </View>
      {/* Robot body */}
      <View style={styles.robotBody}>
        {/* Face area (dark) */}
        <View style={styles.robotFaceArea}>
          {/* Eyes */}
          <View style={styles.robotEyes}>
            <View style={styles.robotEye}>
              <View style={styles.robotPupil} />
            </View>
            <View style={styles.robotEye}>
              <View style={styles.robotPupil} />
            </View>
          </View>
          {/* Mouth */}
          <View style={styles.robotMouth}>
            <Text style={styles.robotMouthText}>‿</Text>
          </View>
        </View>
      </View>
      {/* Robot bottom (rounded) */}
      <View style={styles.robotBottom} />
    </Animated.View>
  );
}

// Quick Action Button Component
function QuickActionButton({
  label,
  icon,
  onPress,
}: {
  label: string;
  icon: string;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity style={styles.quickActionBtn} onPress={onPress}>
      <Text style={styles.quickActionLabel}>{label}</Text>
    </TouchableOpacity>
  );
}

// Chat Message Component
function ChatMessage({ message }: { message: AlanMessage }) {
  const isUser = message.role === 'user';

  return (
    <View style={[styles.messageContainer, isUser ? styles.userMessage : styles.assistantMessage]}>
      <View style={[styles.messageBubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        <Text style={[styles.messageText, isUser && styles.userMessageText]}>
          {message.content}
        </Text>
      </View>
    </View>
  );
}

export default function AlanChatScreen({ onNavigate }: Props) {
  const [messages, setMessages] = useState<AlanMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const scrollViewRef = useRef<ScrollView>(null);

  // Hierarchical navigation state
  const [currentLevel, setCurrentLevel] = useState<'main' | 'sub'>('main');
  const [selectedMainKeyword, setSelectedMainKeyword] = useState<KeywordItem | null>(null);

  // Add assistant message helper (use unique suffix to avoid duplicate keys)
  const addAssistantMessage = (content: string) => {
    const assistantMessage: AlanMessage = {
      id: `${Date.now()}_assistant`,
      role: 'assistant',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMessage]);
  };

  // Handle keyword click (hierarchical navigation)
  const handleKeywordClick = async (keyword: KeywordItem) => {
    if (keyword.children) {
      // Has children -> show sub-menu
      setSelectedMainKeyword(keyword);
      setCurrentLevel('sub');
    } else if (keyword.action) {
      // No children -> execute action
      await executeAction(keyword.action, keyword.label);
    }
  };

  // Go back to main keywords
  const handleBackToMain = () => {
    setCurrentLevel('main');
    setSelectedMainKeyword(null);
  };

  // Execute action and show result
  const executeAction = async (action: string, label: string) => {
    setShowChat(true);
    setIsLoading(true);

    // Add user selection as message (use unique suffix to avoid duplicate keys)
    const userMessage: AlanMessage = {
      id: `${Date.now()}_user`,
      role: 'user',
      content: `${label} 정보를 보여줘`,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    try {
      let response = '';

      switch (action) {
        // ========== 내 돈 관리 ==========
        case 'show_monthly_revenue': {
          const summary = await apiService.getSettlementSummary();
          const currentDate = new Date();
          const daysInMonth = new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 0).getDate();
          const daysPassed = currentDate.getDate();
          const estimatedMonthly = (summary.net_revenue_million / daysPassed) * daysInMonth;

          response = `💰 이번 달 예상 수익\n\n` +
            `📅 ${currentDate.getMonth() + 1}월 ${daysPassed}일 현재\n\n` +
            `현재까지 수익: ${summary.net_revenue_million.toFixed(1)}백만원\n` +
            `월말 예상 수익: ${estimatedMonthly.toFixed(1)}백만원\n\n` +
            `📈 전월 대비: ${summary.net_change_pct >= 0 ? '+' : ''}${summary.net_change_pct.toFixed(1)}%`;
          break;
        }

        case 'show_settlement_detail': {
          const summary = await apiService.getSettlementSummary();
          response = `📋 정산 상세 내역\n\n` +
            `⚡ 발전 수익: ${summary.generation_revenue_million.toFixed(1)}백만원\n` +
            `   (전월 대비 ${summary.generation_change_pct >= 0 ? '+' : ''}${summary.generation_change_pct.toFixed(1)}%)\n\n` +
            `⚖️ 임밸런스 정산: ${summary.imbalance_charges_million.toFixed(1)}백만원\n` +
            `   (전월 대비 ${summary.imbalance_change_pct >= 0 ? '+' : ''}${summary.imbalance_change_pct.toFixed(1)}%)\n\n` +
            `💵 순수익: ${summary.net_revenue_million.toFixed(1)}백만원\n\n` +
            `📊 예측 정확도: ${summary.forecast_accuracy_pct.toFixed(1)}%`;
          break;
        }

        case 'show_penalty': {
          const summary = await apiService.getSettlementSummary();
          const penaltyAmount = Math.abs(summary.imbalance_charges_million);
          response = `⚠️ 페널티(임밸런스) 조회\n\n` +
            `이번 달 위약금: ${penaltyAmount.toFixed(1)}백만원\n\n` +
            `📉 전월 대비: ${summary.imbalance_change_pct >= 0 ? '+' : ''}${summary.imbalance_change_pct.toFixed(1)}%\n\n` +
            `💡 팁: 예측 정확도를 높이면 임밸런스 비용을 줄일 수 있습니다.\n` +
            `현재 예측 정확도: ${summary.forecast_accuracy_pct.toFixed(1)}%`;
          break;
        }

        case 'simulate_revenue': {
          const simulation = await apiService.simulateRevenue(50000, 'solar', 24);
          response = `🔮 수익 시뮬레이션 결과\n\n` +
            `⚡ 예상 발전량: 50MW 기준\n\n` +
            `💰 예상 수익: ${(simulation.expected_revenue / 1000000).toFixed(1)}백만원\n` +
            `📈 최대(낙관): ${(simulation.best_case / 1000000).toFixed(1)}백만원\n` +
            `📉 최소(보수): ${(simulation.worst_case / 1000000).toFixed(1)}백만원\n\n` +
            `🎯 리스크 조정 수익: ${(simulation.risk_adjusted / 1000000).toFixed(1)}백만원`;
          break;
        }

        // ========== 스마트 입찰 ==========
        case 'show_bidding_recommend': {
          const strategy = await apiService.getBiddingStrategy(50000, 'solar', 'moderate');
          const topHours = strategy.recommended_hours.slice(0, 5);
          response = `🎯 내일 입찰 추천\n\n` +
            `📊 리스크 수준: ${strategy.risk_level}\n` +
            `⚡ 총 예상 발전량: ${(strategy.total_generation_kwh / 1000).toFixed(0)}MWh\n` +
            `💰 예상 수익: ${(strategy.total_revenue / 1000000).toFixed(1)}백만원\n\n` +
            `⏰ 추천 시간대 (Top 5):\n` +
            topHours.map((h: number) => `  ${h}시`).join(', ') + '\n\n' +
            `💹 평균 SMP: ${strategy.average_smp.toFixed(0)}원/kWh`;
          break;
        }

        case 'show_generation_forecast': {
          const supply = await apiService.getPowerSupply();
          const currentHour = supply.current_hour;
          const forecastData = supply.data.filter(d => d.is_forecast).slice(0, 6);
          response = `📈 발전량 예측 (향후 6시간)\n\n` +
            `현재 시각: ${currentHour}시\n\n` +
            forecastData.map(d =>
              `${d.hour}시: 태양광 ${d.solar.toFixed(0)}MW, 풍력 ${d.wind.toFixed(0)}MW`
            ).join('\n') +
            `\n\n📊 데이터 출처: ${supply.data_source}`;
          break;
        }

        case 'show_smp_forecast': {
          const forecast = await apiService.getSMPForecast();
          const current = forecast.q50[0];
          const max = Math.max(...forecast.q50);
          const min = Math.min(...forecast.q50);
          const maxHour = forecast.q50.indexOf(max);
          const minHour = forecast.q50.indexOf(min);
          response = `💹 SMP 예측 (24시간)\n\n` +
            `📍 0시 예측 SMP: ${formatNullableNumber(current, 0)}원/kWh\n\n` +
            `📈 최고가: ${max.toFixed(0)}원 (${maxHour}시)\n` +
            `📉 최저가: ${min.toFixed(0)}원 (${minHour}시)\n\n` +
            `🎯 예측 신뢰도: ${typeof forecast.confidence === 'number' ? `${(forecast.confidence * 100).toFixed(0)}%` : '재검증 전'}\n` +
            `🤖 사용 모델: ${forecast.model_used}`;
          break;
        }

        case 'show_curtailment_prob': {
          // Jeju curtailment probability (based on renewable ratio)
          const supply = await apiService.getPowerSupply();
          const currentData = supply.data.find(d => d.hour === supply.current_hour);
          const renewableRatio = currentData ?
            ((currentData.solar + currentData.wind) / currentData.supply * 100) : 0;
          const curtailmentRisk = renewableRatio > 30 ? '높음' : renewableRatio > 20 ? '보통' : '낮음';
          response = `🚨 제주 출력제어 확률\n\n` +
            `📊 현재 재생에너지 비율: ${renewableRatio.toFixed(1)}%\n\n` +
            `⚠️ 출력제어 위험도: ${curtailmentRisk}\n\n` +
            `💡 대응 가이드:\n` +
            (renewableRatio > 30 ?
              `• ESS 충전을 권장합니다\n• 출력제어 대비 발전량 조정 필요` :
              renewableRatio > 20 ?
                `• 상황 모니터링 권장\n• ESS 충전 준비` :
                `• 정상 운영 가능\n• 최대 출력 발전 권장`);
          break;
        }

        // ========== 발전소 상태 ==========
        case 'show_realtime_status': {
          const kpis = await apiService.getDashboardKPIs();
          const published = await apiService.getCurrentSMP().catch(() => null);
          response = `⚡ 실시간 발전 현황\n\n` +
            `🏭 총 설비용량: ${kpis.total_capacity_mw.toFixed(0)}MW\n` +
            `⚡ 현재 출력: ${kpis.current_output_mw.toFixed(1)}MW\n` +
            `📊 가동률: ${kpis.utilization_pct.toFixed(1)}%\n\n` +
            `💰 금일 수익: ${formatNullableNumber(kpis.daily_revenue_million)}백만원\n` +
            `💹 ${describeCurrentSmp(published)}: ${formatNullableNumber(publishedSmpPrice(published), 0)}원/kWh`;
          break;
        }

        case 'show_plant_list': {
          const plants = await apiService.getPowerPlants();
          if (plants.length === 0) {
            response = `📍 등록된 발전소 없음\n\n` +
              `아직 등록된 발전소가 없습니다.\n\n` +
              `💡 발전소를 등록하시려면 메인 화면의 '발전소 등록' 메뉴를 이용해주세요.`;
          } else {
            response = `📍 내 발전소 목록 (${plants.length}개)\n\n` +
              plants.map((p, i) =>
                `${i + 1}. ${p.name}\n` +
                `   유형: ${p.type === 'solar' ? '태양광' : p.type === 'wind' ? '풍력' : 'ESS'}\n` +
                `   용량: ${p.capacity}kW\n` +
                `   상태: ${p.status === 'active' ? '운영중' : p.status === 'maintenance' ? '점검중' : '일시정지'}`
              ).join('\n\n');
          }
          break;
        }

        case 'show_equipment_alerts': {
          // Mock equipment alerts (실제로는 모니터링 시스템 연동 필요)
          response = `🔔 설비 알림\n\n` +
            `✅ 모든 설비 정상 작동 중\n\n` +
            `최근 24시간 알림 없음\n\n` +
            `💡 인버터 효율: 98.5%\n` +
            `💡 접속반 상태: 정상\n` +
            `💡 계량기 통신: 정상`;
          break;
        }

        case 'show_ess_status': {
          // Mock ESS status (실제로는 ESS 모니터링 연동 필요)
          response = `🔋 ESS 상태 정보\n\n` +
            `📊 충전 상태(SOC): 75%\n` +
            `❤️ 배터리 건강(SOH): 96%\n` +
            `🌡️ 셀 온도: 28°C (정상)\n\n` +
            `⚡ 충/방전 현황:\n` +
            `   오늘 충전량: 150kWh\n` +
            `   오늘 방전량: 120kWh\n\n` +
            `💡 권장: 피크 시간대(14-17시) 방전 예정`;
          break;
        }

        // ========== 시장 리포트 ==========
        case 'show_smp_price': {
          const currentSMP = await apiService.getCurrentSMP();
          response = `💹 SMP 시세 정보\n\n` +
            `📍 ${describeCurrentSmp(currentSMP)}: ${formatNullableNumber(publishedSmpPrice(currentSMP), 0)}원/kWh`;
          break;
        }

        case 'show_weather_info': {
          // Alan API에 날씨 질문
          const weatherResponse = await alanApi.sendMessage('제주도 발전소 주변 기상 정보 알려줘');
          response = `🌤️ 기상 정보\n\n${weatherResponse.answer}`;
          break;
        }

        case 'show_market_news': {
          // Alan API에 시장 뉴스 질문
          const newsResponse = await alanApi.sendMessage('최근 전력시장 뉴스나 KPX 제도 변경 사항 알려줘');
          response = `📰 시장 뉴스\n\n${newsResponse.answer}`;
          break;
        }

        case 'show_faq': {
          response = `❓ 자주 묻는 질문\n\n` +
            `Q1. 입찰 마감 시간은?\n` +
            `A1. DAM(하루전시장)은 전일 10시까지, RTM(실시간시장)은 1시간 전까지입니다.\n\n` +
            `Q2. 정산 주기는?\n` +
            `A2. 월별 정산이며, 익월 15일경 확정됩니다.\n\n` +
            `Q3. 출력제어 보상은?\n` +
            `A3. 제주는 출력제어 시 SMP의 80% 수준으로 보상됩니다.\n\n` +
            `Q4. ESS 충전 최적 시간?\n` +
            `A4. SMP가 낮은 새벽(02-06시)에 충전, 높은 오후(14-17시)에 방전이 유리합니다.`;
          break;
        }

        default:
          response = '해당 기능을 준비 중입니다.';
      }

      addAssistantMessage(response);
    } catch (error) {
      console.error('[Alan] Action error:', error);
      addAssistantMessage('죄송합니다. 정보를 가져오는 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
      // Reset navigation state
      setCurrentLevel('main');
      setSelectedMainKeyword(null);
    }
  };

  // Handle send message
  const handleSendMessage = async (text?: string) => {
    const messageText = text || inputText.trim();
    if (!messageText) return;

    setShowChat(true);
    setInputText('');

    // Add user message (use unique suffix to avoid duplicate keys)
    const userMessage: AlanMessage = {
      id: `${Date.now()}_user`,
      role: 'user',
      content: messageText,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);

    setIsLoading(true);

    try {
      // Call Alan API
      const response = await alanApi.sendMessage(messageText);

      // Add assistant message (use unique suffix to avoid duplicate keys)
      const assistantMessage: AlanMessage = {
        id: `${Date.now()}_assistant`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        action: response.action,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      // Handle navigation action
      if (response.action?.type === 'navigate' && onNavigate) {
        setTimeout(() => {
          onNavigate(response.action!.target || '');
        }, 1500);
      }
    } catch (error) {
      console.error('[Alan] Error:', error);
      const errorMessage: AlanMessage = {
        id: `${Date.now()}_error`,
        role: 'assistant',
        content: '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해 주세요.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto scroll to bottom
  useEffect(() => {
    if (scrollViewRef.current && messages.length > 0) {
      setTimeout(() => {
        scrollViewRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);

  // Reset to main view
  const handleReset = () => {
    setShowChat(false);
    setMessages([]);
    alanApi.resetConversation();
  };

  // Render chat view
  if (showChat && messages.length > 0) {
    return (
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.chatHeader}>
          <TouchableOpacity onPress={handleReset} style={styles.backBtn}>
            <Text style={styles.backBtnText}>←</Text>
          </TouchableOpacity>
          <Text style={styles.chatHeaderTitle}>앨런 AI</Text>
          <View style={{ width: 40 }} />
        </View>

        <ScrollView
          ref={scrollViewRef}
          style={styles.chatMessages}
          contentContainerStyle={styles.chatMessagesContent}
          showsVerticalScrollIndicator={false}
        >
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          {isLoading && (
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>앨런이 생각중...</Text>
            </View>
          )}
        </ScrollView>

        <View style={styles.chatInputContainer}>
          <View style={styles.chatInputWrapper}>
            <TextInput
              style={styles.chatInput}
              placeholder="메시지를 입력하세요..."
              placeholderTextColor={colors.textMuted}
              value={inputText}
              onChangeText={setInputText}
              onSubmitEditing={() => handleSendMessage()}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={[styles.sendBtn, !inputText.trim() && styles.sendBtnDisabled]}
              onPress={() => handleSendMessage()}
              disabled={!inputText.trim()}
            >
              <Text style={styles.sendBtnText}>↑</Text>
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>
    );
  }

  // Render main Alan screen (Figma design)
  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
    >
      {/* Gradient Background */}
      <View style={styles.gradientBg}>
        <ScrollView style={styles.mainContent} keyboardShouldPersistTaps="handled">
        {/* Alan Robot */}
        <View style={styles.robotContainer}>
          <AlanRobot />
        </View>

        {/* Hierarchical Keyword Navigation */}
        <View style={styles.quickActionsContainer}>
          {currentLevel === 'main' ? (
            // Main Keywords (4 categories)
            <>
              <Text style={styles.keywordTitle}>앨런에게 물어보세요</Text>
              <View style={styles.quickActionsRow}>
                {keywordTree.slice(0, 2).map((kw) => (
                  <TouchableOpacity
                    key={kw.id}
                    style={styles.keywordBtn}
                    onPress={() => handleKeywordClick(kw)}
                  >
                    <Text style={styles.keywordIcon}>{kw.icon}</Text>
                    <Text style={styles.keywordLabel}>{kw.label}</Text>
                  </TouchableOpacity>
                ))}
              </View>
              <View style={styles.quickActionsRow}>
                {keywordTree.slice(2, 4).map((kw) => (
                  <TouchableOpacity
                    key={kw.id}
                    style={styles.keywordBtn}
                    onPress={() => handleKeywordClick(kw)}
                  >
                    <Text style={styles.keywordIcon}>{kw.icon}</Text>
                    <Text style={styles.keywordLabel}>{kw.label}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </>
          ) : (
            // Sub Keywords (4 items under selected main)
            <>
              <TouchableOpacity
                style={styles.backButton}
                onPress={handleBackToMain}
              >
                <Text style={styles.backButtonText}>
                  ← {selectedMainKeyword?.icon} {selectedMainKeyword?.label}
                </Text>
              </TouchableOpacity>
              <View style={styles.quickActionsRow}>
                {selectedMainKeyword?.children?.slice(0, 2).map((kw) => (
                  <TouchableOpacity
                    key={kw.id}
                    style={styles.keywordBtn}
                    onPress={() => handleKeywordClick(kw)}
                  >
                    <Text style={styles.keywordIcon}>{kw.icon}</Text>
                    <Text style={styles.keywordLabel}>{kw.label}</Text>
                  </TouchableOpacity>
                ))}
              </View>
              <View style={styles.quickActionsRow}>
                {selectedMainKeyword?.children?.slice(2, 4).map((kw) => (
                  <TouchableOpacity
                    key={kw.id}
                    style={styles.keywordBtn}
                    onPress={() => handleKeywordClick(kw)}
                  >
                    <Text style={styles.keywordIcon}>{kw.icon}</Text>
                    <Text style={styles.keywordLabel}>{kw.label}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            </>
          )}
        </View>

        </ScrollView>

        {/* Chat Input Area */}
        <View style={styles.inputArea}>
          <View style={styles.inputWrapper}>
            <TextInput
              style={styles.input}
              placeholder="무엇이든 물어보세요"
              placeholderTextColor={colors.textMuted}
              value={inputText}
              onChangeText={setInputText}
              onSubmitEditing={() => handleSendMessage()}
              returnKeyType="send"
            />
            <TouchableOpacity
              style={styles.voiceBtn}
              onPress={() => handleSendMessage()}
            >
              <Text style={styles.voiceBtnIcon}>🎤</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  gradientBg: {
    flex: 1,
    backgroundColor: colors.background,
  },
  mainContent: {
    flex: 1,
  },

  // Robot Container
  robotContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 30,
    paddingBottom: 20,
  },
  alanRobot: {
    alignItems: 'center',
  },
  // Cat Ears (pointing up like in Figma)
  earsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    width: 130,
    marginBottom: -15,
    zIndex: 1,
  },
  catEarLeft: {
    width: 0,
    height: 0,
    borderLeftWidth: 18,
    borderRightWidth: 18,
    borderBottomWidth: 35,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: colors.white,
    marginRight: 40,
    transform: [{ rotate: '-15deg' }],
  },
  catEarInnerLeft: {
    position: 'absolute',
    top: 12,
    left: -8,
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderBottomWidth: 16,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: themeColors.border.primary,
  },
  catEarRight: {
    width: 0,
    height: 0,
    borderLeftWidth: 18,
    borderRightWidth: 18,
    borderBottomWidth: 35,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: colors.white,
    marginLeft: 40,
    transform: [{ rotate: '15deg' }],
  },
  catEarInnerRight: {
    position: 'absolute',
    top: 12,
    left: -8,
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderBottomWidth: 16,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    borderBottomColor: themeColors.border.primary,
  },
  robotBody: {
    width: 130,
    height: 100,
    backgroundColor: colors.white,
    borderTopLeftRadius: 65,
    borderTopRightRadius: 65,
    borderBottomLeftRadius: 40,
    borderBottomRightRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 12,
    elevation: 5,
  },
  robotFaceArea: {
    width: 90,
    height: 60,
    backgroundColor: colors.robotFace,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 5,
  },
  robotEyes: {
    flexDirection: 'row',
    gap: 16,
    marginTop: -2,
  },
  robotEye: {
    width: 20,
    height: 20,
    backgroundColor: colors.robotEyes,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  robotPupil: {
    width: 6,
    height: 6,
    backgroundColor: themeColors.surface,
    borderRadius: 3,
    marginTop: -3,
    marginLeft: 3,
  },
  robotMouth: {
    marginTop: 4,
  },
  robotMouthText: {
    fontSize: 14,
    color: colors.robotEyes,
    fontWeight: '400',
  },
  robotBottom: {
    width: 90,
    height: 35,
    backgroundColor: colors.white,
    borderBottomLeftRadius: 45,
    borderBottomRightRadius: 45,
    marginTop: -8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },

  // Quick Actions / Keyword Navigation
  quickActionsContainer: {
    paddingHorizontal: 24,
    marginTop: 30,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 12,
    marginBottom: 12,
  },
  quickActionBtn: {
    backgroundColor: colors.white,
    paddingVertical: 14,
    paddingHorizontal: 24,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  quickActionLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.text,
  },

  // Hierarchical Keyword Styles
  keywordTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: 16,
  },
  keywordBtn: {
    backgroundColor: colors.white,
    paddingVertical: 18,
    paddingHorizontal: 20,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: colors.border,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
    minWidth: (SCREEN_WIDTH - 72) / 2,
    alignItems: 'center',
  },
  keywordIcon: {
    fontSize: 28,
    marginBottom: 6,
  },
  keywordLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text,
    textAlign: 'center',
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 16,
    marginBottom: 12,
    backgroundColor: colors.white,
    borderRadius: 20,
    alignSelf: 'flex-start',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  backButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.buttonBlue,
  },

  // Spacer
  spacer: {
    flex: 1,
  },

  // Input Area
  inputArea: {
    paddingHorizontal: 20,
    paddingBottom: Platform.OS === 'ios' ? 30 : 20,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.white,
    borderRadius: 28,
    paddingLeft: 20,
    paddingRight: 6,
    paddingVertical: 6,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 3,
  },
  input: {
    flex: 1,
    fontSize: 15,
    color: colors.text,
    paddingVertical: 8,
  },
  voiceBtn: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.buttonBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  voiceBtnIcon: {
    fontSize: 20,
  },

  // Chat View Styles
  chatHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: colors.white,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  backBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  backBtnText: {
    fontSize: 24,
    color: colors.text,
  },
  chatHeaderTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text,
  },
  chatMessages: {
    flex: 1,
    backgroundColor: colors.background,
  },
  chatMessagesContent: {
    padding: 16,
  },
  messageContainer: {
    marginBottom: 12,
  },
  userMessage: {
    alignItems: 'flex-end',
  },
  assistantMessage: {
    alignItems: 'flex-start',
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 14,
    borderRadius: 18,
  },
  userBubble: {
    backgroundColor: colors.buttonBlue,
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: colors.white,
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 15,
    color: colors.text,
    lineHeight: 22,
  },
  userMessageText: {
    color: colors.white,
  },
  loadingContainer: {
    alignItems: 'flex-start',
    marginTop: 8,
  },
  loadingText: {
    fontSize: 14,
    color: colors.textMuted,
    fontStyle: 'italic',
  },
  chatInputContainer: {
    padding: 12,
    backgroundColor: colors.white,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  chatInputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.inputBg,
    borderRadius: 24,
    paddingLeft: 16,
    paddingRight: 4,
  },
  chatInput: {
    flex: 1,
    fontSize: 15,
    color: colors.text,
    paddingVertical: 12,
  },
  sendBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.buttonBlue,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendBtnDisabled: {
    backgroundColor: colors.textMuted,
  },
  sendBtnText: {
    fontSize: 18,
    color: colors.white,
    fontWeight: 'bold',
  },
});
