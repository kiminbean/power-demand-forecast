/**
 * RE-BMS Mobile App v6.0.0 - Alan AI Edition
 * EastSoft Alan AI Chatbot Integration
 * Design: Figma alan_mobile (100% matching)
 *
 * Screens:
 * - Page 1: SMP Forecast (Tab: SMP예측)
 * - Page 2: Alan Chat (Floating button access)
 * - Page 3: Bidding Management (Tab: 입찰관리)
 * - Page 4: Settlement (Tab: 정산)
 */

import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import {
  View,
  Text,
  StyleSheet,
  Platform,
  TouchableOpacity,
  SafeAreaView,
  Image,
  Dimensions,
} from 'react-native';

// Custom SVG-like icons for web (matching Figma exactly)
const TabIcon = ({ name, size, color }: { name: string; size: number; color: string }) => {
  // Bar chart icon for SMP 예측
  if (name === 'bar-chart' || name === 'bar-chart-outline') {
    return (
      <View style={{ width: size, height: size, justifyContent: 'flex-end', alignItems: 'center', flexDirection: 'row', gap: 2 }}>
        <View style={{ width: 5, height: size * 0.5, backgroundColor: color, borderRadius: 1 }} />
        <View style={{ width: 5, height: size * 0.75, backgroundColor: color, borderRadius: 1 }} />
        <View style={{ width: 5, height: size * 0.6, backgroundColor: color, borderRadius: 1 }} />
      </View>
    );
  }
  // Gavel/Auction icon for 입찰관리
  if (name === 'hammer' || name === 'hammer-outline') {
    return (
      <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
        <View style={{ width: size * 0.7, height: size * 0.25, backgroundColor: color, borderRadius: 2, transform: [{ rotate: '-45deg' }], marginBottom: -4 }} />
        <View style={{ width: size * 0.15, height: size * 0.5, backgroundColor: color, borderRadius: 2, marginTop: -2 }} />
      </View>
    );
  }
  // Box icon for 정산
  if (name === 'cube' || name === 'cube-outline') {
    return (
      <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
        <View style={{
          width: size * 0.75,
          height: size * 0.6,
          borderWidth: 2,
          borderColor: color,
          borderRadius: 2,
          marginTop: size * 0.15,
        }}>
          <View style={{ position: 'absolute', top: -size * 0.15, left: -2, right: -2, height: size * 0.2, borderWidth: 2, borderColor: color, borderRadius: 2, backgroundColor: 'transparent' }} />
        </View>
      </View>
    );
  }
  // Fallback
  return <Text style={{ fontSize: size * 0.8, color }}>●</Text>;
};

// Conditionally import Ionicons for better web compatibility
let Ionicons: any = TabIcon;

try {
  const VectorIcons = require('@expo/vector-icons');
  if (VectorIcons && VectorIcons.Ionicons) {
    Ionicons = VectorIcons.Ionicons;
  }
} catch (e) {
  console.log('Ionicons not available, using custom icons');
}

// Screens
import AlanChatScreen from './src/screens/alan/AlanChatScreen';
import SMPForecastScreen from './src/screens/SMPForecastScreen';
import BiddingScreen from './src/screens/BiddingScreen';
import SettlementScreen from './src/screens/SettlementScreen';
import KPXSimulationScreen from './src/screens/KPXSimulationScreen';
import RTMSimulationScreen from './src/screens/RTMSimulationScreen';

// Alan API
import { alanApi } from './src/services/alan/alanApi';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Design colors from Figma
const colors = {
  primary: '#04265e',      // Header blue (eXeco brand)
  secondary: '#0048ff',    // Accent blue
  background: '#ffffff',
  cardBg: '#f8f8f8',
  text: '#000000',
  textSecondary: '#666666',
  textMuted: '#999999',
  border: '#e0e0e0',
  tabActive: '#0048ff',
  tabInactive: '#999999',
};

// Tab configuration (from Figma design)
type TabKey = 'smp' | 'bidding' | 'settlement';

interface TabConfig {
  key: TabKey;
  label: string;
  icon: string;
  iconActive: string;
}

const tabs: TabConfig[] = [
  { key: 'smp', label: 'SMP 예측', icon: 'bar-chart-outline', iconActive: 'bar-chart' },
  { key: 'bidding', label: '입찰관리', icon: 'hammer-outline', iconActive: 'hammer' },
  { key: 'settlement', label: '정산', icon: 'cube-outline', iconActive: 'cube' },
];

// Header Component (Figma style - white background)
function Header({ title, showBack, onBack }: { title?: string; showBack?: boolean; onBack?: () => void }) {
  return (
    <View style={styles.header}>
      <View style={styles.headerLeft}>
        {showBack ? (
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <Ionicons name="chevron-back" size={24} color={colors.text} />
          </TouchableOpacity>
        ) : (
          <Text style={styles.logoText}>eXeco</Text>
        )}
        {title && <Text style={styles.headerTitle}>{title}</Text>}
      </View>
      <View style={styles.headerRight}>
        {/* Live indicator */}
        <View style={styles.liveIndicator}>
          <View style={styles.liveDot} />
          <Text style={styles.liveText}>Live</Text>
        </View>
        {/* Moon/Dark mode icon */}
        <TouchableOpacity style={styles.headerIconBtn}>
          <Text style={styles.headerIcon}>☽</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

// Bottom Tab Bar (Figma design)
function BottomTabBar({
  activeTab,
  onTabChange,
}: {
  activeTab: TabKey;
  onTabChange: (tab: TabKey) => void;
}) {
  return (
    <View style={styles.tabBar}>
      {tabs.map((tab) => {
        const isActive = activeTab === tab.key;
        return (
          <TouchableOpacity
            key={tab.key}
            style={styles.tabItem}
            onPress={() => onTabChange(tab.key)}
          >
            <Ionicons
              name={isActive ? tab.iconActive as any : tab.icon as any}
              size={24}
              color={isActive ? colors.tabActive : colors.tabInactive}
            />
            <Text style={[styles.tabLabel, isActive && styles.tabLabelActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

// Simulation data type
interface SimulationData {
  segments: { id: number; quantity: number; price: number }[];
  selectedHour: number;
  smpForecast: { q10: number; q50: number; q90: number };
}

// Error Boundary
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: '' };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorContainer}>
          <Ionicons name="warning" size={48} color={colors.secondary} />
          <Text style={styles.errorTitle}>Something went wrong</Text>
          <Text style={styles.errorText}>{this.state.error}</Text>
        </View>
      );
    }
    return this.props.children;
  }
}

// Main Web App with Alan Integration
function WebApp() {
  const [activeTab, setActiveTab] = useState<TabKey>('smp');
  const [simulationScreen, setSimulationScreen] = useState<'none' | 'dam' | 'rtm'>('none');
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null);
  const [alanConfigured, setAlanConfigured] = useState(false);
  const [showAlanChat, setShowAlanChat] = useState(false);

  // Initialize Alan API (placeholder - will be configured with actual credentials)
  useEffect(() => {
    // Check if Alan is configured
    // In production, load credentials from secure storage
    const initAlan = async () => {
      try {
        // Placeholder: will be replaced with actual API key
        // alanApi.initialize({
        //   clientId: 'YOUR_CLIENT_ID',
        //   apiKey: 'YOUR_API_KEY',
        // });
        setAlanConfigured(false); // Set to true when configured
      } catch (error) {
        console.log('Alan not configured, using mock responses');
      }
    };
    initAlan();
  }, []);

  // Navigation handler
  const handleNavigate = (screen: string, params?: any) => {
    if (screen === 'KPXSimulation' || screen === 'DAM') {
      setSimulationData(params);
      setSimulationScreen('dam');
    } else if (screen === 'RTMSimulation' || screen === 'RTM') {
      setSimulationData(params);
      setSimulationScreen('rtm');
    } else if (screen === 'SMP' || screen === 'smp') {
      setActiveTab('smp');
    } else if (screen === 'Bidding' || screen === 'bidding') {
      setActiveTab('bidding');
    } else if (screen === 'Settlement' || screen === 'settlement') {
      setActiveTab('settlement');
    }
  };

  const handleBack = () => {
    setSimulationScreen('none');
    setSimulationData(null);
  };

  // Web navigation handlers for simulation screens
  const webNavigation = {
    navigate: handleNavigate,
    goBack: handleBack,
  };

  // Render simulation screens
  if (simulationScreen === 'dam') {
    return (
      <ErrorBoundary>
        <SafeAreaView style={styles.container}>
          <Header title="DAM 시뮬레이션" showBack onBack={handleBack} />
          <View style={styles.content}>
            <KPXSimulationScreen />
          </View>
        </SafeAreaView>
      </ErrorBoundary>
    );
  }

  if (simulationScreen === 'rtm') {
    return (
      <ErrorBoundary>
        <SafeAreaView style={styles.container}>
          <Header title="RTM 시뮬레이션" showBack onBack={handleBack} />
          <View style={styles.content}>
            <RTMSimulationScreen />
          </View>
        </SafeAreaView>
      </ErrorBoundary>
    );
  }

  // Render main screen based on active tab
  const renderScreen = () => {
    switch (activeTab) {
      case 'smp':
        // SMP Forecast screen (Page 1 from Figma)
        return <SMPForecastScreen />;
      case 'bidding':
        // Bidding Management screen (Page 3 from Figma)
        return <BiddingScreen webNavigation={webNavigation} />;
      case 'settlement':
        // Settlement screen (Page 4 from Figma)
        return <SettlementScreen />;
      default:
        return <SMPForecastScreen />;
    }
  };

  // Render main content (including Alan Chat as overlay within same layout)
  const renderContent = () => {
    if (showAlanChat) {
      return (
        <AlanChatScreen onNavigate={(screen) => {
          setShowAlanChat(false);
          handleNavigate(screen);
        }} />
      );
    }
    return renderScreen();
  };

  return (
    <ErrorBoundary>
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        <Header />
        <View style={styles.content}>{renderContent()}</View>
        <BottomTabBar activeTab={activeTab} onTabChange={(tab) => {
          setShowAlanChat(false);
          setActiveTab(tab);
        }} />

        {/* Floating Alan Button */}
        {!showAlanChat && (
          <TouchableOpacity
            style={styles.floatingAlanBtn}
            onPress={() => setShowAlanChat(true)}
          >
            <View style={styles.alanAvatarSmall}>
              <Text style={styles.alanEmoji}>🤖</Text>
            </View>
          </TouchableOpacity>
        )}
      </SafeAreaView>
    </ErrorBoundary>
  );
}

// Native App (full React Navigation)
function NativeApp() {
  const GestureHandlerRootView = require('react-native-gesture-handler').GestureHandlerRootView;
  const AppNavigator = require('./src/navigation/AppNavigator').default;

  return (
    <GestureHandlerRootView style={styles.nativeContainer}>
      <StatusBar style="light" />
      <AppNavigator />
    </GestureHandlerRootView>
  );
}

// Main App Export
export default function App() {
  if (Platform.OS === 'web') {
    return <WebApp />;
  }
  return <NativeApp />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  nativeContainer: {
    flex: 1,
  },
  content: {
    flex: 1,
  },

  // Header (Figma style - white background)
  header: {
    backgroundColor: colors.background,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  logoText: {
    color: colors.primary,
    fontSize: 24,
    fontWeight: 'bold',
    fontStyle: 'italic',
  },
  headerTitle: {
    color: colors.text,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 12,
  },
  backButton: {
    padding: 4,
    marginRight: 8,
  },
  headerIconBtn: {
    padding: 4,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#4caf50',
  },
  liveText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#4caf50',
  },
  headerIcon: {
    fontSize: 20,
    color: colors.textMuted,
  },

  // Tab Bar
  tabBar: {
    flexDirection: 'row',
    backgroundColor: colors.background,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingBottom: Platform.OS === 'ios' ? 20 : 8,
    paddingTop: 8,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
  },
  tabLabel: {
    fontSize: 11,
    color: colors.tabInactive,
    marginTop: 4,
    fontWeight: '500',
  },
  tabLabelActive: {
    color: colors.tabActive,
    fontWeight: '600',
  },

  // Error
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.background,
    padding: 24,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text,
    marginTop: 16,
  },
  errorText: {
    fontSize: 14,
    color: colors.textSecondary,
    marginTop: 8,
    textAlign: 'center',
  },

  // Alan Chat Header
  alanChatHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: colors.background,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  alanBackBtn: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  alanChatTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.text,
  },

  // Floating Alan Button
  floatingAlanBtn: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 100 : 80,
    right: 20,
    zIndex: 100,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 5,
  },
  alanAvatarSmall: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#f0f4f8',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: colors.secondary,
  },
  alanEmoji: {
    fontSize: 28,
  },
});
