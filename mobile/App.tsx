/**
 * RE-BMS Mobile App v6.1.0 - Cross-Platform Edition
 * 100% Feature Parity with Web-v6.1.0
 * Supports iOS, Android, and Web
 *
 * Tabs (3-tab navigation):
 * - SMP Forecast (24h Prediction)
 * - Bidding (DAM/RTM Management)
 * - Settlement (Revenue Analysis)
 */

import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import {
  View,
  Text,
  StyleSheet,
  Platform,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';

// Use emoji icons for cross-platform compatibility
// Ionicons has compatibility issues with Expo 50 on iOS
const USE_EMOJI_ICONS = true;

// Screens (3-tab: SMP, Bidding, Settlement)
import SMPForecastScreen from './src/screens/SMPForecastScreen';
import BiddingScreen from './src/screens/BiddingScreen';
import SettlementScreen from './src/screens/SettlementScreen';
import KPXSimulationScreen from './src/screens/KPXSimulationScreen';
import RTMSimulationScreen from './src/screens/RTMSimulationScreen';
import AlanChatScreen from './src/screens/alan/AlanChatScreen';

// Design colors (matching web-v6.1.0)
const colors = {
  primary: '#04265e',
  secondary: '#0048ff',
  background: '#ffffff',
  cardBg: '#f8f8f8',
  text: '#000000',
  textSecondary: '#666666',
  textMuted: '#999999',
  border: '#e0e0e0',
  tabActive: '#0048ff',
  tabInactive: '#999999',
};

// Tab Icon using emojis for cross-platform compatibility
const TabIcon = ({ name, focused }: { name: string; focused: boolean }) => {
  const iconMap: { [key: string]: string } = {
    'bar-chart': '📊',
    'hammer': '⚖️',
    'wallet': '💰',
  };

  return (
    <Text style={{
      fontSize: 22,
      opacity: focused ? 1 : 0.6
    }}>
      {iconMap[name] || '•'}
    </Text>
  );
};

// Tab configuration (3 tabs only - matching web-v6.1.0)
type TabKey = 'smp' | 'bidding' | 'settlement';

interface TabConfig {
  key: TabKey;
  label: string;
  icon: string;
}

const tabs: TabConfig[] = [
  { key: 'smp', label: 'SMP예측', icon: 'bar-chart' },
  { key: 'bidding', label: '입찰관리', icon: 'hammer' },
  { key: 'settlement', label: '정산', icon: 'wallet' },
];

// Header Component
function Header({ title, showBack, onBack }: { title?: string; showBack?: boolean; onBack?: () => void }) {
  return (
    <View style={styles.header}>
      <View style={styles.headerLeft}>
        {showBack ? (
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <Text style={{ fontSize: 20 }}>←</Text>
          </TouchableOpacity>
        ) : (
          <Text style={styles.logoText}>eXeco</Text>
        )}
        {title && <Text style={styles.headerTitle}>{title}</Text>}
      </View>
      <View style={styles.headerRight}>
        <View style={styles.liveIndicator}>
          <View style={styles.liveDot} />
          <Text style={styles.liveText}>Live</Text>
        </View>
        <Text style={styles.versionText}>v6.1</Text>
      </View>
    </View>
  );
}

// Bottom Tab Bar
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
            activeOpacity={0.7}
          >
            <TabIcon name={tab.icon} focused={isActive} />
            <Text style={[styles.tabLabel, isActive && styles.tabLabelActive]}>
              {tab.label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
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
          <Text style={styles.errorTitle}>Something went wrong</Text>
          <Text style={styles.errorText}>{this.state.error}</Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => this.setState({ hasError: false, error: '' })}
          >
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        </View>
      );
    }
    return this.props.children;
  }
}

// Main App
export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('smp');
  const [simulationScreen, setSimulationScreen] = useState<'none' | 'dam' | 'rtm'>('none');
  const [showAlanChat, setShowAlanChat] = useState(false);

  // Navigation handler for simulation screens
  const handleNavigate = (screen: string) => {
    if (screen === 'KPXSimulation' || screen === 'DAM') {
      setSimulationScreen('dam');
    } else if (screen === 'RTMSimulation' || screen === 'RTM') {
      setSimulationScreen('rtm');
    } else if (screen.toLowerCase() === 'smp') {
      setActiveTab('smp');
    } else if (screen.toLowerCase() === 'bidding') {
      setActiveTab('bidding');
    } else if (screen.toLowerCase() === 'settlement') {
      setActiveTab('settlement');
    }
    setShowAlanChat(false);
  };

  const handleBack = () => {
    setSimulationScreen('none');
  };

  // Web navigation object for screens that need it
  const webNavigation = {
    navigate: handleNavigate,
    goBack: handleBack,
  };

  // Render simulation screens
  if (simulationScreen === 'dam') {
    return (
      <ErrorBoundary>
        <SafeAreaView style={styles.container}>
          <StatusBar style="dark" />
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
          <StatusBar style="dark" />
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
    if (showAlanChat) {
      return (
        <AlanChatScreen
          onNavigate={(screen) => {
            setShowAlanChat(false);
            handleNavigate(screen);
          }}
        />
      );
    }

    switch (activeTab) {
      case 'smp':
        return <SMPForecastScreen />;
      case 'bidding':
        return <BiddingScreen webNavigation={webNavigation} />;
      case 'settlement':
        return <SettlementScreen />;
      default:
        return <SMPForecastScreen />;
    }
  };

  return (
    <ErrorBoundary>
      <SafeAreaView style={styles.container}>
        <StatusBar style="dark" />
        <Header />
        <View style={styles.content}>{renderScreen()}</View>
        <BottomTabBar
          activeTab={activeTab}
          onTabChange={(tab) => {
            setShowAlanChat(false);
            setActiveTab(tab);
          }}
        />

        {/* Floating Alan AI Button */}
        {!showAlanChat && (
          <TouchableOpacity
            style={styles.floatingAlanBtn}
            onPress={() => setShowAlanChat(true)}
            activeOpacity={0.8}
          >
            <View style={styles.alanAvatar}>
              <Text style={styles.alanEmoji}>🤖</Text>
            </View>
          </TouchableOpacity>
        )}
      </SafeAreaView>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    flex: 1,
  },

  // Header
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
  versionText: {
    fontSize: 12,
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
    fontSize: 10,
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
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: colors.secondary,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  retryText: {
    color: '#fff',
    fontWeight: '600',
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
  alanAvatar: {
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
