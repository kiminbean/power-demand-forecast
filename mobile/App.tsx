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
import { Ionicons } from './src/components/Ionicon';
import {
  View,
  Text,
  StyleSheet,
  Platform,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';

// eXeco DS v1 tokens (dark command-center surfaces, emerald accent)
import { colors } from './src/theme/colors';
import SMPForecastScreen from './src/screens/SMPForecastScreen';
import BiddingScreen from './src/screens/BiddingScreen';
import SettlementScreen from './src/screens/SettlementScreen';
import KPXSimulationScreen from './src/screens/KPXSimulationScreen';
import RTMSimulationScreen from './src/screens/RTMSimulationScreen';
import AlanChatScreen from './src/screens/alan/AlanChatScreen';

// Tab Icon using vector icons (Ionicons, eXeco DS v1: focused emerald / unfocused mute)
const TabIcon = ({ name, focused }: { name: string; focused: boolean }) => {
  const iconMap: { [key: string]: string } = {
    'bar-chart': 'trending-up',
    'hammer': 'hammer',
    'wallet': 'wallet',
  };

  const ionName = (iconMap[name] || 'ellipse') as keyof typeof Ionicons.glyphMap;
  return (
    <Ionicons
      name={focused ? ionName : `${ionName}-outline` as keyof typeof Ionicons.glyphMap}
      size={22}
      color={focused ? '#2563EB' : '#94A3B8'}
    />
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
            <Ionicons name="arrow-back" size={22} color={colors.text.primary} />
          </TouchableOpacity>
        ) : (
          <Text style={styles.logoText}>eXeco</Text>
        )}
        {title && <Text style={styles.headerTitle}>{title}</Text>}
      </View>
      <View style={styles.headerRight}>
        <Text style={styles.versionText}>v6.1</Text>
      </View>
    </View>
  );
}

// Bottom Tab Bar
function BottomTabBar({
  activeTab,
  onTabChange,
  onOpenAlan,
}: {
  activeTab: TabKey | null;
  onTabChange: (tab: TabKey) => void;
  onOpenAlan?: () => void;
}) {
  return (
    <View style={styles.tabBar}>
      <View style={styles.tabsRow} accessibilityRole="tablist">
      {tabs.map((tab) => {
        const isActive = activeTab === tab.key;
        return (
          <TouchableOpacity
            key={tab.key}
            style={styles.tabItem}
            accessibilityRole="tab"
            accessibilityLabel={tab.label}
            accessibilityState={{ selected: isActive }}
            aria-selected={isActive}
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
      {onOpenAlan && (
        <TouchableOpacity
          style={styles.alanNavButton}
          accessibilityRole="button"
          accessibilityLabel="Alan AI 채팅 열기"
          onPress={onOpenAlan}
          activeOpacity={0.8}
        >
          <View style={styles.alanAvatar}>
            <Ionicons name="sparkles" size={24} color={colors.brand.primary} />
          </View>
        </TouchableOpacity>
      )}
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
          activeTab={showAlanChat ? null : activeTab}
          onTabChange={(tab) => {
            setShowAlanChat(false);
            setActiveTab(tab);
          }}
          onOpenAlan={!showAlanChat ? () => setShowAlanChat(true) : undefined}
        />
      </SafeAreaView>
    </ErrorBoundary>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  content: {
    flex: 1,
  },

  // Header
  header: {
    backgroundColor: colors.surface,
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderBottomColor: colors.border.primary,
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
    color: colors.brand.navy,
    fontSize: 24,
    fontWeight: 'bold',
    fontStyle: 'italic',
  },
  headerTitle: {
    color: colors.text.primary,
    fontSize: 18,
    fontWeight: '600',
    marginLeft: 12,
  },
  backButton: {
    width: 44,
    height: 44,
    alignItems: 'center',
    justifyContent: 'center',
    paddingRight: 8,
  },

  versionText: {
    fontSize: 12,
    color: colors.text.muted,
  },

  // Tab Bar
  tabBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border.primary,
    paddingBottom: Platform.OS === 'ios' ? 20 : 8,
    paddingTop: 8,
  },
  tabsRow: {
    flex: 1,
    flexDirection: 'row',
  },
  alanNavButton: {
    paddingHorizontal: 8,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44,
    paddingVertical: 6,
  },
  tabLabel: {
    fontSize: 10,
    color: colors.text.secondary,
    marginTop: 4,
    fontWeight: '500',
  },
  tabLabelActive: {
    color: colors.brand.primary,
    fontWeight: '600',
  },

  // Error
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.bg,
    padding: 24,
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: 20,
  },
  retryButton: {
    backgroundColor: colors.info,
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 14,
  },
  retryText: {
    color: colors.text.primary,
    fontWeight: '600',
  },

  // Alan button
  alanAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.surfaceAlt,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: colors.brand.primary,
  },
});
