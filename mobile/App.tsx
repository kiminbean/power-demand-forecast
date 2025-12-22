/**
 * RE-BMS Mobile App v6.1.0
 * Renewable Energy Bidding Management System with KPX DAM/RTM Simulation
 * Matches web-v6.1.0 features
 */

import React, { useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { View, Text, StyleSheet, Platform, TouchableOpacity, SafeAreaView } from 'react-native';

// Import web-compatible screens
import DashboardScreen from './src/screens/DashboardScreen';
import BiddingScreen from './src/screens/BiddingScreen';
import PortfolioScreen from './src/screens/PortfolioScreen';
import SettlementScreen from './src/screens/SettlementScreen';
import KPXSimulationScreen from './src/screens/KPXSimulationScreen';
import RTMSimulationScreen from './src/screens/RTMSimulationScreen';

const colors = {
  background: '#0e1117',
  secondary: '#1a1f2c',
  primary: '#6366f1',
  text: '#ffffff',
  muted: '#718096',
  border: '#374151',
};

// Simple tab bar for web
function WebTabBar({ activeTab, onTabChange }: { activeTab: string; onTabChange: (tab: string) => void }) {
  const tabs = [
    { key: 'dashboard', label: 'Dashboard', icon: '📊' },
    { key: 'bidding', label: 'Bidding', icon: '📈' },
    { key: 'portfolio', label: 'Portfolio', icon: '📁' },
    { key: 'settlement', label: 'Settlement', icon: '💰' },
  ];

  return (
    <View style={webStyles.tabBar}>
      {tabs.map((tab) => (
        <TouchableOpacity
          key={tab.key}
          style={[webStyles.tab, activeTab === tab.key && webStyles.activeTab]}
          onPress={() => onTabChange(tab.key)}
        >
          <Text style={webStyles.tabIcon}>{tab.icon}</Text>
          <Text style={[webStyles.tabLabel, activeTab === tab.key && webStyles.activeTabLabel]}>
            {tab.label}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

// Error Boundary for web
class ErrorBoundary extends React.Component<{children: React.ReactNode}, {hasError: boolean, error: string}> {
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
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.background, padding: 20 }}>
          <Text style={{ color: '#ef4444', fontSize: 18, fontWeight: 'bold' }}>Something went wrong</Text>
          <Text style={{ color: colors.muted, marginTop: 10, textAlign: 'center' }}>{this.state.error}</Text>
        </View>
      );
    }
    return this.props.children;
  }
}

// Simple placeholder screen
function PlaceholderScreen({ title }: { title: string }) {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.background }}>
      <Text style={{ color: colors.primary, fontSize: 24, fontWeight: 'bold' }}>{title}</Text>
      <Text style={{ color: colors.muted, marginTop: 8 }}>Coming soon...</Text>
    </View>
  );
}

// Simulation data type
interface SimulationData {
  segments: { id: number; quantity: number; price: number }[];
  selectedHour: number;
  smpForecast: { q10: number; q50: number; q90: number };
}

// Web App with simple tab navigation + simulation support
function WebApp() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [simulationScreen, setSimulationScreen] = useState<'none' | 'dam' | 'rtm'>('none');
  const [simulationData, setSimulationData] = useState<SimulationData | null>(null);

  // Web navigation handlers for simulation screens
  const webNavigation = {
    navigate: (screen: string, params?: any) => {
      if (screen === 'KPXSimulation') {
        setSimulationData(params);
        setSimulationScreen('dam');
      } else if (screen === 'RTMSimulation') {
        setSimulationData(params);
        setSimulationScreen('rtm');
      }
    },
    goBack: () => {
      setSimulationScreen('none');
      setSimulationData(null);
    },
  };

  // Render simulation screens
  if (simulationScreen === 'dam' && simulationData) {
    return (
      <ErrorBoundary>
        <SafeAreaView style={webStyles.container}>
          <View style={webStyles.header}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <TouchableOpacity onPress={webNavigation.goBack} style={{ marginRight: 12 }}>
                <Text style={{ color: colors.primary, fontSize: 16 }}>← 뒤로</Text>
              </TouchableOpacity>
              <Text style={webStyles.headerTitle}>DAM 시뮬레이션</Text>
            </View>
          </View>
          <View style={webStyles.content}>
            <KPXSimulationScreen />
          </View>
        </SafeAreaView>
      </ErrorBoundary>
    );
  }

  if (simulationScreen === 'rtm' && simulationData) {
    return (
      <ErrorBoundary>
        <SafeAreaView style={webStyles.container}>
          <View style={webStyles.header}>
            <View style={{ flexDirection: 'row', alignItems: 'center' }}>
              <TouchableOpacity onPress={webNavigation.goBack} style={{ marginRight: 12 }}>
                <Text style={{ color: colors.primary, fontSize: 16 }}>← 뒤로</Text>
              </TouchableOpacity>
              <Text style={webStyles.headerTitle}>RTM 시뮬레이션</Text>
            </View>
          </View>
          <View style={webStyles.content}>
            <RTMSimulationScreen />
          </View>
        </SafeAreaView>
      </ErrorBoundary>
    );
  }

  const renderScreen = () => {
    try {
      switch (activeTab) {
        case 'dashboard':
          return <DashboardScreen />;
        case 'bidding':
          return <BiddingScreen webNavigation={webNavigation} />;
        case 'portfolio':
          return <PortfolioScreen />;
        case 'settlement':
          return <SettlementScreen />;
        default:
          return <DashboardScreen />;
      }
    } catch (e: any) {
      return (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: colors.background }}>
          <Text style={{ color: '#ef4444' }}>Error: {e.message}</Text>
        </View>
      );
    }
  };

  return (
    <ErrorBoundary>
      <SafeAreaView style={webStyles.container}>
        <View style={webStyles.header}>
          <Text style={webStyles.headerTitle}>RE-BMS v6.1 Command Center</Text>
        </View>
        <View style={webStyles.content}>
          {renderScreen()}
        </View>
        <WebTabBar activeTab={activeTab} onTabChange={setActiveTab} />
      </SafeAreaView>
    </ErrorBoundary>
  );
}

// Native App with full navigation
function NativeApp() {
  const GestureHandlerRootView = require('react-native-gesture-handler').GestureHandlerRootView;
  const AppNavigator = require('./src/navigation/AppNavigator').default;

  return (
    <GestureHandlerRootView style={styles.container}>
      <StatusBar style="light" />
      <AppNavigator />
    </GestureHandlerRootView>
  );
}

export default function App() {
  if (Platform.OS === 'web') {
    return <WebApp />;
  }
  return <NativeApp />;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
});

const webStyles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  header: {
    backgroundColor: colors.secondary,
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: colors.secondary,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingBottom: 8,
    paddingTop: 8,
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
  },
  activeTab: {
    borderTopWidth: 2,
    borderTopColor: colors.primary,
  },
  tabIcon: {
    fontSize: 20,
    marginBottom: 4,
  },
  tabLabel: {
    color: colors.muted,
    fontSize: 11,
    fontWeight: '600',
  },
  activeTabLabel: {
    color: colors.primary,
  },
});
