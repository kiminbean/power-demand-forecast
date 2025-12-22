/**
 * RE-BMS Mobile App Navigator
 * Bottom Tab Navigation with Dark Theme
 */

import React from 'react';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';

import { colors } from '../theme/colors';

// Screens
import DashboardScreen from '../screens/DashboardScreen';
import BiddingScreen from '../screens/BiddingScreen';
import BidDetailScreen from '../screens/BidDetailScreen';
import KPXSimulationScreen from '../screens/KPXSimulationScreen';
import RTMSimulationScreen from '../screens/RTMSimulationScreen';
import PortfolioScreen from '../screens/PortfolioScreen';
import SettlementScreen from '../screens/SettlementScreen';

// Types
export type RootTabParamList = {
  DashboardTab: undefined;
  BiddingTab: undefined;
  PortfolioTab: undefined;
  SettlementTab: undefined;
};

export type BiddingStackParamList = {
  BiddingList: undefined;
  BidDetail: { bidId: string };
  CreateBid: undefined;
  KPXSimulation: {
    segments?: { id: number; quantity: number; price: number }[];
    selectedHour?: number;
    smpForecast?: { q10: number; q50: number; q90: number };
  };
  RTMSimulation: {
    segments?: { id: number; quantity: number; price: number }[];
    selectedHour?: number;
    smpForecast?: { q10: number; q50: number; q90: number };
  };
};

// Theme
const DarkTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    primary: colors.brand.primary,
    background: colors.background.primary,
    card: colors.background.secondary,
    text: colors.text.primary,
    border: colors.border.primary,
    notification: colors.status.danger,
  },
};

// Navigators
const Tab = createBottomTabNavigator<RootTabParamList>();
const BiddingStack = createNativeStackNavigator<BiddingStackParamList>();

// Bidding Stack Navigator
function BiddingStackNavigator() {
  return (
    <BiddingStack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: colors.background.secondary },
        headerTintColor: colors.text.primary,
        headerTitleStyle: { fontWeight: 'bold' },
      }}
    >
      <BiddingStack.Screen
        name="BiddingList"
        component={BiddingScreen}
        options={{ title: 'Bidding' }}
      />
      <BiddingStack.Screen
        name="BidDetail"
        component={BidDetailScreen}
        options={{ title: 'Bid Details' }}
      />
      <BiddingStack.Screen
        name="KPXSimulation"
        component={KPXSimulationScreen}
        options={{ title: 'DAM Simulation', headerShown: false }}
      />
      <BiddingStack.Screen
        name="RTMSimulation"
        component={RTMSimulationScreen}
        options={{ title: 'RTM Simulation', headerShown: false }}
      />
    </BiddingStack.Navigator>
  );
}

// Tab Icon Component
function TabIcon({ name, focused, color }: { name: string; focused: boolean; color: string }) {
  const iconName = focused ? name : `${name}-outline`;
  return <Ionicons name={iconName as any} size={24} color={color} />;
}

// Main App Navigator
export default function AppNavigator() {
  return (
    <NavigationContainer theme={DarkTheme}>
      <Tab.Navigator
        screenOptions={{
          tabBarStyle: {
            backgroundColor: colors.background.secondary,
            borderTopColor: colors.border.primary,
            height: 60,
            paddingBottom: 8,
            paddingTop: 8,
          },
          tabBarActiveTintColor: colors.brand.primary,
          tabBarInactiveTintColor: colors.text.muted,
          tabBarLabelStyle: {
            fontSize: 11,
            fontWeight: '600',
          },
          headerStyle: {
            backgroundColor: colors.background.secondary,
          },
          headerTintColor: colors.text.primary,
          headerTitleStyle: {
            fontWeight: 'bold',
          },
        }}
      >
        <Tab.Screen
          name="DashboardTab"
          component={DashboardScreen}
          options={{
            title: 'Dashboard',
            tabBarIcon: ({ focused, color }) => (
              <TabIcon name="grid" focused={focused} color={color} />
            ),
            headerTitle: 'RE-BMS Command Center',
          }}
        />
        <Tab.Screen
          name="BiddingTab"
          component={BiddingStackNavigator}
          options={{
            title: 'Bidding',
            headerShown: false,
            tabBarIcon: ({ focused, color }) => (
              <TabIcon name="trending-up" focused={focused} color={color} />
            ),
          }}
        />
        <Tab.Screen
          name="PortfolioTab"
          component={PortfolioScreen}
          options={{
            title: 'Portfolio',
            tabBarIcon: ({ focused, color }) => (
              <TabIcon name="layers" focused={focused} color={color} />
            ),
            headerTitle: 'Resource Portfolio',
          }}
        />
        <Tab.Screen
          name="SettlementTab"
          component={SettlementScreen}
          options={{
            title: 'Settlement',
            tabBarIcon: ({ focused, color }) => (
              <TabIcon name="wallet" focused={focused} color={color} />
            ),
            headerTitle: 'Settlement & Revenue',
          }}
        />
      </Tab.Navigator>
    </NavigationContainer>
  );
}
