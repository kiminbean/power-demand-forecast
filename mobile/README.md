# RE-BMS Mobile v5.0

**Renewable Energy Bidding Management System** - Mobile App

## Overview

React Native (Expo) mobile application for managing renewable energy bids in the KPX Day-Ahead Market (DAM) and Real-Time Market (RTM).

## Features

- **Command Center Dashboard**: Real-time SMP forecasts and market status
- **10-Segment Bidding**: Create and manage bids with monotonic price constraint
- **Portfolio Management**: Monitor renewable resources (solar/wind)
- **Settlement Analytics**: Track revenue, penalties, and imbalance rates

## Tech Stack

- React Native 0.73
- Expo SDK 50
- React Navigation 6
- react-native-chart-kit
- TypeScript

## Getting Started

### Prerequisites

- Node.js 18+
- Expo CLI
- iOS Simulator or Android Emulator (optional)
- Expo Go app (for physical device testing)

### Installation

```bash
cd mobile
npm install
```

### Running the App

```bash
# Start Expo development server
npm start

# iOS
npm run ios

# Android
npm run android

# Web
npm run web
```

### Backend Connection

The native app defaults to the public Tailscale Funnel origin
`https://ib-macbookpro.tail918684.ts.net:10000`. Web requests default to the
current origin, including when the app is served under `/mobile/`.

To use another API origin (or an origin with a base path), set
`EXPO_PUBLIC_API_URL` before starting Expo or building the app, for example:

```bash
EXPO_PUBLIC_API_URL=https://example.com/backend npm start
```

The override applies to both native and web; an unset or blank value uses the
platform default. Expo embeds public environment variables and the native default
in the app bundle: changing the URL requires a new build/update of that bundle,
and values must not contain secrets. The public proxy deliberately blocks sensitive
endpoints; selecting its origin does not grant access to them.

## Project Structure

```
mobile/
├── App.tsx                 # App entry point
├── app.json               # Expo configuration
├── package.json           # Dependencies
├── src/
│   ├── assets/            # Icons, images
│   ├── components/        # Reusable UI components
│   ├── hooks/             # Custom React hooks
│   │   ├── index.ts
│   │   └── useApi.ts      # API data fetching hooks
│   ├── navigation/        # React Navigation setup
│   │   └── AppNavigator.tsx
│   ├── screens/           # Screen components
│   │   ├── DashboardScreen.tsx
│   │   ├── BiddingScreen.tsx
│   │   ├── BidDetailScreen.tsx
│   │   ├── PortfolioScreen.tsx
│   │   └── SettlementScreen.tsx
│   ├── services/          # API services
│   │   ├── api.ts         # Axios client
│   │   └── index.ts
│   └── theme/             # Styling
│       └── colors.ts      # Dark theme colors
```

## Key Features

### 10-Segment Bidding

Each bid consists of 10 price-quantity segments with a **monotonic constraint**:

```
Price(Seg 1) ≤ Price(Seg 2) ≤ ... ≤ Price(Seg 10)
```

### Jeju Pilot Imbalance Rules

- **Tolerance Band**: ±12%
- **Over-generation Penalty**: 80% of SMP
- **Under-generation Penalty**: 120% of SMP

### Market Deadlines

| Market | Deadline | Granularity |
|--------|----------|-------------|
| DAM | D-1 10:00 | Hourly (24h) |
| RTM | 15 min before | 15-minute |

## Theme

The app uses a dark "Command Center" theme with the following color palette:

- **Background**: #0e1117 (primary), #1a1f2c (secondary)
- **Brand**: #6366f1 (Indigo), #10b981 (Emerald accent)
- **Status**: Success (green), Warning (amber), Danger (red)

## License

MIT
