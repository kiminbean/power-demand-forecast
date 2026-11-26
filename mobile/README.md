# eXeco Jeju Power Market — Mobile Web App

Public, read-only mobile web app for the Jeju electricity market, built with Expo (React Native) and served as a web
export under `/mobile/`. On screen it is called "eXeco 제주 전력시장" and the header shows `v6.1`. The package and Expo
configs still carry older names and versions (`package.json`: `execo-mobile` 6.2.0, `app.json`: `RE-BMS` 5.0.0);
aligning them is approval item P8.

> **2026-11-26 redesign** (branch `ui/mobile-top1`): this README describes the redesigned app. Build contract:
> `docs/2026-11-26-mobile-ui-top1-design.md`, report: `docs/2026-11-26-mobile-ui-top1-report.md`.
> Production keeps serving the previous build until the round 1 evaluation passes and deployment is approved (P1).

## Screens

Four bottom tabs, one URL per screen (path depth 1; sub-states live in the query string).

| Tab | Path | Question it answers | What the visitor sees |
|---|---|---|---|
| 오늘 (Today) | `/` | What does power cost this hour, did the forecast hold, is there enough power? | KPX Jeju day-ahead confirmed price for the current trading hour, the baseline forecast (yesterday's price at the same hour) with its 80% band as a 24-hour step chart or table, demand · reserve margin · supply capacity as one equation and range bar, weather, documented accuracy |
| 입찰 (Bidding) | `/bidding` | When and how do power plants sell? | Day-ahead deadline countdown driven by the server's market status, a 10-step bid ladder for a fictional 50 MW wind plant (bidding calculator output, labeled as an estimate), how the price is set (example), a "my plant" card that says the feature is not offered in the public demo |
| 정산 (Settlement) | `/settlement` | How is sold power paid? | A card saying real settlement data is not public, and an example statement (labeled 예시) with its arithmetic. This tab makes no API calls |
| 알아보기 (Guide) | `/guide` | What is this app, and can the numbers be trusted? | 1-minute tour (5 steps), glossary (13 terms), how to read the numbers, forecast method and accuracy, data sources with `/realtime-status` response states, screen theme, app info |

- Query sub-states: `?hour=` (Today, Bidding), `?risk=` (Bidding), `?tour=1..5`, `?topic=`, `?term=` (Guide).
  Sheets open on any screen with `?sheet=method|sources|term|theme`. Browser back closes an open sheet first.
- Layout: under 600 px a bottom tab bar; 600–1023 px bottom tabs and a centered 640 px column; from 1024 px a
  240 px left rail; from 1280 px an extra "how to read this screen" panel (320 px). Sheets open at the bottom, or as
  a right-hand panel from 1024 px.

## Honesty Rules

- Every number sits in a card with a state chip, its source and a KST time: ● measured / confirmed, ┅ forecast,
  ▨ estimate, ┆ example. KPX values that come without a publication time show "관측 시각 미제공 · 조회 HH:MM KST".
- Invented numbers live only in `src/data/examples.ts` (the `Example<T>` brand) and render with the 예시 ribbon or chip.
- The app issues allow-listed GET requests only. Settlement and power-plant endpoints are not public, and nothing is
  submitted, saved or sent to KPX.
- Accuracy figures are the documented ones only: MAE 11.97 원/kWh, MAPE 11.75 %, R² 0.203 (walk-forward
  2025-07-22 to 12-18, the baseline itself, not a trained model).
- "Now" is the server-corrected clock in KST (fixed +9 h). Market logic never depends on the device time zone.

## Tech Stack

- Expo SDK 50.0.21, React Native 0.73.2, react-native-web 0.19.13, React 18.2, TypeScript
- Charts drawn with `react-native-svg` (`StepChart`, `LadderChart`, `RangeBar`); a small History API router
  (`src/app/router.web.ts`). React Navigation and react-native-chart-kit are used only by legacy screens, which are no
  longer in the web bundle.
- No new npm dependencies (`node_modules` is partly tracked in git).
- System fonts, no web fonts. Long Korean copy is fetched at runtime from `public/content/guide.ko.json` to keep the
  bundle small (budget 640,000 B raw; the redesign build is 624,723 B).

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

### Web export (same build as production)

Leave `EXPO_PUBLIC_API_URL` unset so the web bundle uses same-origin `/api` paths, and export to a folder outside the
repository:

```bash
cd mobile
env -u EXPO_PUBLIC_API_URL npx expo export --platform web --output-dir /path/outside/repo/mobile-web
```

### Tests

```bash
npm test            # node --test over tests/*.test.cjs except the two *.browser.* puppeteer tests
npm run typecheck   # tsc --noEmit -p . (legacy files included)
```

- The 35 tests in the 8 pre-redesign test files must keep passing without edits. One of them locks the hook-free
  `Header` in `App.tsx` (`eXeco`, `v6.1`, the back button as the first `TouchableOpacity`).
- The two browser tests (`chat-short-viewport`, `bidding-compact-layout`) still look for the old tab names, the Alan
  button and the VPP switch, so they fail until they are rewritten (approval item P9).

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
├── App.tsx                    # Hook-free Header (test contract) + route switch + Shell
├── app.json                   # Expo configuration (web name, lang, automatic light/dark)
├── package.json               # Dependencies; scripts "test" and "typecheck"
├── public/
│   ├── index.html             # Web shell: lang="ko", theme-color per theme, skeleton inside #root
│   └── content/guide.ko.json  # Long Korean copy (tour, glossary, method sheets), fetched on demand
├── src/
│   ├── app/                   # routes.ts (pure URL <-> state), router(.web).ts, Shell, SheetHost, theme, head(.web)
│   ├── ui/                    # tokens.ts (only colour source), Text, Icon, cards, chips, controls, Sheet, DataTable, charts/
│   ├── lib/                   # kst.ts (KST clock), format.ts (the only number/time formatter), provenance.ts, resource.ts
│   ├── data/                  # client.ts (allow-listed GETs), types.ts (shape checks), adapters.ts, examples.ts
│   ├── content/               # strings.ts (short copy with numbers), guide.ts (JSON loader and validation)
│   ├── features/              # today/, bidding/, settlement/, guide/: screen parts + model.ts (pure view models)
│   ├── screens/               # SMPForecastScreen.tsx is the Today tab; the other screens are legacy (not bundled)
│   └── services/, config/, hooks/, navigation/, theme/, utils/, types/, components/   # legacy, unchanged
└── tests/                     # node --test suites, fixtures (round-0 API snapshot, 52 state variants), ts-loader helper
```

## Theme

- Light and dark. The app follows the OS setting (`prefers-color-scheme`) by default; Guide › 화면 테마 lets the
  visitor pick System, Light or Dark (stored on this device only). Both `theme-color` meta tags follow the choice.
- Single source of colour: `src/ui/tokens.ts` (same values as the dashboard redesign). `src/theme/colors.ts` is kept
  byte-for-byte for the legacy screens and as the Header's default palette in tests.

| Role | Light | Dark |
|---|---|---|
| canvas (page) | `#F6F8FB` | `#0B1220` |
| surface (cards, header, tab bar) | `#FFFFFF` | `#121B31` |
| well (inner panels, table head) | `#F1F5F9` | `#1A2439` |
| text1 / text2 / text3 | `#0F172A` / `#334155` / `#56657A` | `#F1F5F9` / `#CBD5E1` / `#94A3B8` |
| primary (active tab, focus ring) | `#2563EB` | `#60A5FA` |
| link | `#1D4ED8` | `#60A5FA` |

- Data colours are the same in both themes: demand `#3B82F6`, supply capacity `#64748B`, reserve `#16A34A`,
  SMP `#C026D3` (80% band at 13 % opacity), watch boundary `#B45309`.
- Type: system fonts, five sizes per screen (display 32, or 28 at 360 px and below; title 20; body 15; label 13;
  meta 12, nothing smaller), tabular numbers.
- Every interactive target is at least 44×44 px, and text contrast meets WCAG AA in both themes
  (`tests/tokens-contrast.test.cjs` checks every text/background pair).

## Market Rules Shown in the App

| Market | Rule as shown | Source |
|--------|---------------|--------|
| Day-ahead (DAM) | Bids close at 10:00 KST on the day before the trading day; the countdown follows the server | `/api/v1/market-status` |
| Real-time (RTM) | 15-minute intervals; no RTM price forecast is shown (no validated Jeju model) | `/api/v1/market-status` |

The explanations are simplified and have not had a domain review yet (approval item P7).

## Legacy Screens

The previous screens (command-center dashboard, 10-segment bid editor, portfolio, settlement analytics, plant
registration, DAM/RTM simulators, Alan chat) stay under `src/screens/` because existing tests read them. `App.tsx` no
longer imports them, so they are not part of the public web bundle. Deleting them is approval item P2.

## License

MIT
