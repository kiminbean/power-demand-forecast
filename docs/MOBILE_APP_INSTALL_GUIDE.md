# RE-BMS 모바일 앱 설치 가이드

이 가이드는 친구의 맥북 또는 윈도우 PC에서 RE-BMS 모바일 앱을 설치하고 실행하는 방법을 설명합니다.

---

## 📋 사전 요구사항

| 항목 | macOS | Windows |
|------|-------|---------|
| Node.js | v18 이상 | v18 이상 |
| npm | v9 이상 | v9 이상 |
| Git | 기본 설치됨 | 별도 설치 필요 |
| Expo Go 앱 | iOS/Android 스토어 | Android 스토어 |

---

## 🍎 macOS 설치 방법

### 1단계: Homebrew 설치 (없는 경우)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2단계: Node.js 설치

```bash
brew install node
node -v  # v18 이상 확인
npm -v   # v9 이상 확인
```

### 3단계: 프로젝트 클론

```bash
cd ~
git clone https://github.com/kiminbean/power-demand-forecast.git
cd power-demand-forecast/mobile
```

### 4단계: 의존성 설치

```bash
npm install
```

### 5단계: 앱 실행

```bash
npx expo start
```

### 6단계: 앱 접속

터미널에 QR 코드가 표시됩니다:

- **iOS**: 카메라 앱으로 QR 코드 스캔 → Expo Go에서 열기
- **Android**: Expo Go 앱에서 QR 코드 스캔
- **웹 브라우저**: `w` 키 누르기

---

## 🪟 Windows 설치 방법

### 1단계: Node.js 설치

1. https://nodejs.org 접속
2. **LTS 버전** 다운로드 (v18 이상)
3. 설치 프로그램 실행 → 기본 옵션으로 설치
4. 설치 확인:
   ```cmd
   node -v
   npm -v
   ```

### 2단계: Git 설치

1. https://git-scm.com/download/win 접속
2. 다운로드 후 설치 (기본 옵션)
3. 설치 확인:
   ```cmd
   git --version
   ```

### 3단계: 프로젝트 클론

**PowerShell** 또는 **명령 프롬프트**에서:

```cmd
cd %USERPROFILE%
git clone https://github.com/kiminbean/power-demand-forecast.git
cd power-demand-forecast\mobile
```

### 4단계: 의존성 설치

```cmd
npm install
```

### 5단계: 앱 실행

```cmd
npx expo start
```

### 6단계: 앱 접속

- **Android**: Expo Go 앱에서 QR 코드 스캔
- **웹 브라우저**: `w` 키 누르기

> ⚠️ Windows에서는 iOS 시뮬레이터를 사용할 수 없습니다.

---

## 📱 Expo Go 앱 설치

### iOS (iPhone/iPad)
1. App Store에서 "Expo Go" 검색
2. 설치 후 실행
3. 카메라로 QR 코드 스캔

### Android
1. Play Store에서 "Expo Go" 검색
2. 설치 후 실행
3. 앱 내 QR 스캐너로 스캔

---

## 🌐 API 서버 연결 설정

현재 앱은 다음 서버에 연결되도록 설정되어 있습니다:

```
https://fourpenny-homochrome-amir.ngrok-free.dev
```

### API 서버 URL 변경 방법

`mobile/src/config/environment.ts` 파일 수정:

```typescript
// 현재 설정 (ngrok)
export const API_URL = 'https://fourpenny-homochrome-amir.ngrok-free.dev';
export const CURRENT_ENV: Environment = 'docker';

// 로컬 개발로 변경 시
export const API_URL = 'http://localhost:8000';
export const CURRENT_ENV: Environment = 'local';
```

---

## ❓ 문제 해결

### 1. "npm install" 오류

```bash
# 캐시 삭제 후 재시도
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### 2. Expo 연결 실패

```bash
# Expo 캐시 삭제
npx expo start --clear
```

### 3. QR 코드가 안 보이는 경우

```bash
# 터널 모드로 실행
npx expo start --tunnel
```

### 4. "Network request failed" 오류

- API 서버가 실행 중인지 확인
- ngrok URL이 유효한지 확인 (세션 만료 시 새 URL 필요)
- 방화벽 설정 확인

### 5. Windows: PowerShell 실행 정책 오류

관리자 권한 PowerShell에서:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

---

## 📞 연락처

문제가 해결되지 않으면 프로젝트 관리자에게 연락하세요.

- GitHub Issues: https://github.com/kiminbean/power-demand-forecast/issues
