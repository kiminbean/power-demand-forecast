# RE-BMS 외부 배포 가이드

## 아키텍처

```
┌─────────────────────┐     Internet      ┌─────────────────────┐
│   친구 맥북          │◄────────────────►│   서버 (Docker)      │
│   - iOS App         │     ngrok/VPN     │   - API Server      │
│   - Expo Go         │                   │   - ML Models       │
└─────────────────────┘                   └─────────────────────┘
```

---

## 1. Docker 서버 설정 (API 서버)

### 1.1 사전 요구사항

```bash
# Docker 설치 확인
docker --version
docker-compose --version

# ngrok 설치 (macOS)
brew install ngrok

# ngrok 계정 설정 (https://ngrok.com 에서 가입)
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### 1.2 Docker 이미지 빌드 및 실행

```bash
# 프로젝트 디렉토리로 이동
cd power-demand-forecast

# Docker 이미지 빌드
docker-compose build api

# API 서버 실행
docker-compose up -d api

# 로그 확인
docker-compose logs -f api
```

### 1.3 ngrok으로 외부 노출

```bash
# ngrok 실행 (포트 8000)
ngrok http 8000

# 출력 예시:
# Forwarding   https://xxxx-xxx-xxx.ngrok-free.app -> http://localhost:8000
```

**ngrok URL 복사** (예: `https://xxxx-xxx-xxx.ngrok-free.app`)

---

## 2. 모바일 앱 설정 (친구 맥북)

### 2.1 환경 설정 변경

`mobile/src/config/environment.ts` 파일 수정:

```typescript
// Before (로컬 개발)
export const API_URL = 'http://localhost:8000';
export const CURRENT_ENV: Environment = 'local';

// After (외부 접속)
export const API_URL = 'https://xxxx-xxx-xxx.ngrok-free.app';  // ngrok URL
export const CURRENT_ENV: Environment = 'docker';
```

### 2.2 앱 실행

```bash
# mobile 디렉토리로 이동
cd mobile

# 의존성 설치
npm install

# Expo 실행
npx expo start

# iOS 시뮬레이터에서 실행
# 또는 Expo Go 앱으로 QR 코드 스캔
```

---

## 3. 문제 해결

### 3.1 연결 오류

```bash
# Docker 상태 확인
docker-compose ps
docker-compose logs api

# ngrok 상태 확인
curl localhost:4040/api/tunnels

# 포트 확인
lsof -i :8000
```

### 3.2 ngrok 세션 만료

ngrok 무료 계정은 2시간 후 세션이 만료됩니다:
- ngrok 재시작 후 새 URL 복사
- environment.ts 업데이트
- 앱 재시작

### 3.3 CORS 오류

Docker의 환경변수 확인:
```yaml
# docker-compose.yml
environment:
  - CORS_ORIGINS=*  # 모든 origin 허용
```

---

## 4. 보안 주의사항

- ngrok 무료 계정: 2시간 세션 제한, 랜덤 URL
- 프로덕션 배포 시: HTTPS 필수, API 키 관리, CORS 설정 제한

---

## 5. 대안 배포 방법

### 5.1 포트 포워딩 (공유기 설정)

1. 공유기 관리자 페이지 접속
2. 포트 포워딩 설정: 외부 8000 → 내부 IP:8000
3. 공인 IP 확인: curl ifconfig.me
4. environment.ts에 공인 IP 설정

### 5.2 Tailscale VPN (추천)

```bash
# 양쪽 맥북에 Tailscale 설치
brew install tailscale
tailscale up

# Tailscale IP로 연결 (100.x.x.x)
```

---

## 체크리스트

### 서버 측
- [ ] Docker 설치 완료
- [ ] ngrok 설치 및 설정 완료
- [ ] docker-compose up -d api 실행
- [ ] ngrok http 8000 실행
- [ ] ngrok URL 확인

### 클라이언트 측
- [ ] Node.js 설치 완료
- [ ] npm install 실행
- [ ] environment.ts 업데이트
- [ ] npx expo start 실행
- [ ] API 연결 테스트
