# Project Status Backup
> Last Updated: 2025-12-20 14:10 KST

## Project Overview
- **Project**: Jeju Power Demand Forecast System (RE-BMS)
- **Repository**: https://github.com/kiminbean/power-demand-forecast (PRIVATE)
- **Version**: v6.0.0 (React Desktop Web Application)
- **Release**: https://github.com/kiminbean/power-demand-forecast/releases/tag/v6.0.0
- **License**: Proprietary (All Rights Reserved)

---

## Latest Changes (2025-12-20)

### 🚀 RE-BMS v6.0.0 Release

#### Docker Deployment with Private Access
| 항목 | 상태 |
|------|------|
| Basic Authentication | ✅ 설정 완료 |
| rebms-api 컨테이너 | ✅ 테스트 완료 |
| rebms-web 컨테이너 | ✅ 테스트 완료 |
| 7개 페이지 검증 | ✅ 모두 정상 |

#### Docker 구성
| 서비스 | 컨테이너 | 포트 | 설명 |
|--------|----------|------|------|
| `api` | rebms-api | 8506 | FastAPI 백엔드 |
| `web` | rebms-web | 8600 | React + Nginx + Basic Auth |

#### Docker 테스트 스크린샷
```
docs/screenshots/docker_dashboard.png   - 메인 대시보드
docs/screenshots/docker_smp.png         - SMP 예측
docs/screenshots/docker_bidding.png     - 입찰 관리
docs/screenshots/docker_portfolio.png   - 포트폴리오
docs/screenshots/docker_settlement.png  - 정산
docs/screenshots/docker_map.png         - 제주 지도
docs/screenshots/docker_analysis.png    - 분석
```

### 🔒 보안 설정 변경

#### License 변경
| 항목 | 이전 | 이후 |
|------|------|------|
| 라이선스 | MIT (개방형) | Proprietary (독점) |
| 복사/수정/배포 | ✅ 허용 | ❌ 금지 |
| 상업적 사용 | ✅ 허용 | ❌ 금지 |

#### Repository Visibility
| 항목 | 이전 | 이후 |
|------|------|------|
| 공개 설정 | Public | **Private** |
| 접근 권한 | 누구나 | 소유자/협업자만 |

### 📝 문서 업데이트

#### README.md 변경사항
- Docker 배포 섹션 대폭 확장 (94줄 추가)
- v6 Docker 스크린샷 갤러리 추가
- 라이선스 배지 변경 (MIT → Proprietary)
- 버전 업데이트 (v4.0.7 → v6.0.0)

### Recent Commits (2025-12-20)
```
1954e23 chore: Change license to Proprietary (All Rights Reserved)
b184b6b docs: Add RE-BMS v6.0 Docker deployment guide to README
9582c03 docs: Add Docker deployment test screenshots
bddb954 fix: Docker volume mount configuration for v6 deployment
```

---

## RE-BMS v6.0 Features

### 7 Dashboard Pages
| 페이지 | 경로 | 기능 |
|--------|------|------|
| 대시보드 | `/` | 실시간 전력수급 현황 |
| SMP 예측 | `/smp` | 24시간 SMP 예측 (q10/q50/q90) |
| 입찰 관리 | `/bidding` | 10-Segment KPX 입찰 매트릭스 |
| 포트폴리오 | `/portfolio` | 제주 20개 발전소 관리 |
| 정산 | `/settlement` | 수익/불균형 정산 분석 |
| 제주 지도 | `/map` | Leaflet 발전소 위치 |
| 분석 | `/analysis` | XAI 피처 중요도 |

### Tech Stack
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS
- **Charts**: Recharts, React Leaflet
- **Backend**: FastAPI, Python 3.11, PyTorch
- **Infrastructure**: Docker, Nginx, Basic Auth

### 접속 정보
```
Development: http://localhost:8508
Docker: http://localhost:8600 (인증 필요)
Username: admin
Password: (htpasswd 설정)
```

---

## Docker 명령어

```bash
# 시작
docker-compose -f docker/docker-compose.v6.yml up -d

# 중지
docker-compose -f docker/docker-compose.v6.yml down

# 재빌드
docker-compose -f docker/docker-compose.v6.yml up -d --build

# 로그
docker-compose -f docker/docker-compose.v6.yml logs -f

# 전체 정리
docker system prune -af --volumes
```

---

## Key Files

### Docker Configuration
```
docker/docker-compose.v6.yml    - v6 Docker Compose
docker/Dockerfile.api           - FastAPI 이미지
docker/.env                     - 환경변수
docker/htpasswd                 - Basic Auth 인증 파일
docker/setup-auth.sh            - 인증 설정 스크립트
web-v6/Dockerfile               - React 프론트엔드 이미지
web-v6/nginx.conf               - Nginx 설정 (Basic Auth)
```

### v6 Web Application
```
web-v6/src/pages/Dashboard.tsx      - 대시보드
web-v6/src/pages/SMPPrediction.tsx  - SMP 예측
web-v6/src/pages/Bidding.tsx        - 입찰 관리
web-v6/src/pages/Portfolio.tsx      - 포트폴리오
web-v6/src/pages/Settlement.tsx     - 정산
web-v6/src/pages/Map.tsx            - 제주 지도
web-v6/src/pages/Analysis.tsx       - 분석
```

### License
```
LICENSE                         - Proprietary (All Rights Reserved)
```

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v6.0.0** | **2025-12-20** | **React Desktop Web + Docker Private Deploy** |
| v5.0.0 | 2025-12-19 | React Native Mobile App |
| v4.0.7 | 2025-12-19 | Enhanced chart (예비전력, 태양광, 풍력) |
| v4.0.6 | 2025-12-19 | Reserve rate bug fix |
| v4.0.5 | 2025-12-19 | GE Inertia layout |
| v4.0.4 | 2025-12-19 | Slack webhook |
| v4.0.3 | 2025-12-19 | Email notification |

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Run `git log --oneline -10`
3. Repository is **PRIVATE** - requires authentication
4. License is **Proprietary** - all rights reserved

---

## Environment
- Python 3.13, PyTorch 2.0+
- Node.js 20, React 18, TypeScript
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- Docker Desktop

## Security Notes
- Repository: **PRIVATE**
- License: **Proprietary (All Rights Reserved)**
- Docker: **Basic Authentication required**
- No public access without explicit permission
