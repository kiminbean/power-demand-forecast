# Project Status Backup
> Last Updated: 2026-11-27 morning (mobile web app /mobile/ UI TOP1 redesign on branch ui/mobile-top1: build F0–F12, fix steps F13a–F14, evaluation rounds 1–3 done — round 3 passed the bar (mean 8.81, lowest R1·R8 8.4, all four lenses incl. M1–M9 and I1–I5 pass, 41 defects open incl. 3 high), not deployed, not pushed; 2026-11-26 evening: round 1 passed narrowly (mean 8.53); earlier 2026-11-26: session memory + docs refresh; 2026-11-25: dashboard app_v4 UI TOP1 redesign merged·deployed·pushed; 2026-11-07: CI/CD ran green during a 9-minute public window; 2026-11-06: default model ar_gbm)

## Project Overview
- **Project**: Jeju Power Demand Forecast System (RE-BMS)
- **Repository**: https://github.com/kiminbean/power-demand-forecast (PRIVATE) ✅ Confirmed
- **Current Version**: v8.0.0 (Real-time API Edition) ✅ Complete
- **Previous Version**: v7.0.0 (React Desktop Web Application)
- **License**: Proprietary (All Rights Reserved)

---

## Current Session (2026-11-23 ~ 11-26)

### Task: 모바일 웹앱(/mobile/) UI/UX TOP1 개편 — 빌드 F0\~F12, 수정 F13a\~F14, round 1\~3 평가 완료 (2026-11-26 \~ 27) ⏳ round 3 합격선 통과·운영 미반영·푸시 안 함
- 사용자 목표: 공개 모바일 웹앱(`/mobile/`, Expo + react-native-web)의 UI/UX를 경쟁사 TOP1 수준으로. 기준 제품 Electricity Maps 앱(약 9.1). 합격선 = R1\~R8 모두 8.0 이상 · 평균 8.5 이상 + 게이트 M1\~M9·I1\~I5 전부. round 0(개편 전) 평균 4.34
- 진행: 시안 3개 중 C(비전문가 안내형)를 기반으로 설계 명세 `docs/2026-11-26-mobile-ui-top1-design.md` → 빌드 F0\~F12(단계마다 빌더 1명·로컬 커밋 1개). 워크트리 `.claude/worktrees/mobile-top1`, 브랜치 `ui/mobile-top1`(기준 `f2fbb5a2`)
- 결과: 탭 4개(오늘·입찰·정산·알아보기), 화면마다 주소, 라이트·다크(시스템 + 앱 안 선택), 모든 숫자에 상태 칩·출처·KST 시각, 지어낸 값은 `mobile/src/data/examples.ts`에서만. 앨런 AI·입찰 제출·발전소 등록·시뮬레이션은 공개 빌드에서 뺌(파일은 남김, 삭제는 P2)
- 최종 `check.sh`: tests 213 · fail 0 · 기존 35/35(수정 없음) · tsc 0 · export 0 · 번들 624,723 B(개편 전 725,177) · 금지 문자열 0 · 잠근 파일 변경 0 · 자리표시 0. 번들 sha256 `8c65b3ad…`
- round1-pre 게이트 측정(평가 직전, 같은 측정 도구): 1,876기록(11.5절 상태 103개 + 변형 52개 × 6폭 × 2테마, grid 48, 회전 4) · axe 936회 · 보조 점검 284건 · 키보드 · 시간대 3개에서 M1\~M8·I1\~I5 실패 0. M9는 조건부: 서버 캐시를 데운 뒤 5,427 · 5,427 · 5,420 ms, 있는 그대로 잰 첫 3회 6,631 · 5,412 · 5,420 ms(서버 하루전 가격 캐시가 비어 `/smp/day-ahead` 1,685 ms). 증거 `scratchpad/mobile/eval/round1-pre/`, 게이트 표 `scratchpad/mobile/eval/steps/F12/INDEX.md`
- round 1 평가(렌즈 4개, 17:52\~19:14 KST, 같은 번들·앱 코드 변경 없음): 항목 평균 R1 8.0 · R2 8.6 · R3 8.2 · R4 9.0 · R5 9.0 · R6 8.4 · R7 8.75 · R8 8.3, 평균 8.53. layout_a11y(M1\~M9)·integrity(I1\~I5) 통과 → 워크플로 합격선을 넘었지만 턱걸이다. mentor 렌즈 불합격(MEN-COPY: 둘러보기 2/5 제목 "내일 값은 어제 값으로"가 본문과 반대 — 명세 4.4절 문구 자체의 오류, API 원문 'moderate', 풀이 없는 정확도 낱말 등). M9: 평가자 두 묶음 6회 모두 6.0초 이하(가장 느린 5,969 ms), 캡처의 있는 그대로 첫 회 6,129 ms. 결함 54건(높음 4). 증거 `scratchpad/mobile/eval/round1/`(`ROUND1_RESULT.json`, 렌즈별 JSON)
- 수정 F13a·F13b·F13c(2026-11-26 19:45 \~ 11-27 01:58 KST): 오케스트레이터 수정 결정 D1\~D18(`scratchpad/mobile/eval/round1/FIX_PLAN_r1.md`, 명세 2.4절). round 2(앱 코드 `6fb6bf64`): 평균 8.73이지만 관문 넷(mentor MEN-COPY, layout_a11y M5·FOCUS, integrity NOW-KST)이 떨어짐 → F14(`c08d443b`, 04:02 \~ 06:42 KST)가 고침
- round 3(06:42 \~ 08:59 KST, 앱 코드 `c08d443b`): 항목 평균 R1 8.4 · R2 8.7 · R3 8.65 · R4 9.15 · R5 9.25 · R6 8.85 · R7 9.1 · R8 8.4, 평균 8.81, 네 렌즈 모두 통과(떨어진 관문 0) → 합격선 통과. M9 21회 5,024 \~ 5,160 ms(하루전 가격 +1,200 ms 지연 포함). `check.sh` tests 238 · 기존 35/35 · tsc 0 · 번들 639,947 B(한도까지 53 B). 평가한 export `scratchpad/mobile/build/runs/R3-20261127-064235/export`, 번들 sha256 `a8ec24512d8cf1630aa6d215d447fccafe9b4223e1b1030dd044afedf65308db`. 결함 41건(높음 3: 390 첫 화면에 그림 없음, 768 두 칸의 빈 자리, 번들 여유 53 B). 증거 `scratchpad/mobile/eval/round3/`(`LENSES_r3.json`, `GATES.md`)
- 보고서: `docs/2026-11-26-mobile-ui-top1-report.md`(바뀐 것, 게이트 수치, round 1\~3 점수·관문·결함, 수정 결정 D1\~D18, 캡처 경로와 개편 전후 스크린샷 4장(개편 후는 round 3), 남은 승인). `mobile/README.md`·`CHANGELOG.md` 갱신
- 남은 일:
  - P1(운영 배포) 승인 여부(명세 13절 절차, 위 sha256 확인부터). round 3 높음 결함 셋을 배포 전에 고칠지도 함께 정한다. 고치려면 번들 자리(P3 압축 또는 문구를 `guide.ko.json`으로 이전)가 먼저다(보고서 5.1·5.4절)
  - 대시보드 후속: '관심 경계 (예비율 15%)'·단계 관심·주의·문장 속 `07h · 06–07시`(D9·D11), '보통' 판정 문장(F14). 모바일만 바꿨다
  - 승인 항목 P1(운영 배포, 명세 13절 백업·되돌리기 명령) \~ P11(푸시·병합). 번들 여유를 가장 크게 늘리는 것은 P3(번들 gzip: 639,947 → 188,040 B)이다
  - 측정 도구(`check.sh`·`sweep.py`·`run_gate_checks_v2.sh`)와 증거는 세션 scratchpad(`scratchpad/mobile/`)에 있어 세션이 끝나면 사라진다. 저장소만으로 돌릴 수 있는 것은 `cd mobile && npm test && npm run typecheck`와 웹 export다(`mobile/README.md`)

### Task: 세션 메모리 + 문서 갱신 (2026-11-26) ✅ COMPLETE
- 사용자 요청: "지금까지 대화내용과 작업내용 메모리에 기억", "업데이트된 내용 관련해서 모든 문서 최신 업데이트", "커밋, 푸시"
- 메모리: 대시보드 개편 결과, UI 평가 교훈, 진행 보고 방식, 이번 세션 작업 기록을 추가했다. 공개 주소, 개발 서버, 미해결 과제, 푸시 승인 방식, 사용자 프로필을 갱신했다.
- 문서:
  - README: 대시보드 소개, 스크린샷, 운영·개발 서버, 관문
  - CHANGELOG: 2026-11-23~26 항목
  - CLAUDE.md: Frontend, 관문, 개발 서버
  - AGENTS.md: 대시보드 운영, 관문
  - 설명 페이지: '한계' 칸, 02장 배포 구조(루트 → execo :3311, /jref → :8501)와 그림 1, 웹 v6 종료(공개 웹 버전 4개). 11-10 미커밋 수정까지 함께 커밋(사용자 승인)
  - 개편 보고서: 6.3절 공개 주소 확인
  - 개편 명세: 머리에 결과 안내
- 검토: 독립 검토 2명(사실·형식)의 지적 19건을 반영했다. 문서의 관문 명령을 그대로 실행해 35 passed · 29/29 · [] / []를 확인했다. 커밋 `648698f3`(설명 페이지)·`3790a540`(나머지), CI·CD 성공
- 그대로 둔 것: eXeco 웹앱 문서(DASHBOARD_SCENARIO, system_architecture), 과거 기록 문서(CONVERSATION_PROMPTS, PROMPT_HISTORY, SESSION_SUMMARY), 영업 시나리오(kpx_sales)

### Task: 웹 대시보드(app_v4) UI/UX TOP1 개편 ✅ 병합·운영 반영·푸시 완료 (TOP1 합격선 충족: 10차 루브릭·무결성 + 10b 화면 배치 독립 재평가)
- 사용자 목표: "웹 대시보드의 GUI를(UI/UX) 경쟁 업체 TOP1 수준으로 업그레이드". TOP1 = UI 품질 기준(60Hertz 8.7 / Electricity Maps 9.2). 합격선 = R1~R8 모두 8.0 이상 · 평균 8.5 이상 + 화면 배치·무결성 관문 통과
- 진행: 워크트리 `.claude/worktrees/dashboard-top1`(브랜치 `ui/dashboard-top1`, 지우기 전에 사용자에게 묻는다)에서 설계 명세 1~14장 → 빌드·수정 F1~F12 → 평가 10차(평가자 4: 동등성·멘토·화면 배치·무결성). 평균 8.09 → 8.24 → 8.18 → 8.41 → 8.38 → 8.28 → 8.42 → 8.51 → 8.49 → 8.57
- 10차(최종, `b84f9be3`): 루브릭 평균 8.57 통과(R8 8.05가 가장 낮음), 무결성 27/27 통과, 화면 배치 21/22. 360px '게이지로 보기' 펼침 빈 띠 67~68px(기준 60)로 불합격했다. 이 결함은 개편 전부터 있었고, 마무리 커밋 `778a3974`(CSS 한 줄)로 고친 뒤 평가자 탐침상 49px이다. 11차 전체 평가는 사용자 결정(1~2시간 안 마무리)으로 하지 않았고, 대신 독립 평가자가 화면 배치 관문만 다시 평가해(10b, 21:51~22:32) 22개 항목 모두 통과했다
- 병합·배포: main fast-forward `15b49880` → `launchctl kickstart -k gui/$(id -u)/com.powerdemand.dashboard`. 운영 8504에서 두 테마 × 1440·390 × 4탭 모두 넘침·깨진 글자·콘솔 오류 0. origin 푸시 `bca36f30..15b49880`(이전부터 올리지 않았던 main 커밋 5개 포함)
- 문서: `docs/2026-11-23-dashboard-ui-top1-design.md`(명세), `docs/2026-11-23-dashboard-ui-top1-report.md`(전후 보고서: 남은 결함 22건 5.3절, 승인 대기 P1~P11 5.1절), `docs/2026-11-23-dashboard-ui-top1/`(ui_guard·logic_guard·촬영 도구, 스크린샷)
- 남은 일:
  - 공개 주소 :8443은 2026-11-25 21:45~22:33에 응답하지 않았다(Tailscale CLI 상태 NeedsLogin, Tokyo 중계 연결 실패). 2026-11-26 04:13~04:15에 다시 확인하니 HTTP 200·상태 검사 ok였고, 두 테마 × 1440·390px에서 새 화면이 그려졌다(넘침·오류 0). 04:13 확인 때 PATH의 CLI(1.96.4)는 NeedsLogin을 보고했지만 공개 주소는 응답했고, 04:44에는 같은 CLI가 Running을 보고했다(실제 Tailscale 1.104.7, 원인 미확인). 공개 주소는 주소에 직접 요청해 확인한다
  - SMP 수집기가 `data/smp/smp_real_epsis.csv`를 쓰는 도중 앱이 읽으면 부분 표가 1시간 캐시돼 'SMP 지연 · 1023일 전 가격'이 보인다(21:15 개발 서버에서 관찰, 운영은 재시작으로 정상). 원자적 쓰기(os.replace) 또는 오래된 표 거부가 필요하다(데이터 로직, 승인 필요)
  - 보고서 5.3절 결함(사이드바를 연 1181~1320px 넘침, SMP CSV 내보내기의 0.00 입력 원값 등)과 P1~P11(P11 = app_v4.py 구조 분리, R8 바닥 원인)

---

## Previous Session (2026-11-06 ~ 11-07)

### Task: CI/CD run via temporary public visibility (2026-11-07) ✅ COMPLETE
- 사용자 방식: GitHub Actions 결제 한도 문제를 피해, 저장소를 잠시 공개로 바꿔 CI/CD를 돌린 뒤 다시 비공개로 돌린다. 키 노출을 알리자 사용자는 "지금 바로 진행"을 골랐다.
- 공개 시간: 07:02:34 ~ 07:11:25(약 9분), 그동안 fork 0개, 지금은 PRIVATE
- 1838cf8f 재실행 결과:
  - CI: 8개 작업 모두 성공(Python 3.10·3.11·3.12, Web v7, Lint, Security Scan, API Tests, Build Verification)
  - CD: 성공(GHCR 이미지 빌드·푸시, Staging은 자리표시자, Production은 건너뜀)
- 남은 일:
  - data.go.kr API 키 재발급. 9분 공개된 위치는 CLAUDE.md, docs/CONVERSATION_PROMPTS.md, git 이력이다. 재발급 뒤 두 파일의 키를 자리표시자로 바꾸고 키를 쓰는 서비스를 재시작한다.
  - GHCR 패키지 공개 여부 확인. gh 토큰에 read:packages가 필요하다. 이미지에는 .dockerignore로 docs/·*.md·.env가 빠져 키가 없다.
- 참고: CI/CD의 paths-ignore는 루트 *.md와 docs/**만 제외한다. .claude/·docker/ 아래가 바뀌면 워크플로가 실행되고, 지금은 결제 문제로 작업이 시작되지 않는다.

### Task: Session memory + doc refresh (2026-11-07) ✅ COMPLETE
- 사용자 요청: "지금까지 대화내용과 작업내용 메모리에 기억", "업데이트된 내용 관련 모든 문서 최신 업데이트", "커밋, 푸시"
- 메모리(~/.claude/projects/-Users-ibkim-Projects-power-demand-forecast/memory/): 사용자 선호·승인 방식·보안 규칙·ar_gbm 전환 경위·공개 페이지·경쟁 비교·미해결 과제·참조 주소·작업 요약, 12개 파일과 MEMORY.md 색인
- 문서 갱신:
  - README(현재 운영 모델 성능 표, 멘토 자료 목록, Docker 현재 운영), CHANGELOG(2026-11-05~07), CLAUDE.md(테스트 수, 경쟁 비교 위치, LSTM은 선택형)
  - PROGRESS_REPORT, docs/2026-11-02-forecast-upgrade-status.md(갱신 절), RE-BMS v5/v6 기술문서·TECHNICAL_DOCUMENTATION(옛 6.06% → 현재 수치, 현재 API 예시)
  - docker/README(v7 API 운영), DEPLOYMENT_GUIDE, 재학습 보고서 머리말
- 그대로 둔 것: 당시 기록인 문서(CONVERSATION_PROMPTS, SESSION_SUMMARY_2025-01-10, CLAUDE.md의 2025-12 실험 표), `.claude/worktrees/`의 별도 작업 사본

### Task: Mentor explainer correction + competitor report update ✅ COMPLETE
- docs/mentor-tech-explainer.html (공개 페이지): 낡은 수치 51곳 정정(6.06% → ar_gbm 2025년 4.67%, 옛 LSTM 재채점 9.78%), 최종 검토 14건 반영(수요 모델은 공개 웹 화면에 미연결, v4 '예측 수요'는 모의 곡선, SMP CatBoost v3.19 행 추가), 본문 폭 880→1560px, 600~1728px 넘침 없음
- 근거: docs/benchmark-2026-11-06/rescore_old_lstm.py(+_out.json), compare_default_2025_out.json
- docs/2026-11-06-competitor-benchmark.md: 수요 부분에 ar_gbm 갱신 상자(고정 test 3.63% / 0.954, 2025년 4.67%), 경쟁자 판정은 그대로 '판정 불가'
- 같은 보고서 1.1절(사용자 요청 "경쟁 업체 이름과 실제 데이터도 같이 표시"): 경쟁사 이름·실제 수치와 같은 기간 ar_gbm 수치를 수요·SMP·태양광 표로 나란히 정리. 결론 요약 표에도 이름·수치 추가
  - ar_gbm 맞춘 수치: 2024년 하반기 3.79% / R² 0.963(KPX jlfd 4.67%·jlfdRt 3.21%와 같은 기간), 2022년 3.11% / 0.957, 2023년 R² 0.943, AR 대비 배율 0.96~0.98
  - 재현: docs/benchmark-2026-11-06/competitor_same_basis.py → competitor_same_basis_out.json (로컬 예측 models/demand_ar_gbm_2026-11-06/ 필요)
- 커밋: 892f45d3, 526d57b4, fa1bc2d8, 00c26ab0(1.1절), 27d90c16(설명 페이지 05장), 모두 푸시함
- 설명 페이지 05장 "경쟁 비교" 신설(사용자 승인): 경쟁사 10행 표(수요 5·SMP 3·태양광 2)와 주의 상자(수요는 거래량 대 계통수요라 순위 없음), 기존 05·06장은 06·07로, 용어 사전에 계통수요/전력거래량·MAE 추가. 600~1728px 14개 폭 넘침 없음(scratchpad width_check/compare_check.py)

### Task: Default switch to ar_gbm + holiday table 2026-2027 + Docker API + root test fix ✅ COMPLETE

- 사용자 결정: 기본값 교체 승인(2025 실측에서 이길 때만), 공휴일 표 공식 날짜로 연장, Docker는 API 컨테이너만, 실패 테스트 파일 삭제
- 2025 실측: tools/fetch_kpx_jeju_trade.py → data/raw/jeju_hourly_power_2025.csv (data.go.kr 15133498, 8,760시간, 학습 원본과 같은 계열 확인)
- 2025 시간외 비교(docs/benchmark-2026-11-06/compare_default_2025.py): ar_gbm 4.67% / R² 0.929, LSTM demand_only 5.22% / 0.910, AR 4.78%, ramp7 5.16%, persistence 6.16%. ar_gbm vs LSTM MAE −10.5%, DM −15.39, 12/12개월 우위
- API: DEFAULT_MODEL=ar_gbm(검증·strip), model_type Optional(생략=기본값), ar_gbm 불가·337h 미만이면 demand_only 대체 + model_used 표시(demand_only도 336h가 필요해 336h 미만은 400), /models는 실제 기본값
- 공휴일: LUNAR_HOLIDAYS 2026·2027 추가(월력요항, 독립 검증). 대체공휴일·선거일·노동절·제헌절은 학습 기준대로 제외
- Docker: Dockerfile.api → requirements-docker.txt, 버전 고정(pandas 2.3.3 등), compose v7 api에 수요 모델 마운트·mem_limit 1g. rebms-api :8506 운영 중(482 MiB), 운영 API와 값 동일
- 테스트: tools/crawlers/test_jeju_crawlers.py 삭제(실패 3건), 새 테스트 9건, 루트 pytest 2271 passed / 3 skipped / 실패 0
- 운영 API(launchd :8001) 16:13 재시작: 생략 요청 ar_gbm 539.88 = 로컬, 336h 요청은 demand_only 대체
- 보고서: docs/2026-11-06-demand-retrain-raw-target.md 12절

### Next Steps (after default switch)
- 2026년 실측 공개(2027-07-16 예정) 뒤 같은 비교 재실행, 필요하면 ar_gbm을 2025년까지 포함해 재학습
- 대체공휴일·선거일·노동절·제헌절 반영은 과거 연도 기준 변경 + 재학습·게이트가 필요(별도 결정)
- Streamlit 대시보드는 168·200시간만 보내 이미 모든 예측이 400(별도 수정 필요), scripts/test_air_crawler.py는 실제 키로 외부 호출

### Task: Demand model improvement (raw target, AR/ramp7 gate, AR + correction) ✅ EXPERIMENTS COMPLETE · production unchanged

- 보고서: `docs/2026-11-06-demand-retrain-raw-target.md` · 스크립트: `docs/benchmark-2026-11-06/{compare_demand_retrain,diagnose_ar_residual,report_tables,probe_ar_anchor_collapse}.py`
- 커밋: 5665fac0(클리핑 제거·CSV 919시간 복원), e120d68e(게이트: persistence·ramp7·시간대별 AR, DM p<0.05, MAE 1% 개선, `--anchor ar`)
- 결과(demand_only, 원본 목표, 고정 test 2023-07~2024-12 / walk-forward 2021~2024 평균 MAPE):
  - 운영 LSTM(서빙 경로 재현, 원본 입력) 4.16% · 직전값 앵커 원본 재학습 4.17% / 3.66% 불합격 · AR 앵커 LSTM 3.78% / 3.45% 불합격(학습 붕괴, AR + 상수)
  - AR + GBM 잔차 보정(B) 3.63% / 3.35% **합격**(기준선 3종, 4/4 연도, AR 대비 MAE −4.0% / −2.7%, 서빙 모델 대비 −13%)
  - 시간대별 AR 3.78% / 3.45%
- 정정: 경쟁 비교 보고서의 운영 LSTM 4.31%·"최대 수요 53 MWh 과소예측"은 2026-11-05에 저장된(잘린 입력) 예측을 채점한 착오 → 서빙 경로 4.16% / R² 0.939, 상한 위 631시간 bias −5.9 MWh. 같은 날 보고서 정정
- AR 앵커 붕괴: 원래 체크포인트는 마지막 은닉 ReLU가 모든 test 입력에서 0 → 보정량 = 출력 편향(상수). 잔차 표준화·LeakyReLU(2~3에폭, 시드 1개)로도 학습 손실 불변. GBM은 비슷한 정보로 잔차를 학습했지만 LSTM 실패 원인은 미확정
- 산출물(git 미추적): 예측 CSV(models/demand_retrain_2026-11-06_raw_{last,ar}/, models/demand_ar_gbm_2026-11-06/). 두 실행의 oot_report.json은 커밋

### Task: AR + GBM formalization (training pipeline, serving path, opt-in API) ✅ COMPLETE · default unchanged

- 사용자 결정: 서빙 산출물은 최신 분할 절차(학습 ~2024-06, 검증 2024H2로 트리 수 선택), API는 선택형 `model_type="ar_gbm"`
- 코드: src/models/ar_gbm.py, src/training/train_demand_ar_gbm.py, src/inference/ar_gbm_predictor.py, api/{service,main,schemas,config}.py, requirements(scikit-learn==1.7.2)
- 산출물: models/demand_ar_gbm/ (375그루, 검증 MAE 18.60 vs AR 19.50)
- 검증: 파이프라인 = 진단 스크립트(5개 분할 예측 차이 0), 서빙 경로 = 학습 경로(실데이터 5개 시각 차이 0, 최소 337시간), tests/ 2248 passed
- 독립 리뷰 반영: 게이트 통과 산출물만 서빙, sha256·원자적 쓰기, 버전·상수 검사, 503 응답에서 경로 숨김, sklearn 지연 import, 휴일 표 범위 밖 학습 거부
- 한계: 음력 공휴일 표 2025년까지(2026년 설·추석 누락, LSTM도 동일), Docker 미빌드·compose 마운트 없음
- 운영 API(launchd com.powerdemand.api, :8001) 2026-11-06 12:45 재시작 → /models에 ar_gbm, 실제 /predict ar_gbm 520.74 = 학습 경로, demand_only 525.65, 기본값 LSTM 유지

### Next Steps (demand) — 2026-11-06 오후 처리됨 (위 "Default switch" 참고)
- ~~운영 기본값을 ar_gbm으로 바꿀지~~ → 2025 실측 비교 후 교체 완료
- ~~음력 공휴일 표 2026년 이후 추가~~ → 2027년까지 추가

### Task: Competitor benchmark (MAPE/R² vs operators, companies, papers) ✅ COMPLETE

- 보고서: `docs/2026-11-06-competitor-benchmark.md` · 재현 스크립트: `docs/benchmark-2026-11-06/verify_{ar,smp,solar}.py`
- 수요 LSTM: 같은 목표(KPX 제주 전력거래량) 경쟁자 없음 → 판정 불가. 단 원본 목표 기준 4.31%/R² 0.926으로 시간대별 선형 AR(3.75%/0.951)에 고정 test·walk-forward 4/4 연도 모두 짐 (DM +10.65) — 2026-11-06 정정: 서빙 경로 4.16%/0.939, AR 대비 DM +14.21 (위 Demand model improvement 참고)
- 수요 목표가 694.658 MWh에서 잘려 있음(test 포함 전체 기간 Q3+1.5·IQR, `src/data/preprocessing.py` handle_outliers) → 기존 4.03%/0.937은 외부 보고 금지
- SMP: 서빙 naive(MAE 11.97, 2025H2 walk-forward)는 vpp-capstone LightGBM(naive 대비 −17%, 2025)·VOLTA(−10~27%, 2024)보다 나쁨. 0.5×어제 + 0.5×7일 중앙값 혼합이 MAE −8.8~9.6%
- 태양광(RE-BMS): KPX 오차율 17.0%로 월×시간 기후값 12.5%보다 나쁨. "R²=0.95"는 무작위 분할·전체 fit 스케일러·측정 일사량 평가 수치
- CLAUDE.md 정정: SMP v4.1 8.76%는 육지 SMP 목표 수치, 현재 생산은 naive
- 운영: 2026-11-06 01:04부터 제주 SMP API 429(조사 작업의 대량 수집 직후 일일 한도 소진) → 00:58 마지막 검증 가격으로 폴백 중. 실패 시 backoff 없음(약 30초마다 재호출)

### Next Steps (competitor benchmark)
- 수요: 클리핑 제거 후 원본 목표 재학습, 배포 게이트에 AR·7일 ramp 추가
- SMP: 혼합 규칙 서빙 검토 → LightGBM(D-1 11:00 정보만)
- 태양광: "R²=0.95" 라벨 제거, 시간순 재학습 + 기상청 예보 입력, KPX 규칙 평가
- SMP API 캐시(일일 한도 보호)

### Task: Demand LSTM retrain + out-of-time validation ✅ COMPLETE

- 기존 production 수요 LSTM은 리포트 MAPE 6.06%를 재현 못 함(재현 10.05%), persistence(t-1) 5.12%보다 나쁨
- 직접 타깃 재학습(7.74%/8.46%)도 persistence 미달 → 중단
- 잔차 타깃 LSTM(`LastValueResidual`)으로 재학습: demand_only test 4.03%/R² 0.937, weather_full 4.14%/0.933, walk-forward 2021~2024 평균 3.6% (persistence 4.49%), 4/4 fold 우위
- models/production 교체(이전 버전 백업: models/_pre_retrain_backup_2026-11-05/demand_lstm/), API(launchd com.powerdemand.api, :8001) 재시작 후 /predict가 학습 예측과 일치 확인
- 코드: src/training/retrain_demand_oot.py, src/models/lstm.py(LastValueResidual), src/inference/predict.py(residual_target 래핑), train_production.train_model(model_wrapper 인자), tests/test_retrain_demand_oot.py

### Next Steps
- 다중 시점(24h) 예측에도 잔차 구조 적용 검토
- tools/generate_model_figures.py 그림 재생성 (새 training_report.json 스키마 호환)

---

## Current Session (2025-01-10)

### Task: SMP Model Version History & Documentation ✅ COMPLETE

#### Completed Work

1. **SMP 모델 버전 이력 이미지 생성**
   - 파일: `results/smp_model_version_history.png`
   - 24개 SMP 모델 버전 (v1.0 ~ v5.0) 정리
   - MAPE, R², 학습일자, 모델 아키텍처, 설명 포함

2. **SMP 모델 학습 코드 위치 정리**
   - 디렉토리: `src/smp/models/`
   - 주요 파일: `train_smp_v3_12_stacking.py` (BEST), `train_smp_v3_19_recent.py` (Production)

3. **GitHub 저장소 Private 변경**
   - `gh repo edit --visibility private` 실행
   - 상태: ✅ PRIVATE

4. **세션 요약 저장**
   - 파일: `.claude/backups/SESSION_SUMMARY_2025-01-10.md`

#### SMP Model Summary

| 버전 | MAPE | R² | 상태 |
|------|------|-----|------|
| v3.12 Stacking | 5.25% | 0.834 | ⭐ BEST |
| v3.19 CatBoost | 5.28% | 0.827 | ⭐ Production |

---

## Previous Session (2025-12-28)

### Task: CatBoost v3.19 Documentation & Volatility Analysis ✅ COMPLETE

#### Completed Work

1. **CatBoost v3.19 Deployment**
   - Updated `smp_catboost_predictor.py` with v3.19 version
   - Updated `api/smp_routes.py` with v3.19 metrics
   - Model: MAPE 5.28%, R² 0.827

2. **Colab Notebooks Created**
   - `notebooks/CatBoost_v3_19_Validation_Colab.ipynb` - Model verification
   - `notebooks/CatBoost_v3_19_Full_Pipeline_Colab.ipynb` - Train/Val/Test pipeline

3. **Volatility-Based Performance Analysis**
   | Volatility | Samples | MAPE |
   |------------|---------|------|
   | Normal (<10%) | 97.6% | 2.21% |
   | High (10-20%) | 1.9% | 8.17% |
   | Very High (20-30%) | 0.3% | 13.69% |
   | Extreme (>30%) | 0.2% | 40.68% |

4. **Colab vs Original Metrics Difference Explained**
   - Original: 2024-12-16 ~ (367 days, 8,741 samples, 6.36% volatility)
   - Colab: 2025-03-18 ~ (275 days, 6,556 samples, 6.04% volatility)
   - Original includes volatile winter period (Dec-Feb)
   - `smp_lag1` dominance (61.59%) → high accuracy in stable periods

5. **Technical Documentation Updated**
   - Added Section 9: SMP 예측 모델 (CatBoost v3.19)
   - Version updated to 6.0.0
   - Committed: `c6b8d5d7`

6. **Unit Tests Added**
   - 7 new CatBoost-specific tests in `tests/test_smp.py`
   - All 24 tests passed

#### Key Findings

- Model relies heavily on `smp_lag1` (61.59% importance) → "Naive Forecast" behavior
- Works excellently in stable markets (MAPE ~2%), fails in volatile periods (MAPE ~40%)
- Different test periods explain Colab (0.99%) vs Original (5.28%) MAPE difference

---

## Previous Session (2025-12-26)

### Task: Mobile App v7.0.0 Cross-Platform Edition ✅ COMPLETE

#### Completed Work

1. **ExecoDashboardScreen** - Main command center
   - Real-time KPIs (Demand, SMP, Renewable %, Frequency, Weather)
   - API status indicators (SMP, Power, Weather)
   - Power supply visualization
   - Plant statistics (Solar, Wind, ESS)

2. **BiddingScreen** - Enhanced with web-v7 features
   - Hour selection (00:00 - 23:00)
   - Risk level selection (Conservative/Moderate/Aggressive)
   - 10-segment bidding curve
   - Bid status workflow (Draft → Approved)
   - DAM/RTM simulation navigation

3. **SettlementScreen** - Enhanced with web-v7 features
   - Period selector (1 week/1 month/quarter)
   - DA/RT SMP info cards
   - Current SMP card with high/low
   - Revenue chart and transaction history

4. **App.tsx** - Updated navigation
   - 5-tab structure: Dashboard, SMP, Bidding, Portfolio, Settlement
   - Error boundary for stability
   - Alan AI floating button
   - DAM/RTM simulation screens

5. **Package.json** - Updated to v7.0.0
   - Cross-platform support (iOS, Android, Web)
   - 100% feature parity with web-v7

#### Build Status
- ✅ Web build successful (1.09 MB bundle)
- ✅ TypeScript compilation passed
- ✅ All screens render correctly

---

### Previous Task: Real-time API Integration ✅ COMPLETE

#### Completed Work

1. **제주시범사업 SMP API Integration**
   - Endpoint: `https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2`
   - Returns: Real-time Jeju SMP (원/kWh) + 수요예측량 (MW)
   - Status: ✅ Connected (84.3원/kWh working)

2. **KMA Weather API Integration**
   - Endpoint: `https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst`
   - Returns: Temperature, Humidity, Wind Speed
   - Status: ✅ Connected (2°C working)

3. **Docker Deployment Fixed**
   - Fixed `.dockerignore` (tools directory now included)
   - Fixed `Dockerfile.api` (added `COPY tools/ ./tools/`)
   - Fixed crawler imports (individual try/except)
   - Status: ✅ All containers running

4. **Data Source Priority Configuration**
   | Data Type | Priority 1 | Priority 2 |
   |-----------|------------|------------|
   | SMP | 제주시범사업 API | Crawler |
   | Power Supply | Crawler | - |
   | Weather | KMA API | Crawler |

5. **Git Commit & Push**
   - Commit: "feat: Add Jeju SMP API and improve data source priorities"
   - Commit: "docs: Update CLAUDE.md with Docker deployment and API integration docs"
   - Status: ✅ All pushed to GitHub

6. **Session Cleanup**
   - All background tasks killed (uvicorn, streamlit, vite, expo)
   - Docker containers stopped (rebms-api, rebms-web-v7, rebms-mobile)
   - Status: ✅ Clean state

#### Current Dashboard Status
```
✅ SMP: 84.3원/kWh (제주시범사업 API)
✅ Power Supply: 865MW, 73.4% reserve (Crawler)
✅ Weather: 2°C (KMA API)
✅ Frequency: 59.99Hz (Simulated)
```

---

## Previous Session: SMP Model Experiments

### Goal: Reach R² 0.9+ (Best achieved: v3.12 CatBoost CV R² 0.834)

#### Experiments Completed

| Model | Strategy | Test R² | Test MAPE | Result |
|-------|----------|---------|-----------|--------|
| v3.12 CatBoost CV | 5-fold CV baseline | 0.834 | 5.25% | **BEST** ✅ |
| v3.14 +Fuel | Fuel price features | 0.534 | 12.56% | ❌ FAILED |
| v3.15 +Supply | Supply margin features | 0.107 | 12.09% | ❌ FAILED |
| v3.16 +Weather | Weather features | 0.788 | 5.73% | ❌ FAILED |
| v3.18 Transform | Box-Cox target transform | 0.793 | 5.69% | ❌ FAILED |
| v3.19 +EMA | EMA + recent data | 0.827 | 5.28% | ❌ FAILED |
| v3.20 Recent | 2022+ data only | 0.779 | 7.75% | ❌ FAILED |

#### Key Findings

1. **External data doesn't help**:
   - Fuel prices: 0% feature importance, data mismatch (2022+ only)
   - Weather: <0.5% importance for all weather features
   - Supply margin: Not in top 20 features, only 33% data overlap

2. **smp_lag1 dominates everything**:
   - 40-80% feature importance in ALL models
   - Recent SMP history is the only reliable predictor

3. **Concept drift is severe**:
   - Fold 1 (2020-2021 COVID era) consistently has R² ~ -2.5
   - This drags down CV performance to ~0.1 for most models
   - v3.12's R² 0.834 benefits from Folds 3-5 having similar train/test periods

4. **Target transformation doesn't help**:
   - Box-Cox, log, sqrt all tested - none improved performance
   - The distribution is not the issue

5. **More recent data = worse performance**:
   - Using only 2022+ data: R² 0.779
   - Using only 2023+ data: R² 0.760
   - Less training data hurts more than concept drift

6. **The R² 0.9+ goal may be unreachable**:
   - SMP is inherently noisy and unpredictable
   - External factors not captured in available data
   - v3.12's 0.834 is likely near the ceiling for this feature set

#### Files Created This Session

```
src/smp/models/train_smp_v3_14_fuel.py       - Fuel price features (FAILED)
src/smp/models/train_smp_v3_15_supply.py     - Supply margin features (FAILED)
src/smp/models/train_smp_v3_16_weather.py    - Weather features (FAILED)
src/smp/models/train_smp_v3_18_transform.py  - Target transformation (FAILED)
src/smp/models/train_smp_v3_19_recent.py     - EMA + recent data (FAILED)
src/smp/models/train_smp_v3_20_recent_only.py - Recent data only (FAILED)

models/smp_v3_14_fuel/
models/smp_v3_15_supply/
models/smp_v3_16_weather/
models/smp_v3_18_transform/
models/smp_v3_19_recent/
models/smp_v3_20_recent/
```

#### Recommendations for Future Work

1. **Accept v3.12 as production model**:
   - R² 0.834, MAPE 5.25% is a solid result
   - Significant external data doesn't improve it

2. **Focus on other improvements**:
   - Confidence intervals / uncertainty quantification
   - Hourly prediction for different time horizons
   - Model monitoring and drift detection

3. **Alternative approaches (if R² 0.9+ still needed)**:
   - Real-time market data (if available)
   - Transmission constraints and grid topology
   - Bidding behavior patterns from market participants

---

## Model Comparison (Full History)

| Model | MAPE | R² | Features | Strategy |
|-------|------|-----|----------|----------|
| v3.2 (Optuna) | 7.42% | 0.760 | 22 | BiLSTM+Attention |
| v3.3-v3.6 | - | < 0.6 | - | Generation data (failed) |
| v3.7 | 9.21% | 0.402 | 26 | Long-term MA (failed) |
| v3.8 LightGBM | 5.46% | 0.815 | 36 | LightGBM default |
| v3.9 Optuna LGB | 5.49% | 0.821 | 60 | Optuna-tuned LightGBM |
| v3.10 CatBoost | 5.38% | 0.826 | 60 | Optuna-tuned CatBoost |
| v3.11 XGBoost | 5.39% | 0.822 | 60 | Optuna-tuned XGBoost |
| **v3.12 CatBoost CV** | **5.25%** | **0.834** | 60 | **5-fold CV best** ✅ |
| v3.14 +Fuel | 12.56% | 0.534 | 70 | Fuel prices (FAILED) |
| v3.15 +Supply | 12.09% | 0.107 | 77 | Supply margin (FAILED) |
| v3.16 +Weather | 5.73% | 0.788 | 75 | Weather (FAILED) |
| v3.18 Transform | 5.69% | 0.793 | 60 | Box-Cox (FAILED) |
| v3.19 +EMA | 5.28% | 0.827 | 67 | EMA features (FAILED) |
| v3.20 Recent | 7.75% | 0.779 | 60 | 2022+ data (FAILED) |

---

## Key Files

### SMP Models
```
models/smp_v3_12_stacking/      - v3.12 CatBoost CV (BEST: R² 0.834, MAPE 5.25%)
models/smp_v3_10_catboost/      - v3.10 CatBoost (R² 0.826, MAPE 5.38%)
models/smp_v3_9_optuna_lgb/     - v3.9 Optuna LGB (R² 0.821)
models/smp_v3_8_ensemble/       - v3.8 LightGBM (R² 0.815)
src/smp/models/train_smp_v3_12_stacking.py  - v3.12 source code
```

### Crawlers
```
src/crawlers/public_data_crawler.py    - data.go.kr search
src/crawlers/download_public_data.py   - data.go.kr download
src/crawlers/epsis_crawler.py          - EPSIS SMP data
```

### web-v7 (Current Production)
```
web-v7/src/pages/ExecoDashboard.tsx  - Main dashboard
web-v7/src/hooks/useApi.ts          - API hooks
```

---

## Environment
- Python 3.13, PyTorch 2.0+
- Node.js 20, React 18, TypeScript
- Apple Silicon MPS (M1 MacBook Pro 32GB)
- Docker Desktop

---

## Session Recovery

For next session:
1. Read `.claude/backups/PROJECT_STATUS.md`
2. Run `git log --oneline -10`
3. **DOCKER**: `docker compose -f docker/docker-compose.v7.yml up -d --build --wait api` (API only: rebms-api on localhost:8506; the launchd API runs on 127.0.0.1:8001)
4. **DASHBOARDS** (not deployed as of 2026-11-06; start with `up -d --build web-v7 web-v6 mobile` only if needed):
   - ExecO Dashboard: http://localhost:8700
   - Bidding Dashboard: http://localhost:8600
   - Mobile App: http://localhost:8601
5. **API KEY**: `DATA_GO_KR_API_KEY=7d42f7c08...` (set in .env or Docker environment)
6. **BEST MODEL**: v3.12 CatBoost CV (R² 0.834, MAPE 5.25%)
7. **ALL APIS CONNECTED**: SMP (제주시범사업), Weather (KMA), Power (Crawler)
