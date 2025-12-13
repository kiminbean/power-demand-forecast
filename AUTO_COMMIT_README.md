# 🤖 자동 커밋 시스템 사용 가이드

제주도 전력수요 예측 프로젝트에 자동 커밋 시스템이 설정되었습니다.

## 📋 개요

이 시스템은 ML/DL 프로젝트의 재현성과 안정성을 높이기 위해 다음 항목들을 자동으로 커밋합니다:

- ✅ **코드 변경사항** (`src/`, `tests/`, `scripts/`)
- ✅ **처리된 데이터** (`data/processed/`, `data/features/`)
- ✅ **실험 로그** (`logs/`)
- ✅ **실험 결과** (`results/`)
- ✅ **Best 모델 체크포인트** (`models/*_best.pt`)
- ✅ **설정 파일** (`.json`, `.txt`, `.md`, `.gitignore`)

## 🚀 사용 방법

### 1. 수동 실행 (권장)

작업 후 변경사항을 바로 커밋하고 싶을 때:

```bash
./commit.sh
```

### 2. 자동 주기 실행 (30분마다)

launchd를 사용한 자동 실행 설정:

```bash
# 설치 및 활성화
./setup-autocommit.sh install

# 상태 확인
./setup-autocommit.sh status

# 제거
./setup-autocommit.sh uninstall
```

## 📁 프로젝트 구조

```
power-demand-forecast/
├── data/
│   ├── raw/           # Raw 데이터 (gitignore에서 제외)
│   ├── processed/     # 처리된 데이터 (자동 커밋)
│   └── features/      # 피처 정의 (자동 커밋)
├── src/               # 소스 코드 (자동 커밋)
│   ├── features/
│   ├── training/
│   ├── analysis/
│   └── utils/
├── tests/             # 테스트 코드 (자동 커밋)
├── logs/              # 실험 로그 (자동 커밋)
├── results/           # 실험 결과 (자동 커밋)
├── models/            # 모델 체크포인트 (best 모델만 자동 커밋)
└── notebooks/         # Jupyter 노트북 (gitignore에서 제외)
```

## 🔧 .gitignore 설정

다음 항목들은 자동으로 제외됩니다:

- 🚫 대용량 모델 파일 (best 모델 제외)
- 🚫 Raw 데이터 파일
- 🚫 가상환경 (`.venv/`)
- 🚫 Jupyter 노트북 (`.ipynb`)
- 🚫 IDE 설정 파일
- 🚫 캐시 및 임시 파일

## 📊 커밋 로그 확인

자동 커밋 실행 로그는 다음 위치에 저장됩니다:

```bash
# 정상 로그
tail -f logs/autocommit.log

# 에러 로그
tail -f logs/autocommit.error.log
```

## ⚙️ 커밋 메시지 형식

자동 커밋은 다음과 같은 형식을 따릅니다:

### 코드 변경
```
chore: Auto-commit code changes (N files)

Auto-committed at: 2025-12-14 12:00:00

Changes:
M  src/features/weather_features.py
A  src/utils/preprocessing.py

🤖 Generated with Claude Code Auto-Commit
```

### 실험 로그
```
logs: Auto-commit experiment logs

Auto-committed at: 2025-12-14 12:00:00

Changes:
A  logs/v22_output.txt

🤖 Generated with Claude Code Auto-Commit
```

### 실험 결과
```
results: Auto-commit experiment results

Auto-committed at: 2025-12-14 12:00:00

Changes:
A  results/v22_metrics.json

🤖 Generated with Claude Code Auto-Commit
```

## 🎯 Best Practices

1. **작업 후 즉시 커밋**: `./commit.sh` 실행
2. **실험 전후 커밋**: 실험 전에 한 번, 결과 확인 후 한 번
3. **의미있는 수동 커밋**: 중요한 마일스톤은 수동으로 커밋 메시지 작성
4. **주기적 push**: `git push` 명령으로 원격 저장소에 백업

## 🔄 워크플로우 예시

```bash
# 1. 새로운 피처 개발
vim src/features/new_feature.py

# 2. 자동 커밋 실행
./commit.sh

# 3. 모델 학습
python src/training/train_model.py

# 4. 학습 완료 후 자동 커밋
./commit.sh

# 5. 결과 분석
python src/analysis/analyze_results.py

# 6. 분석 완료 후 자동 커밋
./commit.sh

# 7. 원격 저장소에 푸시
git push
```

## ⚠️ 주의사항

- **대용량 파일**: 100MB 이상 파일은 Git LFS 사용 고려
- **민감 정보**: `.env`, `credentials.json` 등은 자동으로 제외됨
- **수동 검토**: 중요한 변경사항은 `git log`로 확인 후 push

## 🛠️ 문제 해결

### 자동 커밋이 작동하지 않을 때

```bash
# 1. 스크립트 실행 권한 확인
ls -la commit.sh .git/hooks/auto-commit.sh

# 2. 수동으로 권한 부여
chmod +x commit.sh .git/hooks/auto-commit.sh

# 3. Git 상태 확인
git status

# 4. 스크립트 직접 실행 및 디버깅
bash -x .git/hooks/auto-commit.sh
```

### launchd 서비스가 작동하지 않을 때

```bash
# 서비스 상태 확인
./setup-autocommit.sh status

# 서비스 재시작
./setup-autocommit.sh uninstall
./setup-autocommit.sh install

# 로그 확인
tail -f logs/autocommit.log logs/autocommit.error.log
```

## 📚 추가 리소스

- [Git 커밋 컨벤션](https://www.conventionalcommits.org/)
- [ML 프로젝트 구조 Best Practices](https://github.com/drivendata/cookiecutter-data-science)
- [Git LFS 가이드](https://git-lfs.github.com/)

---

**설정 완료일**: 2025-12-14
**설정자**: Claude Code Auto-Commit System
**문의**: 이슈가 있을 경우 프로젝트 관리자에게 문의하세요.
