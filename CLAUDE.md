# Claude Code Instructions

## Backup Protocol

**IMPORTANT**: Update `.claude/backups/PROJECT_STATUS.md` after completing significant tasks:

1. **When to backup**:
   - After completing a major feature
   - After fixing critical bugs
   - Before ending a session
   - Every 30 minutes during long sessions

2. **What to backup**:
   - Current task status
   - Completed items
   - Next steps
   - Recent commits
   - Any blockers or issues

3. **Backup command**:
   ```bash
   # Update PROJECT_STATUS.md with current progress
   ```

---

## Project Context

### 제주도 전력 수요 예측 시스템

**Tech Stack**:
- Backend: Python 3.13, PyTorch, FastAPI
- ML Models: LSTM, BiLSTM, TFT, Ensemble
- Frontend: Streamlit (in progress)
- Database: File-based (CSV, Parquet)

**Key Directories**:
```
/src/           - Source code
/api/           - FastAPI server
/tests/         - Test suite (1,423 tests)
/models/        - Trained models
/data/          - Datasets
/results/       - Analysis results
/.claude/       - Claude backups
```

**Model Performance**:
- MAPE: 6.32%
- R²: 0.852

---

## Session Recovery

If conversation is lost, read:
1. `.claude/backups/PROJECT_STATUS.md` - Current status
2. `CHANGELOG.md` - Version history
3. `README.md` - Project overview
4. `git log --oneline -20` - Recent commits

---

## Coding Standards

- Korean comments for domain logic
- English for technical documentation
- Follow existing code patterns
- Run tests after major changes
- Commit frequently with descriptive messages
