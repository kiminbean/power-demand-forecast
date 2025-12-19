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

---

## CRITICAL: UTF-8 Crash Prevention (Claude Code v2.0.72+)

**FATAL BUG**: Claude Code CLI crashes when Korean text appears in UI elements.

### Technical Cause
```
Rust panic: byte index N is not a char boundary; it is inside '한글' (bytes X..Y)
```
- Korean characters = 3 bytes in UTF-8
- Rust slices by byte index, not character boundary
- UI truncation cuts mid-character → **IMMEDIATE CRASH**

### Real Crash Example (2024-12-18)
```
byte index 5 is not a char boundary; it is inside '화' (bytes 3..6) of `완화)`
fatal runtime error: failed to initiate panic, error 5, aborting
```
- String: "완화)" = 완[0-2] + 화[3-5] + )[6]
- Rust tried to slice at byte 5 (middle of '화') → PANIC

### Crash Triggers
1. **TodoWrite content/activeForm** with Korean
2. **Session history** containing Korean in API responses
3. **Code output** with Korean that gets truncated in status bar
4. **Error messages** with Korean in stack traces

### MANDATORY RULES

1. **TodoWrite tool - ENGLISH ONLY**
   ```json
   // ❌ CRASH: {"content": "모델 학습", "activeForm": "학습 중"}
   // ✅ SAFE:  {"content": "Train model", "activeForm": "Training"}
   ```

2. **All status/progress messages**: English only

3. **Avoid Korean in console output** that may appear in status bar

### Recovery Commands

```bash
# 1. Quick recovery (move todo files)
mkdir -p ~/.claude/todos_backup && mv ~/.claude/todos/*.json ~/.claude/todos_backup/

# 2. Full cleanup (if crashes persist)
rm -rf ~/.claude/todos/*.json
rm -rf ~/.claude/todos_backup/

# 3. Nuclear option (clear all session data)
rm -rf ~/.claude/projects/-Users-ibkim-Ormi-1-power-demand-forecast/
```

### Preventive Checks

```bash
# Check for Korean in todo files
grep -l '[가-힣]' ~/.claude/todos/*.json 2>/dev/null && echo "WARNING: Korean found!"

# Check all claude files for Korean
find ~/.claude -name "*.json" -exec grep -l '[가-힣]' {} \; 2>/dev/null
```

**Bug report**: https://github.com/anthropics/claude-code/issues

---

## Auto Commit Protocol

**IMPORTANT**: Automatically commit changes after completing each task:

1. **When to commit**:
   - After completing a feature or significant code change
   - After fixing a bug
   - After refactoring code
   - Before starting a new, unrelated task

2. **Commit message format**:
   ```
   <type>: <short description>

   <detailed description if needed>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)
   Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
   ```

3. **Commit types**:
   - `feat`: New feature
   - `fix`: Bug fix
   - `refactor`: Code refactoring
   - `docs`: Documentation changes
   - `test`: Test additions/changes
   - `chore`: Maintenance tasks

4. **Do NOT auto-push**: Only commit locally, let user decide when to push
