# AGENTS.md - Power Demand Forecast

> AI Agent Context for Jeju Power Demand Forecasting System

## Project Overview

- **Domain**: Electricity demand forecasting for Jeju Island, Korea
- **Stack**: Python 3.11+, PyTorch (MPS), FastAPI, Streamlit
- **Hardware**: Apple M1 MacBook Pro 32GB (MPS priority)

## Operational Commands

### Build & Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run API Server
```bash
# Development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Single test file
pytest tests/test_lstm.py -v

# Single test function
pytest tests/test_lstm.py::TestLSTMModel::test_model_creation -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Fast subset (skip slow tests)
pytest tests/ -v -m "not slow"
```

### Lint & Format
```bash
# Check formatting
black --check --diff src/ api/ tests/

# Fix formatting
black src/ api/ tests/

# Check imports
isort --check-only --diff src/ api/ tests/

# Fix imports
isort src/ api/ tests/

# Lint (syntax errors only)
flake8 src/ api/ --select=E9,F63,F7,F82

# Full lint
flake8 src/ api/ --max-complexity=15 --max-line-length=120
```

### Docker
```bash
# Build & run
docker-compose up -d

# Rebuild
docker-compose up -d --build

# v6 dashboard
docker-compose -f docker/docker-compose.v6.yml up -d
```

### Dashboard
```bash
# Streamlit v4 (main)
# Dev only: production .env has alerts on, and production 8504 runs under launchd com.powerdemand.dashboard
SLACK_ALERTS_ENABLED=false EMAIL_ALERTS_ENABLED=false PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8514
# Restart production after a merge: launchctl kickstart -k gui/$(id -u)/com.powerdemand.dashboard
# UI gates after changes (docs/2026-11-23-dashboard-ui-top1-design.md sec 9.2/14.7):
#   .venv/bin/python -B -m pytest tests/test_v4_*.py -q -p no:cacheprovider
#   .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/ui_guard.py --app src/dashboard/app_v4.py
#   .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/logic_guard.py --before <pre-redesign app_v4.py> --after src/dashboard/app_v4.py

# React v6
cd web-v6 && npm install && npm run dev
```

## Golden Rules

### Immutable Constraints
- **Korean comments**: Domain logic comments in Korean, technical docs in English
- **MPS first**: Always use Apple Silicon MPS when available
- **No data leakage**: Never process test data with train-fitted transformers
- **Seed fixing**: Always set random seeds for reproducibility

### Do's
- Use `get_device()` from `src/utils/device.py` for device selection
- Follow existing patterns in similar modules
- Add type hints to all function signatures
- Use Google-style docstrings
- Write tests for new functionality
- Use pandas linear interpolation for missing values

### Don'ts
- Never use `as any`, `@ts-ignore` for type suppression
- Never commit `.env` or credentials
- Never use `torch.cuda` directly (use `get_device()`)
- Never mix train/test data before splitting
- Never use magic numbers (define as constants)

## Code Style

### Imports (isort profile: black)
```python
# Standard library
import os
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Local
from src.models.lstm import LSTMModel
from src.utils.device import get_device
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `calculate_thi()` |
| Classes | PascalCase | `LSTMModel` |
| Constants | UPPER_SNAKE | `MAGNUS_A = 17.625` |
| Private | _leading | `_init_weights()` |
| Modules | snake_case | `weather_features.py` |

### Type Hints
```python
def predict(
    self,
    data: pd.DataFrame,
    horizon: int = 24,
    model_type: str = "lstm"
) -> Dict[str, Union[float, List[float]]]:
    """Predict power demand."""
    ...
```

### Docstrings (Google Style)
```python
def calculate_thi(temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    """
    Calculate Temperature-Humidity Index.
    
    Args:
        temp: Temperature in Celsius
        humidity: Relative humidity (0-100%)
        
    Returns:
        THI values (discomfort index)
        
    Example:
        >>> thi = calculate_thi(np.array([30]), np.array([80]))
        >>> print(thi)  # ~82.4 (uncomfortable)
    """
```

### Error Handling
```python
# Use specific exceptions
class InsufficientDataError(ValueError):
    """Raised when data length is insufficient."""
    pass

# Always provide context
raise InsufficientDataError(
    f"Need {required} samples, got {len(data)}"
)

# Log before raise
logger.error(f"Prediction failed: {e}")
raise PredictionError(f"Model inference failed: {e}")
```

### Device Management
```python
# Correct
from src.utils.device import get_device

device = get_device()
model = model.to(device)
data = data.to(device)

# Wrong - don't use directly
device = torch.device("cuda")  # Never assume CUDA
```

## Test Patterns

### Fixtures
```python
@pytest.fixture
def sample_data():
    """Standard test data fixture."""
    return torch.randn(32, 48, 38)

@pytest.fixture
def device():
    """Test device fixture."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Test Structure
```python
class TestLSTMModel:
    """LSTM model test suite."""
    
    def test_model_creation(self):
        """Test basic model instantiation."""
        model = LSTMModel(input_size=38)
        assert model.input_size == 38
    
    def test_forward_shape(self, sample_data, device):
        """Test forward pass output shape."""
        model = LSTMModel(input_size=38).to(device)
        output = model(sample_data.to(device))
        assert output.shape == (32, 1)
```

## Directory Context

| Path | Purpose |
|------|---------|
| `src/models/` | LSTM, Transformer, Ensemble models |
| `src/features/` | Feature engineering (weather, time, solar) |
| `src/training/` | Trainer, EarlyStopping, schedulers |
| `src/analysis/` | Anomaly detection, XAI, scenarios |
| `src/data/` | Dataset, preprocessing, merge |
| `api/` | FastAPI endpoints |
| `tests/` | pytest test suite (1500+ tests) |
| `tools/crawlers/` | Data crawlers (EPSIS, KMA, etc.) |

## CI/CD

GitHub Actions runs on push to `main`/`develop`:
1. **Lint**: black, isort, flake8 (syntax errors fatal)
2. **Test**: pytest with Python 3.11
3. **Coverage**: Report to codecov

## Common Patterns

### Data Pipeline
```
Raw Data → Validation → Interpolation → Feature Engineering → Scaling → Model
```

### Model Training
```python
trainer = Trainer(model, optimizer, criterion, device)
trainer.fit(
    train_loader, 
    val_loader,
    epochs=100,
    early_stopping=EarlyStopping(patience=15)
)
```

### Time Series Split
```python
# Always use temporal order
train, val, test = time_series_split(data, ratios=[0.7, 0.15, 0.15])
# Fit scaler on train ONLY
scaler.fit(train)
train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)
```
