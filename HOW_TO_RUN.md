# How to Run kandc Examples

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Available Examples

### 1. Complete Example (Default/Online Mode)
Full-featured example with cloud sync and authentication:
```bash
python complete_example.py
```

### 2. Offline Mode Example
Run without internet - all data saved locally:
```bash
python offline_example.py
```

### 3. Disabled Mode Example
Zero-overhead mode for production deployment:
```bash
python disabled_example.py
```

### 4. No Code Sync Example
Full tracking without source code capture:
```bash
python no_code_sync_example.py
```

### 5. Ignore Code Example
Selective code capture with custom exclusion patterns:
```bash
python ignore_code_example.py
```

## Requirements

- Python 3.9+ (tested with Python 3.12.11)
- pip 25.1.1+
- PyTorch (installed via requirements.txt)

## What Each Example Shows

| Example                   | Mode     | Authentication | Code Capture | Profiling | Cloud Sync |
| ------------------------- | -------- | -------------- | ------------ | --------- | ---------- |
| `complete_example.py`     | Online   | Required       | ✅ Full       | ✅         | ✅          |
| `offline_example.py`      | Offline  | None           | ✅ Full       | ✅         | ❌          |
| `disabled_example.py`     | Disabled | None           | ❌            | ❌         | ❌          |
| `no_code_sync_example.py` | Online   | Required       | ❌            | ✅         | ✅          |
| `ignore_code_example.py`  | Online   | Required       | ✅ Selective  | ✅         | ✅          |