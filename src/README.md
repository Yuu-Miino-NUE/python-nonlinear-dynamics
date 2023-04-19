# How to use?

```bash
python -m pp duffing/01.json
```

# Directory structure

```bash
# tree
.
├── duffing
│   ├── 01.json
│   ├── 02.json
│   └── system.py
├── forced_van_der_pol
│   ├── 01.json
│   └── system.py
└── pp
    ├── __init__.py
    ├── __main__.py
    ├── pp.py
    └── pptools.py
```

# JSON structure

```json
{
    "y0": [
        0,
        0
    ],
    "params": {
        "k": 0.2,
        "B": 0.2,
        "B0": 0.2
    }
}
```