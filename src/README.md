# How to use?

```bash
python pp.py json/01.json
```

# Directory structure

```bash
# tree
.
├── README.md
├── json
│   ├── duffing
│   │   └── 01.json
│   └── forced_van_der_pol
│       └── 01.json
├── ode_collection.py
├── pp.py
└── pptools.py
```

# JSON structure

```json
{
    "system": "duffing",
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
