# How to use?

```bash
python pp.py 01.json
```

# Directory structure

```bash
# tree
.
├── 01.json
├── 02.json
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
