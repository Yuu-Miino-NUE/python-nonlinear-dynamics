"""
How to use?

```
python duffing.py 01.json
```
"""
import json, sys
from pptools import pp

from nonlinear_systems import duffing, duffing_config
from sympy import pi

# Main part
with open(sys.argv[1]) as f:
    data = json.load(f)

y0 = data["y0"]
params = data["params"]

pp(duffing, y0, params, tend=2 * pi, **duffing_config)
