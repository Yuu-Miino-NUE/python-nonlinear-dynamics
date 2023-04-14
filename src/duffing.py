"""
How to use?

```
python duffing.py 01.json
```
"""
import json, sys
from pptools import pp
from sympy import cos, pi

# Definitions
def ode_fun (t, state, params):
    x, y = state
    k = params['k']
    B = params['B']
    B0 = params['B0']

    return [
        y,
        -k * y - x**3 + B0 + B * cos(t)
    ]

# Main part
with open(sys.argv[1]) as f:
    data = json.load(f)

y0 		= data['y0']
params 	= data['params']

config = {
    'param_keys': ['B', 'B0'],
    'xrange': (-2, 2),
    'yrange': (-2, 2)
}

pp(ode_fun, 2*pi, y0, params, **config)
