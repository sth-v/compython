import math


_pi = math.pi

_TOR_PROPS = {
    '_min': -_pi,
    '_max': _pi,
    '_xf': lambda u, v: math.cos(u) * (math.cos(v) + 3),
    '_yf': lambda u, v: math.sin(u) * (math.cos(v) + 3),
    '_zf': lambda v: math.sin(v)
}
