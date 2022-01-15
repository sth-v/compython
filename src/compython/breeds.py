from geometry_base import Point_3d
import numpy as np
import math
import inspect


def select_init_arg(**dwargs):

    def decorate_init(cls):

        return cls.__new__(**dwargs)

    return decorate_init


class Stass_Panel:

    @classmethod
    def sides_init(*args):
        a = Point_3d(0,0,0)
        b = Point_3d(args[0], 0, 0)
        m1 = np.array([[a.x, a.y],
                       [a.x, a.y]]
                      )
        v1 = np.array([args[1], args[2]])

        r = np.linalg.solve(m1, v1)
        vertex = list(r)
        print(vertex)

        return {'sides': args,
                'vertex': vertex}

    @classmethod
    def vertex_init(*args):
        sides = []
        for i, arg in enumerate(args):
            sides.append(arg.dist(args[i+1]))
        return {'sides': sides,
                'vertex':args}


    def __new__(cls, sides: None, vertex: None, **kwargs):
        _instance: cls

        if vertex:
            __sd, __vt = cls.vertex_init(vertex)
            print(f'call sides __init__ {__sd}')
            return super().__new__(cls, __sd, __vt, **kwargs)


        elif sides:

            __sd, __vt = cls.vertex_init(sides)
            print(f'call vertex __init__ {__vt }')
            return super().__new__(cls, __sd, __vt, **kwargs)

        else:
            print('f{no inits}')



    def __init__(self, sides: None, vertex: None, **kwargs):
        self.sides = sides
        self.vertex = vertex
        self.mx = self.to_mx()
        self.a = self.sides[0]
        self.b = self.sides[1]
        self.c = self.sides[2]



    def to_mx(self):
        mx = []
        for i, j in self.sides, self.vertex:
            mx.append([1, j.x, j.y, j.z])
        return mx

    def __str__(self):
        return f'panel {np.array(self.mx)}'

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c

    def __hash__(self):
        return hash((self.a, self.b, self.c))



