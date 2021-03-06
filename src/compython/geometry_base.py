import math
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans


def point_line_side(points, line_start, line_end):
    left_side = []
    right_side = []
    on_side = []

    line_start_x = line_start[0]
    line_start_y = line_start[1]

    line_end_x = line_end[0]
    line_end_y = line_end[1]
    x_new = points[:, 0]
    y_new = points[:, 1]

    for i in range(np.alen(x_new)):
        d = (
                (x_new[i] -
                 line_start_x) * (line_end_y -
                                  line_start_y) - (y_new[i] -
                                                   line_start_y) * (line_end_x -
                                                                    line_start_x))
        if d > 0:
            left_side.append([x_new[i], y_new[i]])
        if d == 0:
            on_side.append([x_new[i], y_new[i]])
        if d < 0:
            right_side.append([x_new[i], y_new[i]])

    return np.array([left_side, right_side, on_side])


def min_bound(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.
    :param points: a nx2 matrix of coordinates
    :rval: a nx2 matrix of coordinates
    """

    pi2 = np.pi / 2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles - pi2),
        np.cos(angles + pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def get_clusters(data, labels):
    """
    :param data: The dataset
    :param labels: The label for each point in the dataset
    :return: List[np.ndarray]: A list of arrays where the elements of each array
    are data points belonging to the label at that ind
    """
    return [data[np.where(labels == i)] for i in range(np.amax(labels) + 1)]


def p_kmeans(f, p: int):
    """
    p_function:  this function with parameter, supports layer-by-layer analytics
    :param f:  some function, in case: a nx2 matrix of coordinates
    :param p: p - parameter, in function - clusters count
    :return: kmeans (sklearn.cluster.KMeans) , you can get :
    kmeans.labels_,
    cluster_centers_
    more info in the scikit-learn documentation
    """
    return KMeans(n_clusters=p, random_state=0).fit(f)


class Domain:

    def __init__(self, start=float, end=float):
        self.start = start
        self.end = end
        self.length = end - start

    def __str__(self):
        return f'Domain: {self.start} to {self.end}'

    def divide_float(self, parts):
        rng = []
        for i in parts:
            rng.append(self.length * i + self.start)
        return rng

    def divide_steps(self, steps):
        step = self.length / steps
        rng = []
        for i in range(steps):
            rng.append(self.start + (i * step))
        return rng


class UV:
    def __init__(self, u, v, _min, _max, _xf, _yf, _z):
        self._u, self._v, self.min, self.max, self.__xf, self.__yf, self.__zf = u, v, _min, _max, _xf, _yf, _z
        self.x = self.array()[:, 0]
        self.y = self.array()[:, 1]
        self.z = self.array()[:, 2]

    def array(self):
        _xyz = []
        for i in self._u:
            for j in self._v:
                _xyz.append([self.__xf(i, j), self.__yf(i, j), self.__zf(j)])

        return np.array(_xyz)


class ParamFunc:

    def __init__(self, _min, _max, _xf, _yf, _zf):

        self.min, self.max, self.__xf, self.__yf, self.__zf = _min, _max, _xf, _yf, _zf

    def evaluate(func):
        def wrap(self, a, b):
            first, second = func(self, a, b)
            inst = UV(first, second, self.min, self.max, self.__xf, self.__yf, self.__zf)
            return inst

        return wrap

    @evaluate
    def multiply_interpolate(self, a: int, b: int):
        return list(np.linspace(self.min, self.max, a, True)), list(np.linspace(self.min, self.max, b, True))

    @evaluate
    def single_interpolate(self, a, b):
        if self.min <= a <= self.max and self.min <= b <= self.max:
            return [a], [b]
        else:
            print(f'ValueError: {a} &{b} out of range: {self.min} to {self.max}')