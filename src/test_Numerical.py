import numpy as np
import scipy.sparse
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.special import jn, hankel1
from scipy.sparse.linalg import gmres as npgmres
import scipy
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import gmres


def judge(x, y):
    """
    x:  X-axis
    y: Y-axis
    """
    n = x.shape[1]
    index = 0
    for i in range(n):
        if i < n - 1:
            a = [x[0][i], y[0][i], x[0][i + 1], y[0][i + 1]]
        else:
            a = [x[0][i], y[0][i], x[0][0], y[0][0]]
        for j in range(n):
            if j < n - 1:
                b = [x[0][j], y[0][j], x[0][j + 1], y[0][j + 1]]
            else:
                b = [x[0][j], y[0][j], x[0][0], y[0][0]]
            if Intersection(a, b) == True:
                index = index + 1
    # if index == 3 * n - 5 and (x[0, 1] != x[0, -1] or y[0, 1] != y[0, -1]):
    #     # 涓嶈嚜鐩镐氦
    #     return False
    # if index == 3 * n - 3 and x[0, 1] == x[0, -1] and y[0, 1] == y[0, -1]:
    #     # 涓嶈嚜鐩镐氦
    #     return False
    if index == 3 * n:
        return False
    else:
        return True


def Intersection(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    if np.maximum(ax1, ax2) < np.minimum(bx1, bx2):
        return False
    if np.maximum(ay1, ay2) < np.minimum(by1, by2):
        return False
    if np.maximum(bx1, bx2) < np.minimum(ax1, ax2):
        return False
    if np.maximum(by1, by2) < np.minimum(ay1, ay2):
        return False
    if (
        np.cross([ax1 - bx1, ay1 - by1], [bx2 - bx1, by2 - by1])
        * np.cross([bx2 - bx1, by2 - by1], [ax2 - bx1, ay2 - by1])
        < 0
    ):
        return False
    if (
        np.cross([bx1 - ax1, by1 - ay1], [ax2 - ax1, ay2 - ay1])
        * np.cross([ax2 - ax1, ay2 - ay1], [bx2 - ax1, by2 - ay1])
        < 0
    ):
        return False
    else:
        return True


import cupy as cp


def judge_cupy(x, y):
    """
    x:  X-axis
    y: Y-axis
    """
    n = x.shape[1]
    index = 0
    for i in range(n):
        if i < n - 1:
            a = cp.array([x[0][i], y[0][i], x[0][i + 1], y[0][i + 1]])
        else:
            a = cp.array([x[0][i], y[0][i], x[0][0], y[0][0]])
        for j in range(n):
            if j < n - 1:
                b = cp.array([x[0][j], y[0][j], x[0][j + 1], y[0][j + 1]])
            else:
                b = cp.array([x[0][j], y[0][j], x[0][0], y[0][0]])
            if Intersection_cupy(a, b) == True:
                index = index + 1
    if index == 3 * n:
        return False
    else:
        return True


def Intersection_cupy(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    if cp.maximum(ax1, ax2) < cp.minimum(bx1, bx2):
        return False
    if cp.maximum(ay1, ay2) < cp.minimum(by1, by2):
        return False
    if cp.maximum(bx1, bx2) < cp.minimum(ax1, ax2):
        return False
    if cp.maximum(by1, by2) < cp.minimum(ay1, ay2):
        return False
    if (
        cp.cross(cp.array([ax1 - bx1, ay1 - by1]), cp.array([bx2 - bx1, by2 - by1]))
        * cp.cross(cp.array([bx2 - bx1, by2 - by1]), cp.array([ax2 - bx1, ay2 - by1]))
        < 0
    ):
        return False
    if (
        cp.cross(cp.array([bx1 - ax1, by1 - ay1]), cp.array([ax2 - ax1, ay2 - ay1]))
        * cp.cross(cp.array([ax2 - ax1, ay2 - ay1]), cp.array([bx2 - ax1, by2 - ay1]))
        < 0
    ):
        return False
    else:
        return True


def coefficients(phi, M, N):
    """
    phi: complex-valued function
    M: number of discrete points
    N: cardinality of coefficients, cos and sin both have N coefficients,
    a total of 2N+1 coefficients
    Output: trigonometric coefficients of complex-valued function,
    the first 1/2 is the coefficient of the real part,
    the second 1/2 is the coefficient of the imaginary part
    """
    phi_real = np.real(phi)
    phi_imag = np.imag(phi)
    phi_real_fourier = (
        np.fft.fft(np.reshape(phi_real, [1, -1])) * np.sqrt(2 * np.pi) / M
    )
    phi_imag_fourier = (
        np.fft.fft(np.reshape(phi_imag, [1, -1])) * np.sqrt(2 * np.pi) / M
    )
    phi_real_fourier = np.reshape(phi_real_fourier, [1, -1])
    phi_imag_fourier = np.reshape(phi_imag_fourier, [1, -1])
    phi_real_f = resort_fourier(phi_real_fourier, N)
    phi_imag_f = resort_fourier(phi_imag_fourier, N)
    phi_f = np.concatenate([phi_real_f, phi_imag_f], axis=1)
    return phi_f


def coefficients_cupy(phi, M, N):
    """
    phi: complex-valued function
    M: number of discrete points
    N: cardinality of coefficients, cos and sin both have N coefficients,
    a total of 2N+1 coefficients
    Output: trigonometric coefficients of complex-valued function,
    the first 1/2 is the coefficient of the real part,
    the second 1/2 is the coefficient of the imaginary part
    """
    # Ensure phi is a CuPy array (if not, convert it)
    phi = cp.asarray(phi)

    # Separate real and imaginary parts
    phi_real = cp.real(phi)
    phi_imag = cp.imag(phi)

    # Compute Fourier transforms
    phi_real_fourier = cp.fft.fft(phi_real.reshape([1, -1])) * cp.sqrt(2 * cp.pi) / M
    phi_imag_fourier = cp.fft.fft(phi_imag.reshape([1, -1])) * cp.sqrt(2 * cp.pi) / M

    # Reshape the Fourier transforms
    phi_real_fourier = phi_real_fourier.reshape([1, -1])
    phi_imag_fourier = phi_imag_fourier.reshape([1, -1])

    # Resort the Fourier coefficients (you need to define this function in CuPy)
    phi_real_f = resort_fourier_cupy(phi_real_fourier, N)
    phi_imag_f = resort_fourier_cupy(phi_imag_fourier, N)

    # Concatenate real and imaginary parts of Fourier coefficients
    phi_f = cp.concatenate([phi_real_f, phi_imag_f], axis=1)

    return phi_f


def resort_fourier(p, N):
    """ ""
    p: Fourier coefficient(complex), sort by k in [0,1,2,3,...,M,-M-1,-N-2,...,-1],
    we want resort coefficient in sin,cos. Namely, [0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx]

    """

    if len(p.shape) == 1:
        p = np.reshape(p, [1, -1])
        m = 1
    else:
        m, M = p.shape
    f = np.zeros([m, 2 * N + 1])
    f[:, 0] = p[:, 0]
    f[:, 1 : N + 1] = p[:, 1 : N + 1] + p[:, -1 : -N - 1 : -1]
    f[:, N + 1 : 2 * N + 1] = (p[:, 1 : N + 1] - p[:, -1 : -N - 1 : -1]) / -(1j)
    # for i in range(1, N + 1):
    #     f[:, i] = p[:, i] + p[:, -i]
    #     f[:, i + N] = (p[:, i] - p[:, -i]) / -(1j)
    return f / np.sqrt(2 * np.pi)


def resort_fourier_cupy(p, N):
    """
    p: Fourier coefficient(complex), sort by k in [0,1,2,3,...,M,-M-1,-N-2,...,-1],
    we want resort coefficient in sin,cos. Namely, [0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx]
    """

    if len(p.shape) == 1:
        p = p.reshape(1, -1)
        m = 1
    else:
        m, M = p.shape
    f = cp.zeros([m, 2 * N + 1])
    f[:, 0] = p[:, 0]  # Keep the zero-frequency term (real)
    f[:, 1 : N + 1] = p[:, 1 : N + 1] + p[:, -1 : -N - 1 : -1]
    f[:, N + 1 : 2 * N + 1] = (p[:, 1 : N + 1] - p[:, -1 : -N - 1 : -1]) / -(1j)

    return f / cp.sqrt(2 * cp.pi)


def determine(x, y, pointx, pointy, min=0, max=10000, I=True):
    index = np.zeros(pointx.shape)
    polygon = Polygon(
        np.concatenate([np.reshape(x, [-1, 1]), np.reshape(y, [-1, 1])], axis=1)
    )
    x1 = []
    y1 = []
    for i in range(pointx.shape[0]):
        for j in range(pointx.shape[1]):
            point = Point([pointx[i, j], pointy[i, j]])

            if (
                I == True
                and polygon.contains(point) == True
                and polygon.boundary.distance(point) > min
                and polygon.boundary.distance(point) < max
            ):
                index[i, j] = 1
                # print(polygon.boundary.distance(point))
                x1.append(pointx[i, j])
                y1.append(pointy[i, j])
            if (
                I == False
                and polygon.contains(point) == False
                and polygon.boundary.distance(point) > min
                and polygon.boundary.distance(point) < max
            ):
                index[i, j] = 1
                x1.append(pointx[i, j])
                y1.append(pointy[i, j])
    return index, np.reshape(x1, [-1, 1]), np.reshape(y1, [-1, 1])


def block(idx_mat, u):
    u = np.reshape(u, [1, -1])
    c, r = idx_mat.shape
    U = np.nan * np.zeros([c, r])
    k = 0
    for i in range(c):
        for j in range(r):
            if idx_mat[i, j] == 1:
                U[i, j] = u[0, k]
                k = k + 1
    return U


def to_point(p, t):
    if len(p.shape) == 1:
        N = (len(p) - 1) // 2
        p = np.reshape(p, [1, -1])
    else:
        N = (p.shape[1] - 1) // 2
    sin_part = []
    cos_part = []
    for i in range(1, N + 1):
        sin_part.append(np.sin(i * t))
        cos_part.append(np.cos(i * t))
    sin_part = np.reshape(sin_part, [N, -1])
    cos_part = np.reshape(cos_part, [N, -1])
    p_cos = np.reshape(p[:, 1 : N + 1], [-1, N])
    p_sin = np.reshape(p[:, N + 1 :], [-1, N])
    phi = (
        np.matmul(p_sin, sin_part)
        + np.matmul(p_cos, cos_part)
        + np.reshape(p[:, 0], [-1, 1])
    )
    if len(phi.shape) == 1:
        phi = np.reshape(phi, [1, -1])
    return phi


def to_point_cupy(p, t, batch_size=100):
    # Determine the shape of input
    if len(p.shape) == 1:
        N = (len(p) - 1) // 2
        p = cp.reshape(p, [1, -1])
    else:
        N = (p.shape[1] - 1) // 2
    # Calculate sin and cos parts (these do not depend on batch size, so we can calculate them once)
    sin_part = cp.array([cp.sin(i * t) for i in range(1, N + 1)]).reshape(N, -1)
    cos_part = cp.array([cp.cos(i * t) for i in range(1, N + 1)]).reshape(N, -1)

    # Initialize output matrix
    phi_total = []

    # Process in batches
    for i in range(0, p.shape[0], batch_size):
        # Extract p_cos and p_sin for the current batch
        p_batch = p[i : i + batch_size]
        p_cos = p_batch[:, 1 : N + 1]
        p_sin = p_batch[:, N + 1 :]

        # Calculate phi for the current batch
        phi_batch = (
            cp.matmul(p_sin, sin_part)
            + cp.matmul(p_cos, cos_part)
            + p_batch[:, 0].reshape(-1, 1)
        )

        # Append the result to phi_total
        phi_total.append(phi_batch)

    # Concatenate all batch results into the final output
    phi = cp.concatenate(phi_total, axis=0)

    return phi


def d_to_point(p, t):
    if len(p.shape) == 1:
        N = (len(p) - 1) // 2
    else:
        N = (p.shape[1] - 1) // 2
    sin_part = []
    cos_part = []
    for i in range(1, N + 1):
        sin_part.append(i * np.sin(i * t))
        cos_part.append(i * np.cos(i * t))
    sin_part = np.reshape(sin_part, [N, -1])
    cos_part = np.reshape(cos_part, [N, -1])
    p_cos = np.reshape(p[:, 1 : N + 1], [-1, N])
    p_sin = np.reshape(p[:, N + 1 :], [-1, N])
    dphi = np.matmul(p_sin, cos_part) - np.matmul(p_cos, sin_part)
    if len(dphi.shape) == 1:
        dphi = np.reshape(dphi, [1, -1])
    return dphi


def initial(p, t):
    if len(p.shape) == 1:
        p = np.reshape(p, [1, -1])

    N = (p.shape[1] - 2) // 4
    sin_part = []
    cos_part = []
    dsin_part = []
    dcos_part = []
    ddsin_part = []
    ddcos_part = []

    for i in range(1, N + 1):
        sin_part.append(np.sin(i * t))
        cos_part.append(np.cos(i * t))
        dsin_part.append(i * np.cos(i * t))
        dcos_part.append(-i * np.sin(i * t))
        ddsin_part.append(-(i**2) * np.sin(i * t))
        ddcos_part.append(-(i**2) * np.cos(i * t))

    Px = p[:, : 2 * N + 1]
    Py = p[:, 2 * N + 1 :]

    Px = np.reshape(Px, [p.shape[0], -1])
    Py = np.reshape(Py, [p.shape[0], -1])
    sin_part = np.reshape(sin_part, [N, -1])
    cos_part = np.reshape(cos_part, [N, -1])
    dsin_part = np.reshape(dsin_part, [N, -1])
    dcos_part = np.reshape(dcos_part, [N, -1])
    ddsin_part = np.reshape(ddsin_part, [N, -1])
    ddcos_part = np.reshape(ddcos_part, [N, -1])

    Px_cos = Px[:, 1 : N + 1]
    Px_sin = Px[:, N + 1 :]
    Py_cos = Py[:, 1 : N + 1]
    Py_sin = Py[:, N + 1 :]
    x = (
        np.matmul(Px_sin, sin_part)
        + np.matmul(Px_cos, cos_part)
        + np.reshape(Px[:, 0], [-1, 1])
    )
    y = (
        np.matmul(Py_sin, sin_part)
        + np.matmul(Py_cos, cos_part)
        + np.reshape(Py[:, 0], [-1, 1])
    )
    dx = np.matmul(Px_sin, dsin_part) + np.matmul(Px_cos, dcos_part)
    dy = np.matmul(Py_sin, dsin_part) + np.matmul(Py_cos, dcos_part)
    ddx = np.matmul(Px_sin, ddsin_part) + np.matmul(Px_cos, ddcos_part)
    ddy = np.matmul(Py_sin, ddsin_part) + np.matmul(Py_cos, ddcos_part)
    return x, y, dx, dy, ddx, ddy


def initial_cupy(p, t):
    if len(p.shape) == 1:
        p = cp.reshape(p, [1, -1])

    N = (p.shape[1] - 2) // 4
    sin_part = []
    cos_part = []
    dsin_part = []
    dcos_part = []
    ddsin_part = []
    ddcos_part = []

    for i in range(1, N + 1):
        sin_part.append(cp.sin(i * t))
        cos_part.append(cp.cos(i * t))
        dsin_part.append(i * cp.cos(i * t))
        dcos_part.append(-i * cp.sin(i * t))
        ddsin_part.append(-(i**2) * cp.sin(i * t))
        ddcos_part.append(-(i**2) * cp.cos(i * t))

    Px = p[:, : 2 * N + 1]
    Py = p[:, 2 * N + 1 :]

    sin_part = cp.array(sin_part).reshape(N, -1)
    cos_part = cp.array(cos_part).reshape(N, -1)
    dsin_part = cp.array(dsin_part).reshape(N, -1)
    dcos_part = cp.array(dcos_part).reshape(N, -1)
    ddsin_part = cp.array(ddsin_part).reshape(N, -1)
    ddcos_part = cp.array(ddcos_part).reshape(N, -1)

    Px_cos = Px[:, 1 : N + 1]
    Px_sin = Px[:, N + 1 :]
    Py_cos = Py[:, 1 : N + 1]
    Py_sin = Py[:, N + 1 :]

    x = (
        cp.matmul(Px_sin, sin_part)
        + cp.matmul(Px_cos, cos_part)
        + Px[:, 0].reshape(-1, 1)
    )
    y = (
        cp.matmul(Py_sin, sin_part)
        + cp.matmul(Py_cos, cos_part)
        + Py[:, 0].reshape(-1, 1)
    )
    dx = cp.matmul(Px_sin, dsin_part) + cp.matmul(Px_cos, dcos_part)
    dy = cp.matmul(Py_sin, dsin_part) + cp.matmul(Py_cos, dcos_part)
    ddx = cp.matmul(Px_sin, ddsin_part) + cp.matmul(Px_cos, ddcos_part)
    ddy = cp.matmul(Py_sin, ddsin_part) + cp.matmul(Py_cos, ddcos_part)

    return x, y, dx, dy, ddx, ddy


def composite_gaussian_cupy():
    """
    Compute the integral of a function over the interval [a, b] using the composite
    Gaussian quadrature rule.
    The interval is divided into N subintervals,
    and on each subinterval, the 16-point Gaussian quadrature formula is applied.

    Returns:
    - gauss_nodes: Nodes for the 16-point Gaussian quadrature
    - gauss_weights: Weights for the 16-point Gaussian quadrature
    """

    # Nodes and weights for the 16-point Gaussian quadrature rule
    gauss_nodes = cp.array(
        [
            -0.9894009349916499,
            -0.9445750230732326,
            -0.8656312023878318,
            -0.7554044083550030,
            -0.6178762444026438,
            -0.4580167776572274,
            -0.2816035507792589,
            -0.0950125098376374,
            0.0950125098376374,
            0.2816035507792589,
            0.4580167776572274,
            0.6178762444026438,
            0.7554044083550030,
            0.8656312023878318,
            0.9445750230732326,
            0.9894009349916499,
        ]
    )
    gauss_weights = cp.array(
        [
            0.0271524594117540,
            0.0622535239386479,
            0.0951585116824928,
            0.1246289712555339,
            0.1495959888165767,
            0.1691565193950025,
            0.1826034150449236,
            0.1894506104550685,
            0.1894506104550685,
            0.1826034150449236,
            0.1691565193950025,
            0.1495959888165767,
            0.1246289712555339,
            0.0951585116824928,
            0.0622535239386479,
            0.0271524594117540,
        ]
    )

    return gauss_nodes, cp.reshape(gauss_weights, [1, -1])


def composite_gaussian():
    """
    Compute the integral of a function over the interval [a, b] using the composite
    Gaussian quadrature rule.
    The interval is divided into N subintervals, and on each subinterval,
    the 16-point Gaussian quadrature formula is applied.

    Returns:
    - gauss_nodes: Nodes for the 16-point Gaussian quadrature
    - gauss_weights: Weights for the 16-point Gaussian quadrature
    """
    # Nodes and weights for the 16-point Gaussian quadrature rule
    gauss_nodes = np.array(
        [
            -0.9894009349916499,
            -0.9445750230732326,
            -0.8656312023878318,
            -0.7554044083550030,
            -0.6178762444026438,
            -0.4580167776572274,
            -0.2816035507792589,
            -0.0950125098376374,
            0.0950125098376374,
            0.2816035507792589,
            0.4580167776572274,
            0.6178762444026438,
            0.7554044083550030,
            0.8656312023878318,
            0.9445750230732326,
            0.9894009349916499,
        ]
    )
    gauss_weights = np.array(
        [
            0.0271524594117540,
            0.0622535239386479,
            0.0951585116824928,
            0.1246289712555339,
            0.1495959888165767,
            0.1691565193950025,
            0.1826034150449236,
            0.1894506104550685,
            0.1894506104550685,
            0.1826034150449236,
            0.1691565193950025,
            0.1495959888165767,
            0.1246289712555339,
            0.0951585116824928,
            0.0622535239386479,
            0.0271524594117540,
        ]
    )

    return gauss_nodes, np.reshape(gauss_weights, [1, -1])


class IDP:
    """
    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(p, t)

        n1, n2 = dy, -dx
        direction = -1
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -dy, -n2
            direction = 1

        K = (
            1
            / (np.pi)
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            / ((x - x.T) ** 2 + (y - y.T) ** 2)
        )

        diag = 1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = np.diag_indices_from(K)
        K[c, r] = diag
        K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / M**2
        K = K.conj()
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.reshape(1, -1).shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)

        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data, special=False):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * np.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian()
        t = 0.5 * h * np.tile(gauss_nodes, Panel) + 0.5 * h * np.repeat(
            np.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = (
            1
            / (2 * np.pi)
            * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
            / ((x1 - x.T) ** 2 + (y1 - y.T) ** 2)
        )

        phi = to_point(phi_f, t)
        print((np.tile(gauss_weights, Panel).T * K).shape, phi.shape)
        total_integral = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel) * K.T, phi.T)

        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        # phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        if method == "solve":
            phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        elif method == "gmres":
            phi_fourier, _ = npgmres((I - self.K), f_fourier, tol=tol, restart=restart)
        else:
            raise ValueError("No such method: {}".format(method))
        # phi_fourier = np.matmul(np.linalg.inv(I - self.K), f_fourier)
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class IDP_cupy:
    """
    This code is the cupy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        t = t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(p, t)

        n1, n2 = dy, -dx
        direction = -1
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -dy, -n2
            direction = 1

        K = (
            1
            / (cp.pi)
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            / ((x - x.T) ** 2 + (y - y.T) ** 2)
        )

        diag = 1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = cp.diag_indices_from(K)
        K[c, r] = diag
        K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / M**2
        K = K.conj()
        self.x = cp.reshape(x, [1, -1])
        self.y = cp.reshape(y, [1, -1])
        self.dx = cp.reshape(dx, [1, -1])
        self.dy = cp.reshape(dy, [1, -1])
        self.ddx = cp.reshape(ddx, [1, -1])
        self.ddy = cp.reshape(ddy, [1, -1])
        self.M = M
        self.N = (cp.reshape(p, [1, -1]).shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """

        phi_fourier = cp.reshape(phi_fourier, [-1, 1])
        I = cp.eye(self.M)
        f_fourier = cp.matmul((I - self.K), phi_fourier)

        return cp.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """

        p = self.p
        N = self.N
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = cp.reshape(x1, [1, -1])
        y1 = cp.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        t = 0.5 * h * cp.tile(gauss_nodes, Panel) + 0.5 * h * cp.repeat(
            cp.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial_cupy(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = (
            1
            / (2 * cp.pi)
            * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
            / ((x1 - x.T) ** 2 + (y1 - y.T) ** 2)
        )

        phi = to_point_cupy(phi_f, t)
        # print((cp.tile(gauss_weights, Panel).T * K).shape, phi.shape)
        total_integral = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel) * K.T, phi.T)

        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = cp.reshape(f_fourier, [-1, 1])
        I = cp.eye(self.M)
        if method == "solve":
            phi_fourier = cp.linalg.solve((I - self.K), f_fourier)
        elif method == "gmres":
            phi_fourier, _ = gmres((I - self.K), f_fourier, tol=tol, restart=restart)
        else:
            raise ValueError("No such method: {}".format(method))

        phi_fourier = cp.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier_cupy(cp.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class EDP:
    """
    This code is the numpy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1

        K = -1 / (np.pi) * (n1.T * (x - x.T) + n2.T * (y - y.T)) / (
            (x - x.T) ** 2 + (y - y.T) ** 2
        ) - 2 * np.sqrt(dx.T**2 + dy.T**2)
        diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (
            dx**2 + dy**2
        ) - 2 * np.sqrt(dx**2 + dy**2)
        c, r = np.diag_indices_from(K)
        K[c, r] = diag
        K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / M**2
        K = K.conj()
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)
        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian()
        t = 0.5 * h * np.tile(gauss_nodes, Panel) + 0.5 * h * np.repeat(
            np.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = 1 / (2 * np.pi) * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T)) / (
            (x1 - x.T) ** 2 + (y1 - y.T) ** 2
        ) + np.sqrt(dx.T**2 + dy.T**2)

        phi = to_point(phi_f, t)
        total_integral = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel) * K.T, phi.T)

        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class EDP_cupy:
    """
    This code is the cupy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        p = cp.reshape(p, [1, -1])
        t = t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1

        K = -1 / (cp.pi) * (n1.T * (x - x.T) + n2.T * (y - y.T)) / (
            (x - x.T) ** 2 + (y - y.T) ** 2
        ) - 2 * cp.sqrt(dx.T**2 + dy.T**2)
        diag = -1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (
            dx**2 + dy**2
        ) - 2 * cp.sqrt(dx**2 + dy**2)
        c, r = cp.diag_indices_from(K)
        K[c, r] = diag
        K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / M**2
        K = K.conj()
        self.x = cp.reshape(x, [1, -1])
        self.y = cp.reshape(y, [1, -1])
        self.dx = cp.reshape(dx, [1, -1])
        self.dy = cp.reshape(dy, [1, -1])
        self.ddx = cp.reshape(ddx, [1, -1])
        self.ddy = cp.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = cp.reshape(phi_fourier, [-1, 1])
        I = cp.eye(self.M)
        f_fourier = cp.matmul((I - self.K), phi_fourier)
        return cp.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = cp.reshape(x1, [1, -1])
        y1 = cp.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        t = 0.5 * h * cp.tile(gauss_nodes, Panel) + 0.5 * h * cp.repeat(
            cp.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial_cupy(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = 1 / (2 * cp.pi) * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T)) / (
            (x1 - x.T) ** 2 + (y1 - y.T) ** 2
        ) + cp.sqrt(dx.T**2 + dy.T**2)

        phi = to_point(phi_f, t)
        print((cp.tile(gauss_weights, Panel).T * K).shape, phi.shape)
        total_integral = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel) * K.T, phi.T)
        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = cp.reshape(f_fourier, [-1, 1])
        I = cp.eye(self.M)
        if method == "solve":
            phi_fourier = cp.linalg.solve((I - self.K), f_fourier)
        elif method == "gmres":
            phi_fourier, _ = gmres((I - self.K), f_fourier, tol, restart)
        else:
            raise ValueError("No such method: {}".format(method))

        phi_fourier = cp.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier_cupy(cp.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class INP:
    """
    This code is the numpy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):

        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1
            print("diriction:", direction)
            # t = np.linspace(2 * np.pi * (1 - 1 / M), 0, M)
            # x, y, dx, dy, ddx, ddy = initial(p, t)
            # n1, n2 = dy, -dx

        K = (
            1
            / (np.pi)
            * (((x - x.T) * n1 + (y - y.T) * n2) * np.sqrt(dx.T**2 + dy.T**2))
            / (((x - x.T) ** 2 + (y - y.T) ** 2) * np.sqrt(dx**2 + dy**2))
        )
        diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = np.diag_indices_from(K)
        K[c, r] = diag
        # print(K)
        # K_transformed = np.fft.fft(np.conj(np.fft.fft(K, axis=0)), axis=0).T
        # K = np.fft.fft(K_transformed, axis=0) * (2 * np.pi / M**2)
        K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / M**2
        K = K.conj()
        # K = (np.fft.fft2(K).conj().T) * 2 * np.pi / M**2
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)

        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * np.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian()
        t = 0.5 * h * np.tile(gauss_nodes, Panel) + 0.5 * h * np.repeat(
            np.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = -(
            1 / (2 * np.pi) * np.log(np.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
        ) * np.sqrt(dx.T**2 + dy.T**2)

        phi = to_point(phi_f, t)
        total_integral = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel) * K.T, phi.T)
        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        A_ex = np.ones((self.M + 1, self.M + 1), dtype=np.complex64)
        A_ex[: self.M, : self.M] = I - self.K
        A_ex[self.M, self.M] = 0

        if method == "solve":
            phi_fourier = np.linalg.solve(
                A_ex,
                np.concatenate([f_fourier, np.zeros([1, 1]).reshape(-1, 1)], axis=1),
            )
        elif method == "gmres":
            phi_fourier, _ = npgmres(
                A_ex,
                np.concatenate([f_fourier, np.zeros([1, 1]).reshape(-1, 1)], axis=1),
                tol=tol,
                restart=restart,
            )
        else:
            raise ValueError("No such method: {}".format(method))

        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class INP_cupy:
    """
    This code is the cupy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):

        p = cp.reshape(p, [1, -1])
        t = t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1

        K = (
            1
            / (cp.pi)
            * (((x - x.T) * n1 + (y - y.T) * n2) * cp.sqrt(dx.T**2 + dy.T**2))
            / (((x - x.T) ** 2 + (y - y.T) ** 2) * cp.sqrt(dx**2 + dy**2))
        )
        diag = -1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = cp.diag_indices_from(K)
        K[c, r] = diag
        K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / M**2
        K = K.conj()
        self.x = cp.reshape(x, [1, -1])
        self.y = cp.reshape(y, [1, -1])
        self.dx = cp.reshape(dx, [1, -1])
        self.dy = cp.reshape(dy, [1, -1])
        self.ddx = cp.reshape(ddx, [1, -1])
        self.ddy = cp.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = cp.reshape(phi_fourier, [-1, 1])
        I = cp.eye(self.M)
        f_fourier = cp.matmul((I - self.K), phi_fourier)

        return cp.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = cp.reshape(x1, [1, -1])
        y1 = cp.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        t = 0.5 * h * cp.tile(gauss_nodes, Panel) + 0.5 * h * cp.repeat(
            cp.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial_cupy(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = -(
            1 / (2 * cp.pi) * cp.log(cp.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
        ) * cp.sqrt(dx.T**2 + dy.T**2)
        phi = to_point(phi_f, t)
        print((cp.tile(gauss_weights, Panel).T * K).shape, phi.shape)
        total_integral = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel) * K.T, phi.T)
        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = cp.reshape(f_fourier, [-1, 1])
        I = cp.eye(self.M)
        # A_ex = cp.ones((self.M + 1, self.M + 1), dtype=cp.complex64)
        # A_ex[: self.M, : self.M] = I - self.K
        # A_ex[self.M, self.M] = 0
        if method == "solve":
            phi_fourier = cp.linalg.solve(
                # A_ex,
                I-self.K,
                # cp.concatenate([f_fourier, cp.zeros([1, 1]).reshape(-1, 1)], axis=0),
                f_fourier,
            )
        elif method == "gmres":
            phi_fourier, _ = gmres(
                A_ex,
                cp.concatenate([f_fourier, cp.zeros([1, 1]).reshape(-1, 1)], axis=0),
                tol=tol,
                restart=restart,
            )
        else:
            raise ValueError("No such method: {}".format(method))

        phi_fourier = cp.reshape(phi_fourier[:-1], [1, -1])
        phi_f = resort_fourier_cupy(cp.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class ENP:
    """
    This code is the numpy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1
        K = -(
            1
            / (np.pi)
            * (((x - x.T) * n1 + (y - y.T) * n2) * np.sqrt(dx.T**2 + dy.T**2))
            / (((x - x.T) ** 2 + (y - y.T) ** 2) * np.sqrt(dx**2 + dy**2))
        )
        diag = 1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = np.diag_indices_from(K)
        K[c, r] = diag
        K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / M**2
        K = K.conj()
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)
        print(np.linalg.cond(I - self.K))
        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * np.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian()
        t = 0.5 * h * np.tile(gauss_nodes, Panel) + 0.5 * h * np.repeat(
            np.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = -(
            1 / (2 * np.pi) * np.log(np.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
        ) * np.sqrt(dx.T**2 + dy.T**2)

        phi = to_point(phi_f, t)
        total_integral = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel) * K.T, phi.T)
        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier, _ = npgmres(
            (I - self.K),
            f_fourier,
            tol=1e-15,
            restart=10,
        )
        print(np.linalg.cond(I - self.K))
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class ENP_cupy:
    """
    This code is the cupy version

    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    phi: density funtion
    K: kernel of BIEs
    in_data: Points within the domain

    phi_to_pde: Input -->density, integral points; Output -->solution of LVBPS at integral points
    phi_to_f: Input -->density; Output -->boundrary integral of density
    f_to_pde: Input -->rhs, integral points(Optional); Output-->density, solution of LVBPS at integral points(If integral points are given)
    """

    def __init__(self, M, p):
        p = cp.reshape(p, [1, -1])
        t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1
        K = -(
            1
            / (cp.pi)
            * (((x - x.T) * n1 + (y - y.T) * n2) * cp.sqrt(dx.T**2 + dy.T**2))
            / (((x - x.T) ** 2 + (y - y.T) ** 2) * cp.sqrt(dx**2 + dy**2))
        )
        diag = 1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = cp.diag_indices_from(K)
        K[c, r] = diag
        K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / M**2
        K = K.conj()
        self.x = cp.reshape(x, [1, -1])
        self.y = cp.reshape(y, [1, -1])
        self.dx = cp.reshape(dx, [1, -1])
        self.dy = cp.reshape(dy, [1, -1])
        self.ddx = cp.reshape(ddx, [1, -1])
        self.ddy = cp.reshape(ddy, [1, -1])
        self.M = M
        self.N = (p.shape[1] - 2) // 4
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        Input: density fourier coefficients
        Output: rhs fourier coefficients
        """
        phi_fourier = cp.reshape(phi_fourier, [-1, 1])
        I = cp.eye(self.M)
        f_fourier = cp.matmul((I - self.K), phi_fourier)
        return cp.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        """
        Calculate the potential integral

        Input: density Trigonometric coefficients and integral points
        Output: potential integral at integral points
        """
        p = self.p
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = cp.reshape(x1, [1, -1])
        y1 = cp.reshape(x2, [1, -1])

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        t = 0.5 * h * cp.tile(gauss_nodes, Panel) + 0.5 * h * cp.repeat(
            cp.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial_cupy(p, t)
        n1, n2 = dy, -dx
        if direction == 1:
            n1, n2 = -n1, -n2

        K = -(
            1 / (2 * cp.pi) * cp.log(cp.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
        ) * cp.sqrt(dx.T**2 + dy.T**2)
        phi = to_point(phi_f, t)
        print((cp.tile(gauss_weights, Panel).T * K).shape, phi.shape)
        total_integral = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel) * K.T, phi.T)
        return total_integral

    def f_to_pde(self, f_fourier, *in_data, method="solve", tol=1e-15, restart=10):
        """
        Solving BIEs and Calculate the potential integral

        Input: rhs fourier coefficients and integral points(optional)
        Output: density Trigonometric coefficients and potential integral at integral points(if in_data)
        """

        f_fourier = cp.reshape(f_fourier, [-1, 1])
        I = cp.eye(self.M)
        if method == "solve":
            phi_fourier = cp.linalg.solve((I - self.K), f_fourier)
        elif method == "gmres":
            phi_fourier, _ = gmres((I - self.K), f_fourier, tol, restart)
        else:
            raise ValueError("No such method: {}".format(method))

        phi_fourier = cp.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier_cupy(cp.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class Elastostatic:
    """
    E: Young's modulus
    nu: Poisson's ratio
    dx: Boundary derivative in the x-direction
    dy: Boundary derivative in the y-direction
    du1: Displacement derivative in the x-direction
    du2: Displacement derivative in the y-direction
    M: Number of sampling points
    N: Number of terms in the triangular series (sin/cos terms)
    """

    def __init__(self, parag, E, nu, N):
        G = E / (2 * (1 + nu))
        self.G = G
        self.E = E
        self.nu = nu
        self.N = N
        self.parag = parag

    def displacement_to_traction(self, x, y, dx, dy, gradu1, gradu2):
        """
        gradu1: M rows, 2 columns [du11, du12]
        gradu2: M rows, 2 columns [du21, du22]
        """

        G = self.G
        nu = self.nu
        E = self.E
        mu = E / (2 * (1 + nu))
        lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        # du = np.concatenate([gradu1, gradu2], axis=0).T

        n1 = dy / np.sqrt(dx**2 + dy**2)
        n2 = -dx / np.sqrt(dx**2 + dy**2)
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2

        M = dx.shape[1]

        f1 = np.zeros([1, M])
        f2 = np.zeros([1, M])
        for i in range(M):
            epsilon_xx = gradu1[i, 0]
            epsilon_yy = gradu2[i, 1]
            epsilon_xy = 0.5 * (gradu1[i, 1] + gradu2[i, 0])
            sigma_xx = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
            sigma_yy = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
            sigma_xy = 2 * mu * epsilon_xy

            # Calculate the stresses f1 and f2 on the boundary
            f1[0, i] = sigma_xx * n1[0, i] + sigma_xy * n2[0, i]
            f2[0, i] = sigma_xy * n1[0, i] + sigma_yy * n2[0, i]

        f1 = np.fft.fft(f1) * np.sqrt(2 * np.pi) / M
        f2 = np.fft.fft(f2) * np.sqrt(2 * np.pi) / M
        f1 = resort_fourier(f1, self.N)
        f2 = resort_fourier(f2, self.N)
        f = np.concatenate([f1, f2], axis=1)
        return f

    def kernel(self, x, y, dx, dy, x1, y1, *direction):
        """
        x: x coordinates on the boundary
        y: y coordinates on the boundary
        dx: partial derivative of x with respect to x at the boundary point
        dy: partial derivative of y with respect to y at the boundary point
        x1: x coordinates inside the region
        y1: y coordinates inside the region
        """

        # M = self.M
        nu = self.nu
        G = self.G
        # [x1,x2,...,xn]-[x1,x2,...,xn].T=[[x1-x1,x2-x1,...,xn-x1][x1-x2,...]...]
        r = np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr1 = (x - x1.T) / np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr2 = (y - y1.T) / np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)

        if len(direction) != 0:
            direction = direction[0]
            # print(direction)
            if direction == -1:
                n1 = dy / np.sqrt(dx**2 + dy**2)
                n2 = -dx / np.sqrt(dx**2 + dy**2)
            elif direction == 1:
                n1 = -dy / np.sqrt(dx**2 + dy**2)
                n2 = dx / np.sqrt(dx**2 + dy**2)
            else:
                raise ValueError("direction must be either 1 or -1")
        dr_dn = dr1 * n1 + dr2 * n2

        U11 = -1 / (8 * np.pi * G * (1 - nu)) * ((3 - 4 * nu) * np.log(r) - dr1**2)
        U12 = 1 / (8 * np.pi * G * (1 - nu)) * (dr1 * dr2)
        U21 = 1 / (8 * np.pi * G * (1 - nu)) * (dr2 * dr1)
        U22 = -1 / (8 * np.pi * G * (1 - nu)) * ((3 - 4 * nu) * np.log(r) - dr2**2)
        T11 = (-1 / (4 * np.pi * (1 - nu) * r)) * (
            (dr_dn * ((1 - 2 * nu) + 2 * dr1**2))
        )
        T12 = (-1 / (4 * np.pi * (1 - nu) * r)) * (
            (dr_dn * (2 * dr1 * dr2)) - (1 - 2 * nu) * (dr1 * n2 - dr2 * n1)
        )
        T21 = (-1 / (4 * np.pi * (1 - nu) * r)) * (
            (dr_dn * (2 * dr1 * dr2)) - (1 - 2 * nu) * (dr2 * n1 - dr1 * n2)
        )
        T22 = (-1 / (4 * np.pi * (1 - nu) * r)) * (
            (dr_dn * ((1 - 2 * nu) + 2 * dr2**2))
        )
        return U11, U12, U21, U22, T11, T12, T21, T22

    def k1(self, x, y, dx, dy, x1, y1):
        nu = self.nu
        G = self.G
        nb = nu / (1 - nu)
        A1 = -(1 + nb) / (4 * np.pi)
        A2 = (1 - nb) / (1 + nb)

        # [x1,x2,...,xn]-[x1,x2,...,xn].T=[[x1-x1,x2-x1,...,xn-x1][x1-x2,...]...]
        r = np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr1 = (x - x1.T) / r
        dr2 = (y - y1.T) / r
        # n1, n2 = dy, -dx

        # direction = -1
        n1 = dy / np.sqrt(dx**2 + dy**2)
        n2 = -dx / np.sqrt(dx**2 + dy**2)
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        dr_dn = dr1 * n1 + dr2 * n2
        dr_dt = -dr1 * n2 + dr2 * n1
        U22 = (
            -1
            / (8 * np.pi * G)
            * ((3 - nb) * np.log(r) - (1 + nb) * dr1**2 + (7 - nb) / 2)
        )
        U12 = 1 / (8 * np.pi * G) * (dr1 * dr2) * (1 + nb)
        U21 = 1 / (8 * np.pi * G) * (dr2 * dr1) * (1 + nb)
        U11 = (
            -1
            / (8 * np.pi * G)
            * ((3 - nb) * np.log(r) - (1 + nb) * dr2**2 + (7 - nb) / 2)
        )
        T22 = A1 / r * (A2 + 2 * dr1**2) * dr_dn
        T21 = A1 / r * (2 * dr1 * dr2 * dr_dn - A2 * dr_dt)
        T12 = A1 / r * (2 * dr1 * dr2 * dr_dn + A2 * dr_dt)
        T11 = A1 / r * (A2 + 2 * dr2**2) * dr_dn
        return U11, U12, U21, U22, T11, T12, T21, T22

    def to_pde(self, parau, paraf, x1, y1):
        N = self.N
        parag = self.parag
        gauss_nodes, gauss_weights = composite_gaussian()
        H = 120
        h = (2 * np.pi) / H
        M = 256

        xx, yy, dxx, dyy, _, _ = initial(
            parag, np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
        )
        if np.sum(xx * dyy - yy * dxx) < 0:
            direction = 1
        else:
            direction = -1

        # 16 point complex Gaussian integral
        Panel = 40
        # node = np.linspace(0, 2 * np.pi, Panel, endpoint=False)
        h = (2 * np.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian()
        t = 0.5 * h * np.tile(gauss_nodes, Panel) + 0.5 * h * np.repeat(
            np.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial(parag, t)
        u1, u2, _, _, _, _ = initial(parau, t)
        f1, f2, _, _, _, _ = initial(paraf, t)
        U11, U12, U21, U22, T11, T12, T21, T22 = self.kernel(
            x, y, dx, dy, x1, y1, direction
        )
        func1 = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * np.sqrt(dx**2 + dy**2)
        func2 = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * np.sqrt(dx**2 + dy**2)
        total_integral1 = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel), func1.T)
        total_integral2 = 0.5 * h * np.matmul(np.tile(gauss_weights, Panel), func2.T)

        # for i in range(H):
        #     # 计算第i个子区间的左右端点
        #     ai = 0 + i * h
        #     bi = ai + h

        #     # 将高斯节点映射到当前子区间
        #     ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
        #     x, y, dx, dy, _, _ = initial(parag, ti)
        #     u1, u2, _, _, _, _ = initial(parau, ti)
        #     f1, f2, _, _, _, _ = initial(paraf, ti)
        #     # u = np.concatenate([u1, u2], axis=1)
        #     # f = np.concatenate([f1, f2], axis=1)

        #     # f = self.displacement_to_traction(dx, dy, du1, du2)
        #     func1i = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * np.sqrt(
        #         dx**2 + dy**2
        #     )
        #     func2i = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * np.sqrt(
        #         dx**2 + dy**2
        #     )

        #     integral1_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func1i, axis=1)
        #     integral2_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func2i, axis=1)

        #     # 累加到总积分
        #     total_integral1 += integral1_i
        #     total_integral2 += integral2_i
        return total_integral1, total_integral2

    def Euler_to_pde(self, parau, paraf, x1, y1):
        N = self.N
        parag = self.parag
        # H = 80
        M = 512
        t = np.linspace(0, 2 * np.pi, M)
        x, y, dx, dy, _, _ = initial(parag, t)
        if np.sum(x * dy - y * dx) < 0:
            direction = 1
        else:
            direction = -1
        h = (2 * np.pi) / M
        u1, u2, _, _, _, _ = initial(parau, t)
        f1, f2, _, _, _, _ = initial(paraf, t)
        U11, U12, U21, U22, T11, T12, T21, T22 = self.kernel(
            x, y, dx, dy, x1, y1, direction
        )
        # f = self.displacement_to_traction(dx, dy, du1, du2)
        func1i = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * np.sqrt(dx**2 + dy**2)

        func2i = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * np.sqrt(dx**2 + dy**2)

        return np.sum(h * func1i, axis=1), np.sum(h * func2i, axis=1)


class Elastostatic_cupy:
    """
    E: Young's modulus
    nu: Poisson's ratio
    dx: Boundary derivative in the x-direction
    dy: Boundary derivative in the y-direction
    du1: Displacement derivative in the x-direction
    du2: Displacement derivative in the y-direction
    M: Number of sampling points
    N: Number of terms in the triangular series (sin/cos terms)
    """

    def __init__(self, parag, E, nu, N):
        G = E / (2 * (1 + nu))  # 剪切模量
        self.G = G
        self.E = E
        self.nu = nu
        self.N = N
        self.parag = parag

    def displacement_to_traction(self, x, y, dx, dy, gradu1, gradu2):
        """
        gradu1: M rows, 2 columns [du11, du12]
        gradu2: M rows, 2 columns [du21, du22]
        """

        G = self.G
        nu = self.nu
        E = self.E
        mu = E / (2 * (1 + nu))
        lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        # du = cp.concatenate([gradu1, gradu2], axis=0).T

        n1 = dy / cp.sqrt(dx**2 + dy**2)
        n2 = -dx / cp.sqrt(dx**2 + dy**2)
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2

        M = dx.shape[1]

        f1 = cp.zeros([1, M])
        f2 = cp.zeros([1, M])
        for i in range(M):
            epsilon_xx = gradu1[i, 0]
            epsilon_yy = gradu2[i, 1]
            epsilon_xy = 0.5 * (gradu1[i, 1] + gradu2[i, 0])
            sigma_xx = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_xx
            sigma_yy = lambda_ * (epsilon_xx + epsilon_yy) + 2 * mu * epsilon_yy
            sigma_xy = 2 * mu * epsilon_xy

            # Calculate the stresses f1 and f2 on the boundary
            f1[0, i] = sigma_xx * n1[0, i] + sigma_xy * n2[0, i]
            f2[0, i] = sigma_xy * n1[0, i] + sigma_yy * n2[0, i]

        f1 = cp.fft.fft(f1) * cp.sqrt(2 * cp.pi) / M
        f2 = cp.fft.fft(f2) * cp.sqrt(2 * cp.pi) / M
        f1 = resort_fourier_cupy(f1, self.N)
        f2 = resort_fourier_cupy(f2, self.N)
        f = cp.concatenate([f1, f2], axis=1)
        return f

    def kernel(self, x, y, dx, dy, x1, y1, *direction):
        """
        x: x coordinates on the boundary
        y: y coordinates on the boundary
        dx: partial derivative of x with respect to x at the boundary point
        dy: partial derivative of y with respect to y at the boundary point
        x1: x coordinates inside the region
        y1: y coordinates inside the region
        """

        # M = self.M
        nu = self.nu
        G = self.G
        # [x1,x2,...,xn]-[x1,x2,...,xn].T=[[x1-x1,x2-x1,...,xn-x1][x1-x2,...]...]
        r = cp.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr1 = (x - x1.T) / cp.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr2 = (y - y1.T) / cp.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)

        if len(direction) != 0:
            direction = direction[0]
            # print(direction)
            if direction == -1:
                n1 = dy / cp.sqrt(dx**2 + dy**2)
                n2 = -dx / cp.sqrt(dx**2 + dy**2)
            elif direction == 1:
                n1 = -dy / cp.sqrt(dx**2 + dy**2)
                n2 = dx / cp.sqrt(dx**2 + dy**2)
            else:
                raise ValueError("direction must be either 1 or -1")
        dr_dn = dr1 * n1 + dr2 * n2

        U11 = -1 / (8 * cp.pi * G * (1 - nu)) * ((3 - 4 * nu) * cp.log(r) - dr1**2)
        U12 = 1 / (8 * cp.pi * G * (1 - nu)) * (dr1 * dr2)
        U21 = 1 / (8 * cp.pi * G * (1 - nu)) * (dr2 * dr1)
        U22 = -1 / (8 * cp.pi * G * (1 - nu)) * ((3 - 4 * nu) * cp.log(r) - dr2**2)
        T11 = (-1 / (4 * cp.pi * (1 - nu) * r)) * (
            (dr_dn * ((1 - 2 * nu) + 2 * dr1**2))
        )
        T12 = (-1 / (4 * cp.pi * (1 - nu) * r)) * (
            (dr_dn * (2 * dr1 * dr2)) - (1 - 2 * nu) * (dr1 * n2 - dr2 * n1)
        )
        T21 = (-1 / (4 * cp.pi * (1 - nu) * r)) * (
            (dr_dn * (2 * dr1 * dr2)) - (1 - 2 * nu) * (dr2 * n1 - dr1 * n2)
        )
        T22 = (-1 / (4 * cp.pi * (1 - nu) * r)) * (
            (dr_dn * ((1 - 2 * nu) + 2 * dr2**2))
        )
        return U11, U12, U21, U22, T11, T12, T21, T22

    def k1(self, x, y, dx, dy, x1, y1):
        nu = self.nu
        G = self.G
        nb = nu / (1 - nu)
        A1 = -(1 + nb) / (4 * cp.pi)
        A2 = (1 - nb) / (1 + nb)

        # [x1,x2,...,xn]-[x1,x2,...,xn].T=[[x1-x1,x2-x1,...,xn-x1][x1-x2,...]...]
        r = cp.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr1 = (x - x1.T) / r
        dr2 = (y - y1.T) / r
        # n1, n2 = dy, -dx

        # direction = -1
        n1 = dy / cp.sqrt(dx**2 + dy**2)
        n2 = -dx / cp.sqrt(dx**2 + dy**2)
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        dr_dn = dr1 * n1 + dr2 * n2
        dr_dt = -dr1 * n2 + dr2 * n1
        U22 = (
            -1
            / (8 * cp.pi * G)
            * ((3 - nb) * cp.log(r) - (1 + nb) * dr1**2 + (7 - nb) / 2)
        )
        U12 = 1 / (8 * cp.pi * G) * (dr1 * dr2) * (1 + nb)
        U21 = 1 / (8 * cp.pi * G) * (dr2 * dr1) * (1 + nb)
        U11 = (
            -1
            / (8 * cp.pi * G)
            * ((3 - nb) * cp.log(r) - (1 + nb) * dr2**2 + (7 - nb) / 2)
        )
        T22 = A1 / r * (A2 + 2 * dr1**2) * dr_dn
        T21 = A1 / r * (2 * dr1 * dr2 * dr_dn - A2 * dr_dt)
        T12 = A1 / r * (2 * dr1 * dr2 * dr_dn + A2 * dr_dt)
        T11 = A1 / r * (A2 + 2 * dr2**2) * dr_dn
        return U11, U12, U21, U22, T11, T12, T21, T22

    def to_pde(self, parau, paraf, x1, y1):
        N = self.N
        parag = self.parag
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        H = 120
        h = (2 * cp.pi) / H
        M = 256

        xx, yy, dxx, dyy, _, _ = initial_cupy(
            parag, cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        )
        if cp.sum(xx * dyy - yy * dxx) < 0:
            direction = 1
        else:
            direction = -1

        # 16 point complex Gaussian integral
        Panel = 40
        # node = cp.linspace(0, 2 * cp.pi, Panel, endpoint=False)
        h = (2 * cp.pi) / Panel
        gauss_nodes, gauss_weights = composite_gaussian_cupy()
        t = 0.5 * h * cp.tile(gauss_nodes, Panel) + 0.5 * h * cp.repeat(
            cp.arange(1, 2 * Panel + 1, 2), 16
        )

        x, y, dx, dy, _, _ = initial_cupy(parag, t)
        u1, u2, _, _, _, _ = initial_cupy(parau, t)
        f1, f2, _, _, _, _ = initial_cupy(paraf, t)
        U11, U12, U21, U22, T11, T12, T21, T22 = self.kernel(
            x, y, dx, dy, x1, y1, direction
        )
        func1 = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * cp.sqrt(dx**2 + dy**2)
        func2 = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * cp.sqrt(dx**2 + dy**2)
        total_integral1 = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel), func1)
        total_integral2 = 0.5 * h * cp.matmul(cp.tile(gauss_weights, Panel), func2)

        # for i in range(H):
        #     # 计算第i个子区间的左右端点
        #     ai = 0 + i * h
        #     bi = ai + h

        #     # 将高斯节点映射到当前子区间
        #     ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
        #     x, y, dx, dy, _, _ = initial(parag, ti)
        #     u1, u2, _, _, _, _ = initial(parau, ti)
        #     f1, f2, _, _, _, _ = initial(paraf, ti)
        #     # u = np.concatenate([u1, u2], axis=1)
        #     # f = np.concatenate([f1, f2], axis=1)

        #     # f = self.displacement_to_traction(dx, dy, du1, du2)
        #     func1i = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * np.sqrt(
        #         dx**2 + dy**2
        #     )
        #     func2i = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * np.sqrt(
        #         dx**2 + dy**2
        #     )

        #     integral1_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func1i, axis=1)
        #     integral2_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func2i, axis=1)

        #     # 累加到总积分
        #     total_integral1 += integral1_i
        #     total_integral2 += integral2_i
        return total_integral1, total_integral2

    def Euler_to_pde(self, parau, paraf, x1, y1):
        N = self.N
        parag = self.parag
        # H = 80
        M = 512
        t = cp.linspace(0, 2 * cp.pi, M)
        x, y, dx, dy, _, _ = initial_cupy(parag, t)
        if cp.sum(x * dy - y * dx) < 0:
            direction = 1
        else:
            direction = -1
        h = (2 * cp.pi) / M
        u1, u2, _, _, _, _ = initial_cupy(parau, t)
        f1, f2, _, _, _, _ = initial_cupy(paraf, t)
        U11, U12, U21, U22, T11, T12, T21, T22 = self.kernel(
            x, y, dx, dy, x1, y1, direction
        )
        # f = self.displacement_to_traction(dx, dy, du1, du2)
        func1i = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * cp.sqrt(dx**2 + dy**2)

        func2i = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * cp.sqrt(dx**2 + dy**2)

        return cp.sum(h * func1i, axis=1), cp.sum(h * func2i, axis=1)


class Helmholtz:
    """
    para:几何的三角系数
    M:离散点数
    k:波数
    eta:耦合系数
    """

    def __init__(self, para, M, k, eta):
        self.para = para
        self.M = M
        self.k = k
        self.eta = eta

    def BIE(self, f_f, method="solve", tol=1e-4, restart=10):
        """
        para: 几何的三角系数
        k: 波数
        eta: 耦合系数
        f_f: 积分方程右端项的三角系数
        M: 离散点数
        输出:phi的三角系数
        """
        para = self.para
        k = self.k
        eta = self.eta
        M = self.M
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(para, t)
        t = np.reshape(t, [1, -1])
        n1, n2 = dy, -dx
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        distance = np.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)
        L = -(
            1j
            * k
            / 2
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            * hankel1(1, k * distance)
            / distance
        )
        c, r = np.diag_indices_from(L)
        diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        L[c, r] = diag
        L1 = (
            (k / (2 * np.pi))
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            * jn(1, k * distance)
            / distance
        )
        c, r = np.diag_indices_from(L1)
        L1[c, r] = 0
        L2 = L - L1 * np.log(4 * (np.sin((t - t.T) / 2)) ** 2)
        c, r = np.diag_indices_from(L2)
        L2[c, r] = diag
        M = 1j / 2 * hankel1(0, k * distance) * np.sqrt(dx.T**2 + dy.T**2)
        C = 0.57721566490153286060651209
        M1 = (-1 / (2 * np.pi)) * jn(0, k * distance) * np.sqrt(dx.T**2 + dy.T**2)
        M2 = M - M1 * np.log(4 * (np.sin((t - t.T) / 2)) ** 2)
        diag = (
            (1j / 2) - C / np.pi - 1 / np.pi * np.log(k / 2 * np.sqrt(dx**2 + dy**2))
        ) * np.sqrt(dx**2 + dy**2)
        c, r = np.diag_indices_from(M2)
        M2[c, r] = diag
        K1 = L1 + 1j * eta * M1
        K2 = L2 + 1j * eta * M2
        R = 0
        n = t.shape[1] // 2

        for i in range(1, n):
            R = R + 1 / i * np.cos(i * (t - t.T))
        R = -2 * np.pi / n * R - np.pi / n**2 * np.cos(n * (t - t.T))

        I = np.eye(2 * n)

        K = R * K1 + np.pi / n * K2
        f_real, f_imag, _, _, _, _ = initial(f_f, t)
        f = f_real + 1j * f_imag

        if method == "solve":
            phi = np.linalg.solve((I - K.T), np.reshape(f, [-1, 1]))
        elif method == "gmres":
            phi, _ = npgmres(I - K.T, np.reshape(f, [-1, 1]), tol=tol, restart=restart)
        else:
            raise ValueError("No such method: {}".format(method))
        # phi = np.linalg.solve(I - K.T, np.reshape(f, [-1, 1]))
        N = (f_f.shape[1] - 2) // 4
        phi_f = coefficients(phi, 2 * n, N)
        return phi_f

    def u_infity(self, x1, y1, phi_f):
        """
        x1:远场位置第一个分量
        y1:远场位置第二个分量
        phi_f: phi的三角系数,前半部分是实部,后半部分是虚部
        """
        k = self.k
        M = self.M
        para = self.para
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, _, _ = initial(para, t)

        n1, n2 = dy / np.sqrt(dx**2 + dy**2), -dx / np.sqrt(dx**2 + dy**2)
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        phi_real, phi_imag = initial(phi_f, t)
        phi = phi_real + 1j * phi_imag
        K1 = (
            (k * (n1 * x1 + n2 * y1) + k)
            * np.exp(-1j * k * (x1 * x + y1 * y))
            * np.sqrt(dx**2 + dy**2)
        )
        # n = x1.shape[1] // 2
        n = M // 2
        u_infity = (
            np.exp(-1j * np.pi / 4)
            / np.sqrt(8 * np.pi * k)
            * np.matmul(K1, phi.T)
            * np.pi
            / n
        )
        return u_infity

    def u_scatter(self, x1, y1, phi_f):
        """
        x1:远场位置第一个分量
        y1:远场位置第二个分量
        phi_f: phi的三角系数,前半部分是实部,后半部分是虚部
        """
        eta = self.eta
        k = self.k
        M = self.M
        para = self.para
        t = np.linspace(0, 2 * np.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial(para, t)
        t = np.reshape(t, [1, -1])
        n1, n2 = dy, -dx
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        distance = np.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2)
        L = -(
            1j
            * k
            / 2
            * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
            * hankel1(1, k * distance)
            / distance
        )
        # c, r = np.diag_indices_from(L)
        # diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        # L[c, r] = diag
        # L = (
        #     (k / (2 * np.pi))
        #     * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
        #     * jn(1, k * distance)
        #     / distance
        # )
        # c, r = np.diag_indices_from(L1)
        # L1[c, r] = 0
        # L2 = L - L1 * np.log(4 * (np.sin((t - t.T) / 2)) ** 2)
        # c, r = np.diag_indices_from(L2)
        # L2[c, r] = diag
        M = 1j / 2 * hankel1(0, k * distance) * np.sqrt(dx.T**2 + dy.T**2)
        C = 0.57721566490153286060651209
        # M1 = (-1 / (2 * np.pi)) * jn(0, k * distance) * np.sqrt(dx.T**2 + dy.T**2)
        # M2 = M - M1 * np.log(4 * (np.sin((t - t.T) / 2)) ** 2)
        # diag = (
        #     (1j / 2) - C / np.pi - 1 / np.pi * np.log(k / 2 * np.sqrt(dx**2 + dy**2))
        # ) * np.sqrt(dx**2 + dy**2)
        # c, r = np.diag_indices_from(M2)
        # M2[c, r] = diag
        # K1 = L1 + 1j * eta * M1
        # K2 = L2 + 1j * eta * M2
        K = L + 1j * eta * M
        R = 0
        n = t.shape[1] // 2

        # for i in range(1, n):
        #     R = R + 1 / i * np.cos(i * (t - t.T))
        # R = -2 * np.pi / n * R - np.pi / n**2 * np.cos(n * (t - t.T))

        I = np.eye(2 * n)

        # K = R * K1 + np.pi / n * K2
        h = 2 * np.pi / self.M
        phi_real, phi_imag, _, _, _, _ = initial(phi_f, t)
        phi = phi_real + 1j * phi_imag
        phi = np.reshape(phi, [-1, 1])
        u_scatter = np.matmul(K.T * h, phi)

        return u_scatter


class Helmholtz_cupy:
    """
    para: 几何的三角系数
    M: 离散点数
    k: 波数
    eta: 耦合系数
    """

    def __init__(self, para, M, k, eta):
        self.para = para
        self.M = M
        self.k = k
        self.eta = eta

    def BIE(self, f_f, method="solve", tol=1e-4, restart=10):
        """
        para: 几何的三角系数
        k: 波数
        eta: 耦合系数
        f_f: 积分方程右端项的三角系数
        M: 离散点数
        输出: phi的三角系数
        """
        para = self.para
        k = self.k
        eta = self.eta
        M = self.M
        t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(para, t)
        t = cp.reshape(t, [1, -1])
        n1, n2 = dy, -dx
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        distance = cp.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)
        L = -(
            1j
            * k
            / 2
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            * hankel1(1, k * distance)
            / distance
        )
        c, r = cp.diag_indices_from(L)
        diag = -1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        L[c, r] = diag
        L1 = (
            (k / (2 * cp.pi))
            * (n1.T * (x - x.T) + n2.T * (y - y.T))
            * jn(1, k * distance)
            / distance
        )
        c, r = cp.diag_indices_from(L1)
        L1[c, r] = 0
        L2 = L - L1 * cp.log(4 * (cp.sin((t - t.T) / 2)) ** 2)
        c, r = cp.diag_indices_from(L2)
        L2[c, r] = diag
        M = 1j / 2 * hankel1(0, k * distance) * cp.sqrt(dx.T**2 + dy.T**2)
        C = 0.57721566490153286060651209
        M1 = (-1 / (2 * cp.pi)) * jn(0, k * distance) * cp.sqrt(dx.T**2 + dy.T**2)
        M2 = M - M1 * cp.log(4 * (cp.sin((t - t.T) / 2)) ** 2)
        diag = (
            (1j / 2) - C / cp.pi - 1 / cp.pi * cp.log(k / 2 * cp.sqrt(dx**2 + dy**2))
        ) * cp.sqrt(dx**2 + dy**2)
        c, r = cp.diag_indices_from(M2)
        M2[c, r] = diag
        K1 = L1 + 1j * eta * M1
        K2 = L2 + 1j * eta * M2
        R = 0
        n = t.shape[1] // 2

        for i in range(1, n):
            R = R + 1 / i * cp.cos(i * (t - t.T))
        R = -2 * cp.pi / n * R - cp.pi / n**2 * cp.cos(n * (t - t.T))

        I = cp.eye(2 * n)

        K = R * K1 + cp.pi / n * K2
        f_real, f_imag, _, _, _, _ = initial_cupy(f_f, t)
        f = f_real + 1j * f_imag
        # phi = cp.linalg.solve(I - K.T, cp.reshape(f, [-1, 1]))
        if method == "solve":
            phi = cp.linalg.solve((I - K.T), cp.reshape(f, [-1, 1]))
        elif method == "gmres":
            phi, _ = gmres((I - K.T), cp.reshape(f, [-1, 1]), tol, restart)
        else:
            raise ValueError("No such method: {}".format(method))
        N = (para.shape[1] - 2) // 4
        phi_f = coefficients_cupy(phi, 2 * n, N)
        return phi_f

    def u_infity(self, x1, y1, phi_f):
        """
        x1: 远场位置第一个分量
        y1: 远场位置第二个分量
        phi_f: phi的三角系数, 前半部分是实部, 后半部分是虚部
        """
        k = self.k
        M = self.M
        para = self.para
        t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, _, _ = initial_cupy(para, t)

        n1, n2 = dy / cp.sqrt(dx**2 + dy**2), -dx / cp.sqrt(dx**2 + dy**2)
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        phi_real, phi_imag = initial_cupy(phi_f, t)
        phi = phi_real + 1j * phi_imag
        K1 = (
            (k * (n1 * x1 + n2 * y1) + k)
            * cp.exp(-1j * k * (x1 * x + y1 * y))
            * cp.sqrt(dx**2 + dy**2)
        )
        n = M // 2
        u_infity = (
            cp.exp(-1j * cp.pi / 4)
            / cp.sqrt(8 * cp.pi * k)
            * cp.matmul(K1, phi.T)
            * cp.pi
            / n
        )
        return u_infity

    def u_scatter(self, x1, y1, phi_f):
        """
        x1: 远场位置第一个分量
        y1: 远场位置第二个分量
        phi_f: phi的三角系数, 前半部分是实部, 后半部分是虚部
        """
        eta = self.eta
        k = self.k
        M = self.M
        para = self.para
        t = cp.linspace(0, 2 * cp.pi, M, endpoint=False)
        x, y, dx, dy, ddx, ddy = initial_cupy(para, t)
        t = cp.reshape(t, [1, -1])
        n1, n2 = dy, -dx
        if cp.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
        distance = cp.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2)
        L = -(
            1j
            * k
            / 2
            * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
            * hankel1(1, k * distance)
            / distance
        )
        M = 1j / 2 * hankel1(0, k * distance) * cp.sqrt(dx.T**2 + dy.T**2)
        C = 0.57721566490153286060651209
        K = L + 1j * eta * M
        R = 0
        n = t.shape[1] // 2

        I = cp.eye(2 * n)
        h = 2 * cp.pi / self.M
        phi_real, phi_imag, _, _, _, _ = initial_cupy(phi_f, t)
        phi = phi_real + 1j * phi_imag
        phi = cp.reshape(phi, [-1, 1])
        u_scatter = cp.matmul(K.T * h, phi)

        return u_scatter
