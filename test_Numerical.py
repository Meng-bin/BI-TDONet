import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.special import jn, hankel1


def coefficients(phi, M, N):
    """
    phi: 复值函数
    M:离散的点数
    N:系数的基数,cos,sin都有N个,总共有2N+1个系数
    输出：复值函数的三角系数,前1/2是实部的系数,后1/2是虚部的系数
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


def euler_quadrature(func, a, b, N):
    # 计算子区间的长度
    h = (b - a) / N

    total_integral = 0.0
    for i in range(N):
        # 计算第i个子区间的左右端点
        ai = a + i * h
        bi = ai + h

        # 将高斯节点映射到当前子区间
        # xi = 0.5 * (bi - ai) * 1 + 0.5 * (ai + bi)

        # 计算子区间上的积分
        # print(func(xi).shape)
        integral_i = 0.5 * (bi - ai) * np.sum((func(ai) + func(bi)), axis=1)

        # 累加到总积分
        total_integral += integral_i

    return total_integral


def composite_gaussian():
    """
    使用复合高斯求积法则计算函数在区间[a, b]上的积分。
    区间被等分为N个子区间,在每个子区间上采用16点高斯求积公式。

    参数:
    - func: 被积函数
    - a: 区间左端点
    - b: 区间右端点
    - N: 子区间的数量

    返回值:
    - 区间[a, b]上函数的近似积分值
    """
    # 16点高斯求积法则的节点和权重
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


# Define the periodic kernel function
def periodic_kernel(x, x_prime, l, p=2 * np.pi):
    """Calculate the periodic kernel between two points."""
    return np.exp(-2 * (np.sin(np.pi * np.abs(x - x_prime) / p) ** 2) / l**2)


class IDP:
    """
    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos
    f: Fourier coefficient of right-hand side of BIES
    in_data: Points within the domain
    """

    def __init__(self, M, N, p):

        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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

        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = N
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)

        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        p = self.p
        N = self.N
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 40 point complex Gaussian integral

        total_integral = 0
        h = (2 * np.pi) / 40
        gauss_nodes, gauss_weights = composite_gaussian()
        for i in range(40):
            ai = 0 + i * h
            bi = ai + h
            ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
            x, y, dx, dy, _, _ = initial(p, ti)
            n1, n2 = dy, -dx
            if direction == 1:
                n1, n2 = -n1, -n2

            K = (
                1
                / (2 * np.pi)
                * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T))
                / ((x1 - x.T) ** 2 + (y1 - y.T) ** 2)
            )

            phi = to_point(phi_f, ti)
            integral_i = 0.5 * (bi - ai) * np.sum(gauss_weights.T * K * phi.T, axis=0)
            total_integral += integral_i

        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        # phi_fourier = np.matmul(np.linalg.inv(I - self.K), f_fourier)
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if not in_data:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class EDP:
    """
    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos

    f: Fourier coefficient of bountary condation
    in_data: Points within the domain
    """

    def __init__(self, M, N, p):
        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = N
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)
        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        p = self.p
        N = self.N
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 40 point complex Gaussian integral

        total_integral = 0
        h = (2 * np.pi) / 40
        gauss_nodes, gauss_weights = composite_gaussian()
        for i in range(40):
            ai = 0 + i * h
            bi = ai + h
            ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
            x, y, dx, dy, _, _ = initial(p, ti)
            n1, n2 = dy, -dx
            if direction == 1:
                n1, n2 = -n1, -n2

            K = 1 / (2 * np.pi) * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T)) / (
                (x1 - x.T) ** 2 + (y1 - y.T) ** 2
            ) + np.sqrt(dx.T**2 + dy.T**2)

            phi = to_point(phi_f, ti)
            integral_i = 0.5 * (bi - ai) * np.sum(gauss_weights.T * K * phi.T, axis=0)
            total_integral += integral_i

        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if in_data == False:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


def EDP_PDE(para_f, phi_f, in_data):
    p = para_f
    N = (phi_f.shape[1] - 1) // 2
    N = 40
    h = (2 * np.pi) / N
    gauss_nodes, gauss_weights = composite_gaussian()
    gauss_weights = np.tile(gauss_weights, (1, N))
    t = np.reshape(0.5 * h * gauss_nodes + 0.5 * h, [1, -1])
    for i in range(1, N):
        ai = 0 + i * h
        bi = ai + h
        ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
        ti = np.reshape(ti, [1, -1])
        t = np.concatenate([t, ti], axis=1)
        # t.append(ti)
    # t_array = np.array(t)
    # t = t_array.reshape(1, -1)
    x, y, dx, dy, _, _ = initial(p, t)
    n1, n2 = dy, -dx

    if np.sum(x * dy - y * dx) < 0:
        n1, n2 = -n1, -n2

    # n1, n2 = dy, -dx
    # if direction == 1:
    #     n1, n2 = -n1, -n2
    x1 = in_data[:, 0]
    x2 = in_data[:, 1]
    x1 = np.reshape(x1, [1, -1])
    y1 = np.reshape(x2, [1, -1])
    K = 1 / (2 * np.pi) * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T)) / (
        (x1 - x.T) ** 2 + (y1 - y.T) ** 2
    ) + np.sqrt(dx.T**2 + dy.T**2)
    # 16 point complex Gaussian integral
    phi = to_point(phi_f, t)
    # np.tile(, (1, N))
    total_integral = 0.5 * h * np.matmul(gauss_weights * K.T, phi.T)
    # total_integral = 0
    # h = (2 * np.pi) / 40
    # gauss_nodes, gauss_weights = composite_gaussian()
    # for i in range(40):
    #     ai = 0 + i * h
    #     bi = ai + h
    #     ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
    #     x, y, dx, dy, _, _ = initial(p, ti)
    #     n1, n2 = dy, -dx
    #     if direction == 1:
    #         n1, n2 = -n1, -n2

    #     K = 1 / (2 * np.pi) * (n1.T * (x1 - x.T) + n2.T * (y1 - y.T)) / (
    #         (x1 - x.T) ** 2 + (y1 - y.T) ** 2
    #     ) + np.sqrt(dx.T**2 + dy.T**2)

    #     phi = to_point(phi_f, ti)
    #     integral_i = 0.5 * (bi - ai) * np.sum(gauss_weights.T * K * phi.T, axis=0)
    #     total_integral += integral_i

    return total_integral


class INP:
    """
    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos

    f: Fourier coefficient of bountary condation
    in_data: Points within the domain
    """

    def __init__(self, M, N, p):

        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
        x, y, dx, dy, ddx, ddy = initial(p, t)
        n1, n2 = dy, -dx
        direction = -1
        if np.sum(x * dy - y * dx) < 0:
            n1, n2 = -n1, -n2
            direction = 1
        K = (
            1
            / (np.pi)
            * (((x - x.T) * n1 + (y - y.T) * n2) * np.sqrt(dx.T**2 + dy.T**2))
            / (((x - x.T) ** 2 + (y - y.T) ** 2) * np.sqrt(dx**2 + dy**2))
        )
        diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
        c, r = np.diag_indices_from(K)
        K[c, r] = diag
        K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / M**2
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = N
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)
        return np.reshape(f_fourier, [1, -1])

    def phi_to_f(self, phi_fourier):
        """
        phi_fourier: Fourier coefficient of phi
        f_fourier: Fourier coefficient of f

        """
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)

        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        p = self.p
        N = self.N
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 40 point complex Gaussian integral

        total_integral = 0
        h = (2 * np.pi) / 40
        gauss_nodes, gauss_weights = composite_gaussian()
        for i in range(40):
            ai = 0 + i * h
            bi = ai + h
            ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
            x, y, dx, dy, _, _ = initial(p, ti)
            n1, n2 = dy, -dx
            if direction == 1:
                n1, n2 = -n1, -n2

            K = -(
                1 / (2 * np.pi) * np.log(np.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
            ) * np.sqrt(dx.T**2 + dy.T**2)

            phi = to_point(phi_f, ti)
            integral_i = 0.5 * (bi - ai) * np.sum(gauss_weights.T * K * phi.T, axis=0)
            total_integral += integral_i

        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        print(np.linalg.cond(I - self.K))
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)
        if in_data == False:
            return phi_f
        else:
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class ENP:
    """
    M: number of discrete point of [0,2pi]
    N: terms of boundaries =====> (a0,cosx,cos2x,...,cosNx,sinx,sin2x,...,sinNx)
    para: coefficient of boundaries,sin,cos

    f: Fourier coefficient of bountary condation
    in_data: Points within the domain
    """

    def __init__(self, M, N, p):
        p = np.reshape(p, [1, -1])
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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
        self.x = np.reshape(x, [1, -1])
        self.y = np.reshape(y, [1, -1])
        self.dx = np.reshape(dx, [1, -1])
        self.dy = np.reshape(dy, [1, -1])
        self.ddx = np.reshape(ddx, [1, -1])
        self.ddy = np.reshape(ddy, [1, -1])
        self.M = M
        self.N = N
        self.p = p
        self.t = t
        self.K = K
        self.direction = direction

    def phi_to_f(self, phi_fourier):
        phi_fourier = np.reshape(phi_fourier, [-1, 1])
        I = np.eye(self.M)
        f_fourier = np.matmul((I - self.K), phi_fourier)
        print(np.linalg.cond(I - self.K))
        return np.reshape(f_fourier, [1, -1])

    def phi_to_pde(self, phi_f, in_data):
        p = self.p
        N = self.N
        direction = self.direction

        x1 = in_data[:, 0]
        x2 = in_data[:, 1]
        x1 = np.reshape(x1, [1, -1])
        y1 = np.reshape(x2, [1, -1])

        # 40 point complex Gaussian integral

        total_integral = 0
        h = (2 * np.pi) / 40
        gauss_nodes, gauss_weights = composite_gaussian()
        for i in range(40):
            ai = 0 + i * h
            bi = ai + h
            ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
            x, y, dx, dy, _, _ = initial(p, ti)
            n1, n2 = dy, -dx
            if direction == 1:
                n1, n2 = -n1, -n2

            K = -(
                1 / (2 * np.pi) * np.log(np.sqrt((x1 - x.T) ** 2 + (y1 - y.T) ** 2))
            ) * np.sqrt(dx.T**2 + dy.T**2)

            phi = to_point(phi_f, ti)
            integral_i = 0.5 * (bi - ai) * np.sum(gauss_weights.T * K * phi.T, axis=0)
            total_integral += integral_i

        return total_integral

    def f_to_pde(self, f_fourier, *in_data):
        f_fourier = np.reshape(f_fourier, [-1, 1])
        I = np.eye(self.M)
        phi_fourier = np.linalg.solve((I - self.K), f_fourier)
        # print(np.linalg.cond(I - self.K))
        phi_fourier = np.reshape(phi_fourier, [1, -1])
        phi_f = resort_fourier(np.reshape(phi_fourier, [1, -1]), self.N)

        # 检查是否有额外的输入数据
        if len(in_data) == 0:  # 没有额外的输入数据
            return phi_f
        else:  # 有额外的输入数据
            in_data = in_data[0]
            u = self.phi_to_pde(phi_f, in_data)
            return phi_f, u


class Elastostatic:
    """
    E:杨氏模量 Young's modulus
    nu:泊松比 Poisson ratio
    dx:边界对x的方向导数
    dy:边界对y的方向导数
    du1:位移对x的方向导数
    du2:位移对y的方向导数
    M: 采样点数
    N: 三角级数sin/cos项数
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
        gradu1: M 行两列[du11,du12]
        gradu2: M 行两列[du21,du22]
        """
        G = self.G
        nu = self.nu
        E = self.E
        mu = E / (2 * (1 + nu))
        lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        # du = np.concatenate([gradu1, gradu2], axis=0).T
        # 计算应变张量的分量

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
            # 计算边界上的应力f1和f2
            f1[0, i] = sigma_xx * n1[0, i] + sigma_xy * n2[0, i]
            f2[0, i] = sigma_xy * n1[0, i] + sigma_yy * n2[0, i]
        # for i in range(M):
        #     gradu = np.concatenate(
        #         [np.reshape(gradu1[i, :], [1, -1]), np.reshape(gradu2[i, :], [1, -1])],
        #         axis=0,
        #     )
        #     epsion = 1 / 2 * (gradu + gradu.T)
        #     sigma = 2 * G * epsion + 2 * G * nu / (1 - 2 * nu) * np.trace(
        #         epsion
        #     ) * np.eye(2)
        #     f1[0, i] = sigma[0, 0] * n1[0, i] + sigma[0, 1] * n2[0, i]
        #     f2[0, i] = sigma[1, 0] * n1[0, i] + sigma[1, 1] * n2[0, i]

        # for i in range(M):
        #     f1[0, i] = (2 * G * nu / (1 - 2 * nu)) * (gradu1[i, 0] + gradu2[i, 1]) * n1[
        #         0, i
        #     ] + (G * (gradu1[i, 1] + gradu2[i, 0])) * n2[0, i]
        #     f2[0, i] = (2 * G * nu / (1 - 2 * nu)) * (gradu1[i, 0] + gradu2[i, 1]) * n2[
        #         0, i
        #     ] + (G * (gradu1[i, 1] + gradu2[i, 0])) * n1[0, i]
        f1 = np.fft.fft(f1) * np.sqrt(2 * np.pi) / M
        f2 = np.fft.fft(f2) * np.sqrt(2 * np.pi) / M
        f1 = resort_fourier(f1, self.N)
        f2 = resort_fourier(f2, self.N)
        f = np.concatenate([f1, f2], axis=1)
        return f

    def kernel(self, x, y, dx, dy, x1, y1, *direction):
        """
        x:边界上x坐标
        y:边界上y坐标
        dx:边界上x点处x的偏导数
        dy:边界上x点处y的偏导数
        x1:区域内x坐标
        y1:区域内y坐标
        """
        # M = self.M
        nu = self.nu
        G = self.G
        # [x1,x2,...,xn]-[x1,x2,...,xn].T=[[x1-x1,x2-x1,...,xn-x1][x1-x2,...]...]
        r = np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr1 = (x - x1.T) / np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        dr2 = (y - y1.T) / np.sqrt((x - x1.T) ** 2 + (y - y1.T) ** 2)
        # n1, n2 = dy, -dx

        # n1 = dy / np.sqrt(dx**2 + dy**2)
        # n2 = -dx / np.sqrt(dx**2 + dy**2)
        # if np.sum(x * dy - y * dx) < 0:
        #     n1, n2 = -n1, -n2

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
                raise ValueError("direction 必须是 1 或者 -1")

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
        total_integral1 = 0
        total_integral2 = 0
        for i in range(H):
            # 计算第i个子区间的左右端点
            ai = 0 + i * h
            bi = ai + h

            # 将高斯节点映射到当前子区间
            ti = 0.5 * (bi - ai) * gauss_nodes + 0.5 * (ai + bi)
            x, y, dx, dy, _, _ = initial(parag, ti)
            u1, u2, _, _, _, _ = initial(parau, ti)
            f1, f2, _, _, _, _ = initial(paraf, ti)
            # u = np.concatenate([u1, u2], axis=1)
            # f = np.concatenate([f1, f2], axis=1)
            U11, U12, U21, U22, T11, T12, T21, T22 = self.kernel(
                x, y, dx, dy, x1, y1, direction
            )
            # f = self.displacement_to_traction(dx, dy, du1, du2)
            func1i = (U11 * f1 + U12 * f2 - T11 * u1 - T12 * u2) * np.sqrt(
                dx**2 + dy**2
            )
            func2i = (U21 * f1 + U22 * f2 - T21 * u1 - T22 * u2) * np.sqrt(
                dx**2 + dy**2
            )

            integral1_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func1i, axis=1)
            integral2_i = 0.5 * (bi - ai) * np.sum(gauss_weights * func2i, axis=1)

            # 累加到总积分
            total_integral1 += integral1_i
            total_integral2 += integral2_i
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

    def BIE(self, f_f):
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
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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
        phi = np.linalg.solve(I - K.T, np.reshape(f, [-1, 1]))
        N = (para.shape[1] - 2) // 4
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
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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
        t = np.linspace(0, 2 * np.pi * (1 - 1 / M), M)
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
