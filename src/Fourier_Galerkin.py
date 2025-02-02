import cupy as cp
import numpy as np
from scipy.sparse.linalg import gmres as npgmres
from cupyx.scipy.sparse.linalg import gmres as cpgmres


def IDP_BIE_cupy(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
    n1, n2 = dy, -dx
    if cp.sum(x * dy - y * dx) < 0:
        n1, n2 = -dy, -n2

    K = (
        1
        / (cp.pi)
        * (n1.T * (x - x.T) + n2.T * (y - y.T))
        / ((x - x.T) ** 2 + (y - y.T) ** 2)
    )

    diag = 1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
    c, r = cp.diag_indices_from(K)
    K[c, r] = diag
    K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / (K.shape[0]) ** 2
    K = K.conj()
    # print(diag, K, I - K)

    # Solve the system using GMRES
    I = cp.eye(K.shape[0])
    if method == "solve":
        phi = cp.linalg.solve(I - K, cp.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi, _ = cpgmres(
            I - K,
            cp.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    # phi, _ = gmres((I - K), (cp.reshape(f, [-1, 1])), tol=1e-20)
    # Convert result back to CuPy array
    # return cp.asnumpy(phi)
    return phi


def EDP_BIE_cupy(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
    n1, n2 = dy, -dx
    direction = -1
    if cp.sum(x * dy - y * dx) < 0:
        n1, n2 = -n1, -n2
        direction = 1

    K = -1 / (cp.pi) * (n1.T * (x - x.T) + n2.T * (y - y.T)) / (
        (x - x.T) ** 2 + (y - y.T) ** 2
    ) - 2 * cp.sqrt(dx.T**2 + dy.T**2)
    diag = -1 / (2 * cp.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2) - 2 * cp.sqrt(
        dx**2 + dy**2
    )
    c, r = cp.diag_indices_from(K)
    K[c, r] = diag
    K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / K.shape[0] ** 2
    K = K.conj()
    # Solve the system using GMRES
    I = cp.eye(x.shape[1])
    if method == "solve":
        phi = cp.linalg.solve(I - K, cp.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi = cpgmres(
            I - K,
            cp.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    # Convert result back to CuPy array
    # return cp.asnumpy(phi)
    return phi


def INP_BIE_cupy(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
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
    K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / K.shape[0] ** 2
    K = K.conj()
    # Solve the system using GMRES
    I = cp.eye(x.shape[1])
    A = I - K
    ones_col = cp.ones((x.shape[1], 1))
    A_ext = cp.hstack((A, ones_col))
    ones_row = cp.ones((1, x.shape[1] + 1))
    A_ext = cp.vstack((A_ext, ones_row))
    A_ext[-1, -1] = 0
    b = cp.concatenate((f.reshape(-1, 1), cp.array([0]).reshape(-1, 1)), axis=0)
    # print(A.shape, b.shape)
    phi = cp.linalg.solve(A_ext, b)
    if method == "solve":
        phi = cp.linalg.solve(A_ext, b)
    elif method == "gmres":
        phi = cpgmres(
            A_ext,
            b,
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")
    phi = phi[:-1]
    # return cp.asnumpy(phi)
    return phi


def ENP_BIE_cupy(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
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
    K = cp.fft.fft(cp.conjugate(cp.fft.fft(K)).T) * 2 * cp.pi / K.shape[0] ** 2
    K = K.conj()
    # Solve the system using GMRES
    I = cp.eye(x.shape[1])
    if method == "solve":
        phi = cp.linalg.solve(I - K, cp.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi = cpgmres(
            I - K,
            cp.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    # Convert result back to CuPy array
    # return cp.asnumpy(phi)
    return phi


def IDP_BIE(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
    n1, n2 = dy, -dx
    if np.sum(x * dy - y * dx) < 0:
        n1, n2 = -n1, -n2

    # Compute matrix K
    K = (
        1
        / (np.pi)
        * (n1.T * (x - x.T) + n2.T * (y - y.T))
        / ((x - x.T) ** 2 + (y - y.T) ** 2)
    )

    diag = 1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2)
    c, r = np.diag_indices_from(K)
    K[c, r] = diag
    K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / (K.shape[0] ** 2)
    K = K.conj()
    # Solve the system using GMRES
    I = np.eye(K.shape[0])
    if method == "solve":
        phi = np.linalg.solve(I - K, np.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi, _ = npgmres(
            I - K,
            np.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    return phi


def EDP_BIE(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
    n1, n2 = dy, -dx
    if np.sum(x * dy - y * dx) < 0:
        n1, n2 = -n1, -n2
        direction = 1

    K = -1 / (np.pi) * (n1.T * (x - x.T) + n2.T * (y - y.T)) / (
        (x - x.T) ** 2 + (y - y.T) ** 2
    ) - 2 * np.sqrt(dx.T**2 + dy.T**2)
    diag = -1 / (2 * np.pi) * (n1 * ddx + n2 * ddy) / (dx**2 + dy**2) - 2 * np.sqrt(
        dx**2 + dy**2
    )
    c, r = np.diag_indices_from(K)
    K[c, r] = diag
    K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / K.shape[0] ** 2
    K = K.conj()

    # Solve the system using GMRES
    I = np.eye(x.shape[1])
    if method == "solve":
        phi = np.linalg.solve(I - K, np.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi, _ = npgmres(
            I - K,
            np.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    return phi


def INP_BIE(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
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
    K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / K.shape[0] ** 2
    K = K.conj()

    # Solve the system using GMRES
    I = np.eye(x.shape[1])
    A = I - K
    ones_col = np.ones((x.shape[1], 1))
    A_ext = np.hstack((A, ones_col))
    ones_row = np.ones((1, x.shape[1] + 1))
    A_ext = np.vstack((A_ext, ones_row))
    A_ext[-1, -1] = 0
    b = np.concatenate((f.reshape(-1, 1), np.array([0]).reshape(-1, 1)), axis=0)
    if method == "solve":
        phi = np.linalg.solve(A_ext, b)
    elif method == "gmres":
        phi, _ = npgmres(
            A_ext,
            b,
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    phi = phi[:-1]
    return phi


def ENP_BIE(x, y, dx, dy, ddx, ddy, f, method="solve", tol=1e-15, restart=10):
    # Normal vector computation
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
    K = np.fft.fft(np.conjugate(np.fft.fft(K)).T) * 2 * np.pi / K.shape[0] ** 2
    K = K.conj()

    # Solve the system using GMRES
    I = np.eye(x.shape[1])
    if method == "solve":
        phi = np.linalg.solve(I - K, np.reshape(f, [-1, 1]))
    elif method == "gmres":
        phi, _ = npgmres(
            I - K,
            np.reshape(f, [-1, 1]),
            tol=tol,
            restart=restart,
        )
    else:
        raise ValueError("Invalid method. Please choose 'solve' or 'gmres'.")

    return phi
