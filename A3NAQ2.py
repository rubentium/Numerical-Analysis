import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

# f_1(θ, φ) = a cos(θ) + b cos(θ + φ) − x_p
# f_2(θ, φ) = a sin(θ) + b sin(θ + φ) − y_p

# df_1/dθ = d/dθ(a cos(θ) + b cos(θ + ϕ) - x_p) = -a sin(θ) - b sin(θ + ϕ)
# df_1/dϕ = d/dϕ(a cos(θ) + b cos(θ + x) - x_p) = -b sin(θ + ϕ)

# df_2/dθ = d/dθ(a sin(θ) + b sin(θ + ϕ) - y_p) = a cos(θ) + b cos(θ + ϕ)
# df_2/dϕ = d/dx(a sin(θ) + b sin(θ + ϕ) - y_p) = b cos(θ + ϕ)

def robotarm( a, b, p, r, maxit=20, tol=10**(-8)):
    guess_v = r
    x_p, y_p = p[0], p[1]

    f_1 = lambda theta, phi, a, b, x_p: a*cos(theta) + b*cos(theta + phi) - x_p
    f_2 = lambda theta, phi, a, b, y_p: a*sin(theta) + b*sin(theta + phi) - y_p

    def f(v:list, a:float, b:float, x_p:float, y_p:float, f_1=f_1, f_2=f_2) -> list:
        # v is a matrix
        theta, phi = v[0], v[1]
        return np.array([f_1(theta, phi, a, b, x_p), f_2(theta, phi, a, b, y_p)])


    # Partial Derivative functions

    d_f1_theta = lambda theta, phi, a, b: -a*sin(theta) - b*sin(theta + phi)
    d_f1_phi = lambda theta, phi, b: -b*sin(theta + phi)

    d_f2_theta = lambda theta, phi, a, b: a*cos(theta) + b*cos(theta + phi)
    d_f2_phi = lambda theta, phi, b: b*cos(theta + phi)

    def Jac(v:list, a:float, b: float, d_f1_t = d_f1_theta,
            d_f1_p=d_f1_phi, d_f2_t=d_f2_theta, d_f2_p=d_f2_phi):

        theta, phi = v[0], v[1]
        row_1 = [ d_f1_t(theta, phi, a, b), d_f1_p(theta, phi, b) ]
        row_2 = [ d_f2_t(theta, phi, a, b), d_f2_p(theta, phi, b) ]

        return np.array([row_1, row_2])


    def delta_solver(original_v: list, a:float, b:float, x_p:float, y_p:float, J=Jac, f=f):

        f_vect = f(original_v, a, b, x_p, y_p)
        Jac_mat = J(original_v, a, b)

        delta_v = np.linalg.solve(Jac_mat, -1*f_vect)
        new_v = delta_v + original_v

        return new_v

    v = guess_v
    count = 0
    while count<=maxit and np.linalg.norm(f(v, a, b, x_p, y_p)) > tol:
        v = delta_solver(v, a, b, x_p, y_p)
        count += 1

    return v

# r = np.array([pi/4, -pi/2])
# p = np.array([1.2, 1.6])
# print(robotarm(1.5, 1, p, r))

def Output(a, b, p, r):
    maxit = 20
    data = []
    f_1 = lambda theta, phi, a, b, x_p: a*cos(theta) + b*cos(theta + phi) - x_p
    f_2 = lambda theta, phi, a, b, y_p: a*sin(theta) + b*sin(theta + phi) - y_p

    def f(v:list, a:float, b:float, x_p:float, y_p:float, f_1=f_1, f_2=f_2) -> list:
        # v is a matrix
        theta, phi = v[0], v[1]
        return np.array([f_1(theta, phi, a, b, x_p), f_2(theta, phi, a, b, y_p)])

    for i in range(maxit+1):
        v = robotarm(a, b, p, r, i)
        theta, phi = v[0], v[1]
        data.append([theta, phi, np.linalg.norm(f(v, a, b, p[0], p[1]))/np.linalg.norm(f(r, a, b, p[0], p[1])),
        a*cos(theta) + b*cos(theta + phi), a*sin(theta) + b*sin(theta + phi)])

    or_x = [0, a*cos(data[0][0]), data[0][3]]
    or_y = [0, a*sin(data[0][0]), data[0][4]]

    x = [0, a*cos(data[-1][0]), data[-1][3]]
    y = [0, a*sin(data[-1][0]), data[-1][4]]

    plt.plot(or_x, or_y, linestyle='dashed')
    plt.plot(x, y, linestyle='solid')
    plt.show()

    df = pd.DataFrame(data, columns = ['θ (theta)', 'φ (phi)', 'residual ||f(x^(k))||/||f(x^(0))||', 'x-coord', 'y-coord' ])
    return df

r = np.array([pi/4, -pi/2])
p = np.array([1.2, 1.6])
print(Output(1.5, 1, p, r))



