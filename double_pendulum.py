import pandas as pd
import numpy as np
import scipy.linalg as sci
import matplotlib.pyplot as plt
from math import sin, cos, pi, sqrt

# f_1(θ, φ) = a cos(θ) + b cos(θ + φ) − x_p
# f_2(θ, φ) = a sin(θ) + b sin(θ + φ) − y_p

# df_1/dθ = d/dθ(a cos(θ) + b cos(θ + ϕ) - x_p) = -a sin(θ) - b sin(θ + ϕ)
# df_1/dϕ = d/dϕ(a cos(θ) + b cos(θ + x) - x_p) = -b sin(θ + ϕ)

# df_2/dθ = d/dθ(a sin(θ) + b sin(θ + ϕ) - y_p) = a cos(θ) + b cos(θ + ϕ)
# df_2/dϕ = d/dx(a sin(θ) + b sin(θ + ϕ) - y_p) = b cos(θ + ϕ)

def robotarm( a, b, p, r, maxit, tol):
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


    def delta_solver(original_v: list, a:float, b:float, x_p:float,
                    y_p:float, J=Jac, f=f):

        f_vect = f(original_v, a, b, x_p, y_p)
        Jac_mat = J(original_v, a, b)

        delta_v = sci.solve(Jac_mat, -1*f_vect)
        new_v = delta_v + original_v

        return new_v

    v = guess_v
    count = 0
    while count < maxit and np.linalg.norm(f(v, a, b, x_p, y_p)) > tol:
        v = delta_solver(v, a, b, x_p, y_p)
        count += 1

    return v


def Output(a, b, p, r, maxit = 20, tol = 10**(-8)):
    
    data = []
    f_1 = lambda theta, phi, a, b, x_p: a*cos(theta) + b*cos(theta + phi) - x_p
    f_2 = lambda theta, phi, a, b, y_p: a*sin(theta) + b*sin(theta + phi) - y_p

    def f(v:list, a:float, b:float, x_p:float, y_p:float, f_1=f_1, f_2=f_2) -> list:
        # v is a matrix
        theta, phi = v[0], v[1]
        return np.array([f_1(theta, phi, a, b, x_p), f_2(theta, phi, a, b, y_p)])

    for i in range(maxit+1):
        v = robotarm(a, b, p, r, i, tol)
        theta, phi = v[0], v[1]
        data.append([theta, phi, np.linalg.norm(f(v, a, b, p[0], p[1]))/np.linalg.norm(
                    f(r, a, b, p[0], p[1])), a*cos(theta) + b*cos(theta + phi),
                    a*sin(theta) + b*sin(theta + phi)])

    or_x = [0, a*cos(data[0][0]), data[0][3]] # original vector data
    or_y = [0, a*sin(data[0][0]), data[0][4]]

    x = [0, a*cos(data[-1][0]), data[-1][3]]  # final vector data
    y = [0, a*sin(data[-1][0]), data[-1][4]]

    plt.plot(or_x, or_y, linestyle='dashed', color='blue')
    plt.plot(x, y, linestyle='solid', color='red')
    plt.plot(data[0][3], data[0][4], color='blue', marker='o')
    plt.plot(data[-1][3], data[-1][4], color='red', marker='o')
    plt.legend(['Original position of the arm', 'Final position of the arm',
                'original position of tip of arm', 'final position of tip of arm'])
    plt.show()

    df = pd.DataFrame(data, columns = ['θ (theta)', 'φ (phi)',
                    'residual ||f(x^(k))||/||f(x^(0))||', 'x-coord', 'y-coord'])
    if data[-1][2]>tol:
        out_row = 0
        min  = data[0][2]
        for row in data:
            if row[2]<min:
                min=row[2]
                out_row = row
        out_row[0] = round(out_row[0], 6)
        out_row[1] = round(out_row[1], 6)
        out_row[2] = round(out_row[2], 6)
        out_row[3] = round(out_row[3], 6)
        out_row[4] = round(out_row[4], 6)

        string = f'{out_row}'[1:-1]
        # The Warning of non-convergence
        print(f'WARNING: The sequence did not converge within the tolerance, \
        \n however the most accurate result (according to the residual) is \n {string}')
    return df

r = np.array([pi/4, -pi/2])
r_other = np.array([pi/4, pi/2])
a, b = 1.5, 1

p = np.array([1.2, 1.6])
print(Output(a, b, p, r))
print()

p = np.array([1.2, 1.6])
print(Output(a, b, p, r_other))
print()

t = sqrt((a+b)**2-1.2**2)
p = np.array([1.2, t])
print(Output(a, b, p, r))
print()

p = np.array([1.2, 2.2])
print(Output(a, b, p, r))




