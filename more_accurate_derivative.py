import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

h_list = []
for i in range(1, 18):
    h_list.append( 10**(-1*i) )


def der_sqrt(x):
    '''derivative of sqrt(x) '''
    return 1/(2 * sqrt(x))

def g(x,h):
    '''straight away calculation of g(x)'''
    g = (sqrt(x+h) - sqrt(x-h)) /(2*h)
    return g

def f(x, h, part_term_2 = der_sqrt):
    '''
    Function built according to Q2 part c
    0 <= |h| < x
    '''
    if 0 <= abs(h) < x:
        part_coeff = x**2 - (abs(h)/x)**2
        rad = sqrt(x + abs(h)/x)
        max_rad = sqrt(max(0, x - abs(h)/x ))
        term_1 = part_coeff * (rad - max_rad)/2
        term_2 = (1 - (abs(h)/x)*part_coeff)*part_term_2(x)
        tot = term_1 + term_2
        return term_1 + term_2
    else:
        raise ValueError('Incorrect input')

    

def output(x, h_list=h_list, der=der_sqrt, g=g, f=f):
    '''Returns the needed result for each 
    i as a tuple '''
    for h in h_list:
        i = h_list.index(h) + 1
        d0 = der(x)
        d1 = g(x, h)
        d2 = f(x, h)
        delta1 = d0-d1
        delta2 = d0-d2

        yield i, d0, d1, d2, abs(delta1), abs(delta2)


def print_result(num, gen=output):
    '''Prints every tuple from output
    and plots the graphs'''
    delta1_list = []
    delta2_list = []

    for e in gen(num):
        print(e)
        delta1_list.append(abs(e[4]))
        delta2_list.append(abs(e[5]))
    

    delta1_list.reverse()
    delta2_list.reverse()
    h_list.reverse()

    x = np.array(h_list)
    y = np.array(delta1_list)
    z = np.array(delta2_list)
    default_x_ticks = range(len(x))
    plt.plot(default_x_ticks, y, linestyle='solid')
    plt.plot(default_x_ticks, z, linestyle='dashed')
    plt.xticks(default_x_ticks, x)
    plt.xlabel('h values')
    plt.ylabel('Error magnitude')
    plt.legend(['d0-d1', 'd0-d2'])
    plt.yscale("log")
    plt.show()

print_result(1)
