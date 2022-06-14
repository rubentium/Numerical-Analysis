import numpy as np
import scipy.interpolate as sc
from math import pi, log
import pandas as pd

a = 0
b = 3.1
xe = np.linspace(a, b, 1000)
ye = np.cos(pi*xe)

print(['interval (' + str(a) + ', ' + str(b)+ ')'])
print('n    err poly   err lin_spl')
data1 = []
for nn in range(1, 7):
    n = 2**nn
    xi = np.linspace(a, b, n+1)
    yi = np.cos(pi*xi)
    yp = np.polyval(np.polyfit(xi, yi, n), xe)
    yl = sc.interp1d(xi, yi, 'linear')(xe)
    ep = max(abs(ye-yp))
    el = max(abs(ye-yl))
    data1.append([n, ep, el])

df1 = pd.DataFrame(data1, columns=['n', 'err poly', 'err lin_spl'])
print(df1.to_string(index=False))

print('\n')
print(['interval (' + str(a) + ', ' + str(b)+ ')'])
data = []
for nn in range(4, 11):
    n = 2**nn
    xi = np.linspace(a, b, n+1)
    yi = np.cos(pi*xi)
    q = lambda xi: ((-5*pi*np.sin(11/10*pi))/31)*xi**2
    new_fi = np.cos(pi*xi) - q(xi)
    yl = sc.interp1d(xi, yi, 'linear')(xe) # linear
    ycc = sc.CubicSpline(xi, new_fi)(xe)
    ecc = max(abs(ye-ycc-q(xe)))
    el = max(abs(ye-yl))
    data_list = [n, el, ecc]
    data.append(data_list)


count = 1
while count < len(data):
    data[count].extend([log(data[count-1][1]/data[count][1])/log(2),
                        log(data[count-1][2]/data[count][2])/log(2)])
    count += 1

df = pd.DataFrame(data, columns=['n', 'linear s. error', 'clamped cubic s. error',
                                 'linear s. order of conv', ' clamped cubic s. order of conv'])
print(df.to_string(index=False))
