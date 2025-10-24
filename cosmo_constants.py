import numpy as np

Tcmb = 2.725
h = 0.6766
H0 = 100*h
ombh2 = 0.02242
omb = ombh2/h**2
omch2 = 0.11933
omc = omch2/h**2
om = omb+omc
ns = 0.965
c = 2.997e5
c_si = c*1e3
arcmin2rad = 0.000291
planckh = 6.62607e-34
k = 1.38e-23
Tcmb_uK = Tcmb*1e6
GHz2Hz = 1e9