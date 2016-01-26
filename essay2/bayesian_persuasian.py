
import random
import numpy as np
from numpy import linspace, negative
from matplotlib.pyplot import *


A = [float(x) for x in linspace(-1, 1, 2000)]
W = [float(x) for x in linspace(1, 1, 2000)]
Alpha = [float(x) for x in linspace(0, 1, 1000)]


alpha = round(random.uniform(0.1,1), 1)
w = round(random.uniform(0.1,1), 1)

wstar = w * 1.5
astar = alpha * w + (1-alpha) * wstar




def u(a, w):
    "utility of sender/underwriter"
    return -(a - w)**2

def v1(a, astar):
    "utility of receiver/issuer"
    return -(a - astar)**2

def v2(a, w, b):
    "Crawford-Sobel utility of receiver"
    return -(a - (w - b))**2







def v3(mu, k):
    if mu < k/4:
        return 0
    elif k/4 <= mu < k/2:
        return (1+mu)*k/2
    else:
        return (1+mu)*mu


def v4(mu, k, l=2.25):
    if mu > 0:
        return mu ** k
    else:
        return -l * (negative(mu) ** k)
    # (with a typical a = 0.88 and l = 2.25)


def v5(mu, k, l=2.25):
    if mu < 0:
        return negative(l) * (negative(mu) ** k)
    elif mu < 1/l:
        return mu ** k
    else:
        return mu ** (w/l)
    # (with a typical a = 0.88 and l = 2.25)



"""
We are interesting in the way the equilibrium depend on the extent of preference disagreement:
    captured by alpha and w_star differences.
Increase in alpha or decrease in w_star makes preferences more aligned according to the definition in Section IV
"""

# plot(A, [u(a,w) for a in A])
# plot(A, [v(a,astar) for a in A])
# plot(A, [v2(a, w, b=w/4) for a in A], linestyle='--')
# plot(A, [v2(a, w, b=w/2) for a in A], linestyle=':')




def u2(a, mu, k=0.5, l=2.25):
    "Generalized prospect theory function with k as reference point"
    "k is the threshold parameter:"
    x = a * (1+mu) - k
    ## w = {1, 2}; so x in [0, 2], then minus 1 means x in [-1, 1]
    if x > k:
        return ((x) ** k) - a**2
    else:
        if x < 0:
            return  -l * ((-x) ** k) - a**2
        elif x >= 0:
            return x ** k - a**2


def u2(a, mu, k=0.5, l=2.25):
    "Generalized prospect theory function with k as reference point"
    "k is the threshold parameter:"
    x = a * (1+mu) - k
    ## w = {1, 2}; so x in [0, 2], then minus 1 means x in [-1, 1]
    if x > k:
        return ((x) ** k) - a**2
    else:
        if x < 0:
            return  -l * ((-x) ** k) - a**2
        elif x >= 0:
            return x ** k - a**2



A = [float(n) for n in linspace(0, 1, 1000)]
l=2.2
# k=1.1

plot(A, [u2(a, mu=0.1, k=k, l=l) for a in A], linestyle='-', label='Pr(good)=0.1')
plot(A, [u2(a, mu=0.5, k=k, l=l) for a in A], linestyle='--', label='Pr(good)=0.5')
plot(A, [u2(a, mu=0.9, k=k, l=l) for a in A], linestyle='-.', label='Pr(good)=0.9')
legend()





def u(a, mu, k=1.5):
    # (1+mu) = (mu*2 + (1-mu)*1) = E[w]
    effort = negative(a**2)
    if a*(1+mu) > k:
        return a*mu + effort
    else:
        return effort

k=1.1 # some reference level, anchor
plot(A, [u(a, mu=0.2, k=k) for a in A], linestyle='-')
plot(A, [u(a, mu=0.4, k=k) for a in A], linestyle='--')
plot(A, [u(a, mu=0.8, k=k) for a in A], linestyle='-.')




# prospect theory kinked utility
MU = linspace(-1, 1, 2000)
k=0.5
# k=0.3
plot(MU, [v5(mu, k, l=1) for mu in MU], linestyle='-')
plot(MU, [v5(mu, k, l=2.25) for mu in MU], linestyle='--')
plot(MU, [v5(mu, k, l=5) for mu in MU], linestyle='--')



plot(MU, [v3(mu, k=1) for mu in MU], linestyle='-')
plot(MU, [v4(mu, k=1) for mu in MU], linestyle='-')



