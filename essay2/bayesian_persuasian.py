
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


def a3(mu, k):
    if mu < k/4:
        return 0
    elif k/4 <= mu < k/2:
        return (1+mu)*k/2
    else:
        return (1+mu)*mu

def v(a, w, k):
    if a*w > k:
        return a*w - a**2
    else:
        return -(a**2)


def a(mu):
    w = (1+mu)
    if v


plot(A, [v(a, w=1, k=1.2) for a in A], linestyle='--')
plot(A, [v(a, w=2, k=1.2) for a in A], linestyle='-')

for ww in linspace(1.1, 1.9, 9):
    plot(A, [v(a, w=ww, k=1.2) for a in A], linestyle=':')




"""
We are interesting in the way the equilibrium depend on the extent of preference disagreement:
    captured by alpha and w_star differences.
Increase in alpha or decrease in w_star makes preferences more aligned according to the definition in Section IV
"""




A = [float(n) for n in linspace(0, 1, 1000)]
l=2.2



def u2(a, w, k=0.5, l=2.25):
    "Generalized prospect theory function with k as reference point"
    "k is the threshold parameter:"
    x = a * w - k
    if x >= 0:
        return x**k - a**2
    elif x < 0:
            return  -l * (-x)**k - a**2


def u2(a, w, k=0.5, l=2.25):
    "Generalized prospect theory function with k as reference point"
    "k is the threshold parameter:"
    x = a * w - k
    if x >= 0:
        return x - a**2
    elif x < 0:
            return  -l * (-x)**k - a**2

# utility function must be function of actions and state ONLY
plot(A, [u2(a, w=1, k=k, l=l) for a in A], linestyle='-', label='w=1')
plot(A, [u2(a, w=2, k=k, l=l) for a in A], linestyle='--', label='w=2')
for ww in linspace(1.2, 1.8, 4):
    plot(A, [u2(a, w=ww, k=k, l=l) for a in A], linestyle='-.', label='w={}'.format(ww))
legend()
# for low levels of k, both utility functions have a positive point
# for high levels of k, only w=2 has a position point.


def action(a, mu, k):
    x = a * (1+mu) - k
    if x >= 0:
        return x**k - a**2
    elif x < 0:
            return  -l * (-x)**k - a**2
# return probability (mu) weighted utility and characterize maximum (a*)




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





# tenure example in Gentzkow and Kamenica (2011)
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



