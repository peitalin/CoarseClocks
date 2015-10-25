

from numpy import exp, linspace, e, log
from numpy.random import poisson

from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
# from matplotlib.pylab import plot, xlim, ylim, legend, ylabel, xlabel, title,
from matplotlib.pylab import *
from random import normalvariate
import seaborn as sb

import pandas as pd
import decimal
from decimal import *
from itertools import accumulate

pwd = '/Users/peitalin/Data/IPO/NASDAQ/'


def impresario_price_amendments():
    df = pd.read_csv(pwd + "df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
    # early amenders
    dfa = df[df.percent_first_price_update.notnull()]
    # 60 impresarios who amend up, then back down
    dfi = dfa[(dfa.percent_first_price_update > dfa.percent_final_price_revision) & (dfa.percent_first_price_update >= 0)]
    ## Underwriters who revise down a little, then down as lot
    # dfi = dfa[dfa.percent_first_price_update > dfa.percent_final_price_revision]

    dat = dfi
    plot(dat.delay_in_price_update, dat.percent_first_price_update, '.', label="1st price update")
    plot(dat.delay_in_price_update, dat.percent_final_price_revision, '.', label="Final price update")
    plot(dat.delay_in_price_update, dat.close_return, '.', label="Close Return")

    xlim(0, 1.1)
    ylim(-100, 200)
    ylabel("% Percent Offer Price Change")
    xlabel("Delay in 1st price amendment (0 ~ 1)")
    legend()



def Phi(l, n, ti, t0):
    """ PHI(t_0 | t_i)
    l: lambda, n: awareness window, ti: agent's time, t0: time bubble began """
    # assert n > 0
    # assert t0 <= ti <= t0 + n
    top = exp(l*n) - exp(l*(ti - t0))
    bottom = exp(l*n) - 1
    return top/bottom

def phi(l, n, ti, t0):
    "l: lambda, n: awareness window, ti: agent's time, t0: time bubble began"
    assert n > 0
    top = l*exp(l*(ti-t0))
    bottom = exp(l*n) - 1
    return top/bottom

def hazard(l, n, ti, t0):
    top = phi(l, n, ti, t0)
    bottom = 1 - Phi(l, n, ti, t0)
    return top/bottom

def d_hazard(l, ti, t0):
    top = l**2 * exp(-l*(ti-t0))
    bot = (1 - exp(-l*(ti-t0)))**2
    return top/bot

def d_lamb(t, num, lp, h=2,l=1):
    t, lp, h, l = map(Decimal, [t, lp, h, l])
    num = Decimal(int(num))
    top = exp((h-l)*t) * (h - l)**2 * (h - lp)
    low = (h/l)**num * (lp - l)
    bottom = low * (1 + (exp((h-l)*t) * (h - lp)) / low)**2
    return top/bottom


def hazard_rates(l=1, n=1, linestyle='--', j=2):

    ltrue = [Decimal(1/10), Decimal(1/20), Decimal(1/100)]
    h, l = Decimal(1/10), Decimal(1/100)
    lprior = Decimal((h + l)/2)
    num = poisson(ltrue[j] , 1000)
    time = linspace(0, 5, 1000)  # t0 must be before ti
    tt = list(map(Decimal, linspace(ti-n, ti, 1001)))[:-1]


    ll = [] # lambda
    # dll = [] # derivative of lambda
    for i, t in enumerate(time):
        lambda_post = lam(time[i], num[i], lp=lprior, h=h, l=l)
        # lambda_del = d_lamb(time[i], num[i], lp=lprior, h=h, l=l)
        lprior = lambda_post
        ll.append(lambda_post)
        # dll.append(lambda_del)


    color = ['dodgerblue', 'mediumorchid', 'palevioletred']
    for i, ti in enumerate([0.75, 1, 1.25]):

        tt = list(map(Decimal, linspace(ti-n, ti, 1001)))[:-1]
        # t0 must be before ti
        h = [hazard(Decimal(lt[0]), Decimal(n), Decimal(ti), lt[1]) for lt in zip(ll,tt)]
        dh = [d_hazard(Decimal(lt[0]), Decimal(ti), lt[1]) for lt in zip(ll,tt)]
        plot(tt, h, color=color[i], linestyle=linestyle, label=r"$\lambda$: {}, $t_i$: {}".format(l, ti))

    xlim(-0.5, 1.5)
    ylim(0, 4)
    legend()


def phis(l=1, n=1, linestyle='--'):

    # color = ['black', 'gray', 'silver']
    color = ['dodgerblue', 'mediumorchid', 'palevioletred']
    agent = ['i', 'j', 'k']
    linestyles = ['-', '--', '-.']
    linestyles = ['--', '--', '--']
    l, n = 1, 1

    for i, ti in enumerate([1, 1.5, 2]):

        tt = linspace(ti-n, ti, 501)  # t0 must be before ti
        y = [phi(l, n, ti, t0) for t0 in tt]
        y[0], y[-1] = 0, 0
        plot(tt, y, color=color[i], linestyle=linestyles[i], label=r"$\lambda$: {}, $t_{}$: {}".format(l, agent[i], ti))
        text(ti-1.05, 1.6, r'$\phi(t_0|t_{})$'.format(agent[i]), fontsize='15')
        text(ti+0.02, 0.02, r"$t_{}$".format(agent[i]), fontsize='14')

    # plot t_0 support
    plot(linspace(0.25, 1,100), [0.01 for x in linspace(0.25,1,100)], color=color[0])
    plot([0.25,0.25],[0.01,0.03], color=color[0])
    text(0.02, 0.02, r"$t_i - \eta$".format(agent[i]), fontsize='14')
    text(0.26, 0.02, r"$t_i - \eta\kappa$".format(agent[i]), fontsize='14')
    annotate(r"$t_0^{supp}(t_i)$", xy=(0.4, 0), xytext=(0.6, 0.03), fontsize='12',
            arrowprops={'arrowstyle': '->', 'facecolor':'black'})
    xlim(-0.5, 2.5)
    ylim(0, max(y)+1)
    # legend()
    title(r'Posterior beliefs of $t_0$ (density), $\kappa=3/4$')



def plot_figure1_price_path():

    from random import normalvariate
    color = ['dodgerblue', 'mediumorchid', 'palevioletred']
    dtime = linspace(0, 2, 400)
    g = 0.15
    r = 0.05

    price_mu = [exp((g-r)*t) for t in dtime]
    price_random = [exp((g-r)*t) + rnorm(0, 0.01) for t in dtime]
    plot(dtime, price_mu, color=color[0], linestyle='--', label=r"$p_t=e^{gt}$")
    plot(dtime, price_random, color=color[0], linestyle='-', linewidth=1, alpha=0.4)

    # (1 - (exp(-0.06) - exp(-0.15))) * exp(0.15)
    price_B_mu = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) for t in dtime]
    price_B_random = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) + rnorm(0, 0.01) for t in dtime]
    plot(dtime, price_B_mu, color=color[2], linestyle='--', label=r"$(1-\beta(t-t_0)p_t$")
    plot(dtime, price_B_random, color=color[2], linestyle='-', linewidth=1, alpha=0.4)

    # before bubble
    pretime = linspace(-1,0, 200)
    preprice_mu = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) for t in pretime]
    preprice_random = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) + rnorm(0, 0.01) for t in pretime]
    plot(pretime, preprice_mu, color=color[1], linestyle='--')
    plot(pretime, preprice_random, color=color[1], linestyle='-', linewidth=1, alpha=0.4)

    ylabel(r"$p_t$")
    xlabel("time")

    text(1.55, 1.12, r"$p_t=e^{gt}$", fontsize='14')
    text(1.55, 1.03, r"$(1-\beta(t-t_0)) p_t$", fontsize='14')
    axvline(x=0, linewidth=1, color='k', linestyle='--',
            label=r"Bubble Start Time", alpha=0.5)

# hazard_rates(l=1, n=1, linestyle='-')
# hazard_rates(l=2, n=1, linestyle='--')
# phis(l=1, n=1, linestyle='-')
# phis(l=2, n=1, linestyle='--')
def equilibrium_delay():

    r=0.05
    g=0.15
    f=0
    tau = linspace(0, 2, 500)

    def beta(t, g=g, f=f, r=r):
        return 1 - exp(-(g+f-r)*t)

    color = ['dodgerblue', 'palevioletred', 'black']
    l = 0.08
    f = 0.0
    plot(tau, list(map(lambda B: -log((g+f-r)/(g+f-r-l*B))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='--', color=color[0])

    l = 0.14
    f = 0.0
    plot(tau, list(map(lambda B: -log((g+f-r)/(g+f-r-l*B))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='-', color=color[0])

    l = 0.14
    f = 0.02
    plot(tau, list(map(lambda B: -log((g+f-r)/(g+f-r-l*B))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='-', color=color[1])

    l = 0.11
    f = 0.0
    plot(tau, list(map(lambda B: -log((g+f-r)/(g+f-r-l*B))/l, tau)),
            label=r"$f={},\, \lambda={}$ (arbitraguer)".format(f,l), linestyle='-.', color=color[2], alpha=0.5)


    # endogenous crash time: \tau + nk
    n = 10
    k = 0.75
    cost_benefit = [(g-r)/(n*k*t) for t in tau]
    hazard_rate_eq = l/(1-exp(-l*n*k))
    equilibrium_tau = (g-r)/hazard_rate_eq
    axvline(x=equilibrium_tau, linewidth=1, color='k', linestyle='--',
            label=r"Endogenous Burst Time: $\tau^* + \eta\kappa; \, \kappa=3/4,\, \eta=10$", alpha=0.5)

    # plot labels
    legend(loc='left')
    xlabel(r"Elapsed time since bubble began: $t - t_0$")
    ylabel(r"Waiting Time:  $\tau^* - \xi$")
    title(r"Degree of Preemption (reduction in waiting time)")
    ylim(-30, 0)



    def plot_endog_crash():
        plot(tau, [(g-r)/(n*k*t) for t in tau],
                label=r"$Cost-Benefit: (g-r)/\beta(\tau); \kappa=3/4$", linestyle='--', color=color[2], alpha=0.5)

        plot(tau, [hazard_rate_eq for t in tau],
                label=r"$h^*=\lambda/(1-e^{-\lambda\eta\kappa})$", linestyle='-', color=color[0], alpha=0.5)

        plot(tau, [equilibrium_tau for t in tau],
                label=r"$\tau^*$", linestyle='--', color=color[1], alpha=0.5)



# clock-games are isomorphic to a multi-unit reverse first price auction with a stochastic outside option.

# wolfram comparative statics
# d/df: swapped r with f
# \xi - 1/\lambda * log((g+r-f) / (g+r-f-\lambda*B))

# d/d\lambda
# \xi - 1/x * log((g+f-r) / (g+f-r-x*B))
# http://www.wolframalpha.com/input/?i=%5Cxi+-+1%2Fx+*+log%28%28g%2Bf-r%29+%2F+%28g%2Bf-r-x*B%29%29





def lam(t, num, lp=(2+1)/2, h=2, l=1):
    "lp: lam prior, h: lam_high, l: lam_low"

    def ln(x):
        return x.ln()

    # Must use Decimal, otherwise rounding error in prior, l=lp
    t, lp, h, l = map(Decimal, [t, lp, h, l])
    num = Decimal(int(num))
    top = h - l
    bottom = 1 + (h - lp)/(lp - l) * exp((h-l)*t - num*(h/l))
    return l + top/bottom


def plot_lambdas():
    # (h-l) / (1 + ((h-o)/(o-l)) * exp((h-l)*t) * exp(-n*log(h/l)))
    from numpy.random import poisson

    # j=0

    # ti=0
    # n=1
    # ltrue = [Decimal(1/10), Decimal(1/20), Decimal(1/200)]
    # h, l = Decimal(1/10), Decimal(1/200)
    # lprior = Decimal((h + l)/2)
    # num = poisson(ltrue[j] , 1000)
    # time = linspace(0, 10, 1000)  # t0 must be before ti
    # tt = list(map(Decimal, linspace(ti-n, ti, 1001)))[:-1]

    for i in range(150):
        color = ['dodgerblue', 'palevioletred']
        ltrue = [Decimal(1/2), Decimal(1/200)]
        h, l = ltrue[0], ltrue[1]
        # h, l = Decimal(1/10), Decimal(1/20)
        lprior = Decimal((h + l)/2)
    # Draw random samples
        num = poisson(ltrue[j] , 1000)
    # num = list(accumulate(num))
        time = linspace(0, 5, 1000)  # t0 must be before ti
    # yy = [lam(t,n) for t, n in zip(time, num)]
    # effective hazard rate is num-obs/num-periods

        ll = []
        for i, t in enumerate(time):
            lambda_post = lam(time[i], num[i], lp=lprior, h=h, l=l)
            lprior = lambda_post
            ll.append(lambda_post)

        plot(time, ll, color=color[j], alpha=0.1)
    plot(time, ll, color=color[j], alpha=0.1, label=r'$\lambda={}$'.format(round(ltrue[j], 3)))


# legend(); ylabel(r'$\lambda$'); xlabel('Time Periods'); title(r"Posterior Belief of $\lambda$")
########
# For a long enough awareness window, beliefs eventually converge back to lambda=l
# Shorter awareness windows relative to hazard rate (better synchnronization) leads to more stock recalls
# More obvious arbitrage opportunities are more likely to cause coarse clocks adverse selection







