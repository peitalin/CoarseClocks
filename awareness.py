

from numpy import exp, linspace, e, log, array
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



def Phi(l, n, ti, t0):
    """ PHI(t_0 | t_i)
    l: lambda, n: awareness window, ti: agent's time, t0: time bubble began """
    # assert n > 0
    if not (ti <= t0+n):
        if round(t0+n - ti, 10) != 0:
            raise(AssertionError("ti: {} <= t0+n: {}".format(ti, t0+n)))
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
    return phi(l, n, ti, t0) / (1 - Phi(l, n, ti, t0))

def reverse_hazard(l, n, ti, t0):
    return phi(l, n, ti, t0) / Phi(l, n, ti, t0)


def F(l, n, ti, t0):
    """ PHI(t_0 | t_i)
    l: lambda, n: awareness window, ti: agent's time, t0: time bubble began """
    if not (ti <= t0+n):
        if round(t0+n - ti, 10) != 0:
            raise(AssertionError("ti: {} <= t0+n: {}".format(ti, t0+n)))
    top = exp(l*n) - exp(l*(ti - t0))
    bottom = exp(l*n) - 1
    return top/bottom

def f(l, n, ti, t0):
    "l: lambda, n: awareness window, ti: agent's time, t0: time bubble began"
    assert n > 0
    top = l*exp(l*(ti-t0))
    bottom = exp(l*n) - 1
    return top/bottom




def hazard_rates(l=1, n=1, linestyle='-', j=2):

    j=2
    ti=1
    n=1
    l=0.1
    linestyle='-'
    ltrue = [Decimal(1/10), Decimal(1/20), Decimal(1/100)]
    h, l = Decimal(1/10), Decimal(1/100)
    lprior = Decimal((h + l)/2)
    num = poisson(ltrue[j] , 1000)
    time = linspace(0, 5, 1000)  # t0 must be before ti
    tt = list(map(Decimal, linspace(ti-n, ti, 1001)))[:-1]

    # time-varying lambda
    ll = [] # lambda
    # dll = [] # derivative of lambda
    for i, t in enumerate(time):
        lambda_post = lam(time[i], num[i], lp=lprior, h=h, l=l)
        # lambda_del = d_lamb(time[i], num[i], lp=lprior, h=h, l=l)
        lprior = lambda_post
        ll.append(lambda_post)
        # dll.append(lambda_del)

    # static lambda
    color = ['dodgerblue', 'mediumorchid', 'palevioletred']
    for i, l in enumerate([0.1, 0.3]):

        tt = list(map(Decimal, linspace(ti-n, ti, 1001)))[1:-1] # t0 must be before ti
        ll = [l]*len(time)

        # h = [hazard(Decimal(lt[0]), Decimal(n), Decimal(ti), lt[1]) for lt in zip(ll,tt)]
        # plot(tt, h, color=color[i], linestyle=linestyle, label=r"$\lambda$: {}, $t_i$: {}".format(l, ti))

        rh = [reverse_hazard(Decimal(lt[0]), Decimal(n), Decimal(ti), lt[1]) for lt in zip(ll,tt)]
        plot(tt, rh, color=color[i], linestyle=linestyle, label=r"$\lambda$: {}, $t_i$: {}".format(l, ti))


def stochastic_dominance_plots():


    phiL = phiL[1:-1]
    PhiL = PhiL[1:-1]
    phiH = phiH[1:-1]
    PhiH = PhiH[1:-1]

    fL = fL[1:-1]
    FL = FL[1:-1]
    fH = fH[1:-1]
    FH = FH[1:-1]


    tt0L = list(linspace(t0, ti, nobs+1))[:-1]
    tt0H = list(linspace(t0, ti, nobs+1))[:-1]

    ttL = btimesL[1:-1]
    ttH = btimesH[1:-1]

    # Reverse hazard rate dominance
    RH_L = [p/P for p,P in zip(phiL,PhiL)]
    RH_H = [p/P for p,P in zip(phiH,PhiH)]
    plot(tt0L[1:-1], RH_L, color=color[0], linestyle="--", label=r"$\phi_L(t)/\Phi_L(t)$")
    plot(tt0H[1:-1], RH_H, color=color[1], linestyle="-", label=r"$\phi_H(t)/\Phi_H(t)$")
    ylim(0,1)
    legend(prop={'size':14})

    # Reverse hazard rate dominance
    RH_L = [p/P for p,P in zip(fL,FL)]
    RH_H = [p/P for p,P in zip(fH,FH)]
    plot(ttL, RH_L, color=color[0], linestyle="--", label=r"$f_L(t)/F_L(t)$")
    plot(ttH, RH_H, color=color[1], linestyle="-", label=r"$f_H(t)/F_H(t)$")
    ylim(0,1)

    # Likelihood ratio dominance
    likelihood_ratio = [l/h for l,h in zip(phiL, phiH)]
    plot(tt, likelihood_ratio, color=color[1], linestyle="-", label=r"$\phi_L(t)/\phi_H(t)$")
    legend(loc="lower left", prop={'size':14})

    # Hazard-rate dominance
    HR_L = [p/(1-P) for p,P in zip(phiL,PhiL)]
    HR_H = [p/(1-P) for p,P in zip(phiH,PhiH)]
    plot(tt, HR_L, color=color[0], linestyle="--", label=r"$f_L(t)/(1 - F_L(t))$")
    plot(tt, HR_H, color=color[1], linestyle="-", label=r"$f_H(t)/(1 - F_H(t))$")
    ylim(0,1)
    legend(loc='lower right', prop={'size':14})

    # Dispersive Order
    DO_L = [phiL[i]-phiH[i] for i in range(len(phiL))]
    plot(tt, DO_L, color=color[0], linestyle="--", label=r"$r(t_L) - t_L$")
    legend(loc='lower right', prop={'size':14})

    # Star Order
    SO_L = [phiL[i]/phiH[i] for i in range(len(phiL))]
    plot(tt, SO_L, color=color[0], linestyle="--", label=r"$r(t_L)/t_L$")
    legend(loc='lower right', prop={'size':14})

    # Convex Order
    # r(t_l) is convex on S_L

    # Virtual valuations
    # "regular" if v - (1-F)/f is increasing in v
    MR_L = [ (tt[i] - (1-PhiL[i])/phiL[i] ) for i in range(len(phiL))]
    MR_H = [ (tt[i] - (1-PhiH[i])/phiH[i] ) for i in range(len(phiH))]
    plot(tt, MR_L, color=color[0], linestyle="--", label=r"$ t_L - \frac{1-F_L(t)}{f_L(t)} $")
    plot(tt, MR_H, color=color[1], linestyle="--", label=r"$ t_H - \frac{1-F_H(t)}{f_H(t)} $")
    legend(loc='lower right', prop={'size':14})


    # Virtual costs
    MC_L = [ (tt[i] + (PhiL[i])/phiL[i] ) for i in range(len(phiL))]
    MC_H = [ (tt[i] + (PhiH[i])/phiH[i] ) for i in range(len(phiH))]
    plot(tt, MR_L, color=color[0], linestyle="-", label=r"$ t_L + \frac{F_L(t)}{f_L(t)} $")
    plot(tt, MR_H, color=color[1], linestyle="-", label=r"$ t_H + \frac{F_H(t)}{f_H(t)} $")
    legend(loc='lower right', prop={'size':14})






def asymmetric_auctions_plots():

    color = ['dodgerblue', 'mediumorchid', 'palevioletred', 'steelblue', 'seagreen']

    def B(t, g, rf, t0=0):
        "Bubble component of price (fraction)"
        bubble = 1 - exp( -(g-rf)*(t-t0) )
        assert bubble <= 1
        return bubble

    def match(key_value, lookup_list, return_list):
        "v_L -> v_H"
        idx = np.abs(np.array(lookup_list) - key_value).argmin()
        return return_list[idx]

    def round_up(x, places):
        return round(x, places) if round(x, places) >= x else round(x + 1/10**places, places)

    def tau_star(hazrate, g, rf):
        assert hazrate/(1-exp(-(hazrate*n*kappa))) > (g-rf)
        # h(t) > (g-rf) otherwise nan (never crashes)
        return 1/(g-rf) * log(hazrate/(hazrate - (g-rf)*(1-exp(-(hazrate*n*kappa))))) - n*kappa

    def g_upper_bound(hazrate, rf, kappa, n):
        "Returns max growth rate for endogenous crash, otherwise diverges"
        return hazrate/(1-exp(-(hazrate*n*kappa))) + rf - rf/100000

    def r(t, FL, FH, btimesL, btimesH):
        "Match type t_L with t_H of same rank in F_H (CDF of burst times)"
        # Takes a type_L and returns a type_H with the same rank/percentile
        # t is a time/type in the PhiL distribution
        return match(FL[btimesL.index(t)], FH, btimesH)

    def j(t, J_L, J_H, btimesL, btimesH):
        "Match type t_L with t_H with same virtual valuation"
        "J_H^-1 (J_L(t))"
        return match(J_L[btimesL.index(t)], J_H, btimesH)

    def printJ(t):
        idx1 = np.abs(array(btimesL) - t).argmin()
        idx2 = np.abs(array(btimesH) - t).argmin()
        print(r"J_L = {}".format(J_L[idx1]))
        print(r"J_H = {}".format(J_H[idx2]))



    alphas = [0.4, 0.6, 0.8]
    n_params = [15, 20, 25]
    kappa = 0.5
    n=25
    nobs = 2000

    plttype = "kappa"
    # plttype = "tau"
    # plttype = "n"

    if plttype == "tau":
        iter_params = tau_params = [kappa, kappa]
    elif plttype == "kappa":
        iter_params = k_params = [0.3, 0.6]
    elif plttype == "n":
        iter_params = n_params = [10, 25, 40]


    for num, kappa in enumerate(iter_params):

        rf = 0.01
        g = .06
        hazrateH = .04
        hazrateL = .02

        # g = g_upper_bound(hazrateL, rf, kappa) - 0.001
        print("g upper bound: {}".format(g_upper_bound(hazrateL, rf, kappa, n)))
        assert g < g_upper_bound(hazrateL, rf, kappa, n)

        ti = n
        t0 = ti-n
        # ## Distribution of bubble begin times: t0
        tt0L = list(linspace(t0, ti, nobs+1))[:-1]
        tt0H = list(linspace(t0, ti, nobs+1))[:-1]

        ## Bubble start time posteriors: Phi(t0|ti)
        phiL = [phi(hazrateL, n, ti, t0) for t0 in tt0L]
        PhiL = [Phi(hazrateL, n, ti, t0) for t0 in tt0L]

        phiH = [phi(hazrateH, n, ti, t0) for t0 in tt0H]
        PhiH = [Phi(hazrateH, n, ti, t0) for t0 in tt0H]




        if ti < n*kappa:
            b0 = lambda tau: t0 + tau + n*kappa
            bi = lambda tau: ti + tau + n*kappa
        else:
            b0 = lambda tau: t0 + tau
            bi = lambda tau: ti + tau
        # Levin notes: Bubbles and Crashes page 6



        # burst times
        tauL = tau_star(hazrateL, g, rf)
        tauH = tau_star(hazrateH, g, rf)
        # tauH = tau_star(hazrateH, g, rf) + tauL
        # Plus tau: arbitraguer sells out after tau periods, meaning lender
        # becomes aware tau periods after the arbitrageur

        print("tauL: {}\ntauH: {}".format(tauL, tauH))
        btimesL = list(linspace(b0(tauL), bi(tauL), nobs+1))[:-1]
        # Distributional Shift
        btimesH = list(linspace(b0(tauH), bi(tauH), nobs+1))[:-1]

        if 0:
            # Distribution Stretch
            # btimesH = list(linspace(b0(tauL), bi(tauH), nobs+1))[:-1]
            fH = [f(hazrateH, bi(tauH) - b0(tauL), bi(tauH), t) for t in btimesH]
            FH = [F(hazrateH, bi(tauH) - b0(tauL), bi(tauH), t) for t in btimesH]


        fL = [f(hazrateL, n, bi(tauL), t) for t in btimesL]
        FL = [F(hazrateL, n, bi(tauL), t) for t in btimesL]
        fH = [f(hazrateH, n, t) for t in btimesH]
        FH = [F(hazrateH, n, bi(tauH), t) for t in btimesH]

        if plttype == 'tau' and num == 1:
            fH = [f(hazrateH, n, bi(tauH+tauL), t) for t in btimesH]
            FH = [F(hazrateH, n, bi(tauH+tauL), t) for t in btimesH]


        if 0:
            # Awareness distributions share the same support however, the posterior burst distributions do not
            plot(btimesL, FL, color=color[0], linestyle='-', label=r"Posterior Burst Times: $F_L(t|t_L)$")
            plot(btimesH, FH, color=color[1], linestyle='-', label=r"Posterior Burst Times: $F_H(t|t_H)$")
            # plot(btimesH, (1-array(FH)), color=color[1], linestyle='-', label=r"Loan supply: $1-F_H(t|t_H)$")
            # plt.axvline(t0+tauL+n*kappa)
            plot(btimesL, fL, color=color[0], linestyle='-', label=r"Posterior Burst Times: $F_L(t|t_L)$")
            plot(btimesH, fH, color=color[1], linestyle='-', label=r"Posterior Burst Times: $F_H(t|t_H)$")
            plot(tt0L,  PhiL, color=color[0], linestyle='--',  label=r"Awareness CDF: $\Phi_L(t_0|t_L)$")
            plot(tt0H,  PhiH, color=color[1], linestyle='--',  label=r"Awareness CDF: $\Phi_H(t_0|t_H)$")
            legend(loc="lower right")
            title(r"Awareness and Burst Time distributions" +
                "($\kappa={}$, $\lambda_H={}$, $\lambda_L={}$)".format(kappa, hazrateH,hazrateL))
            xlabel("Time")

            plot(btimesH, array(FL) - array(FL)**2, label=r"Adverse Selection cost")
            plot(btimesL, np.convolve(1-array(fH), fL, mode='same'))
            G = 1 - array(FH)
            plot(btimesH, [g*f for g,f in zip(G, FH)])


        B_L =  [ B(btimesL[i], g, rf, t0=t0) for i in range(len(fL))]
        B_H =  [ B(btimesH[i], g, rf, t0=t0) for i in range(len(fH))]

        # HR_L = [ hazrateL/(1-exp(-hazrateL*n*kappa)) for i in range(len(fL))]
        # HR_H = [ hazrateH/(1-exp(-hazrateH*n*kappa)) for i in range(len(fH))]
        # plot(btimesH, [(g-rf)/B(n*kappa + t, g=g,rf=rf) for t in btimesH])
        RH_L = [ (1-FL[i])/fL[i] for i in range(len(fL))]
        RH_H = [ (1-FH[i])/fH[i] for i in range(len(fH))]

        J_L = array(RH_L) - array(B_L)/(g-rf)
        J_H = array(RH_H) - array(B_H)/(g-rf)

        # J_L = array(RH_L) - array(B_L)/(g-rf)
        # J_H = array(RH_H) - array(B_H)/(g-rf)


        if 0:
            plot(btimesL, J_L, color=color[0], linestyle="-",
                    label=r"$type:t_L, \kappa:{}, \eta:{}$".format(kappa, n))
            plot(btimesL, J_H, color=color[1], linestyle="-",
                    label=r"$type:t_H, \kappa:{}, \eta:{}$".format(kappa, n))
            xlabel("Price Burst Times")
            ylabel(r"Virtual Valuation: $J(t)$")
            title(r"Virtual Valuations: $J_L(t_L)= \frac{\beta(t_L - t_0)}{g-r} - \frac{1-F_L(t)}{f_L(t)}$")
            legend(loc="lower right")


        r_t = [r(t, FL, FH, btimesL, btimesH) for t in btimesL]
        j_t = [j(t, J_L, J_H, btimesL, btimesH) for t in btimesL]

        "Plot t_l on the x-axis"
        plt.subplot(1, len(iter_params), num+1)
        plt.plot(btimesL, btimesH, color='black', linestyle='--', alpha=0.4, label=r"$j_1(t)$")
        plt.plot(btimesL, btimesL, color='black', linestyle=':', linewidth=1, alpha=0.99, label=r"$t_H=t_H, 45^o$")

        plt.plot(btimesL, r_t, color=color[2], linestyle="-", linewidth=1, label=r"$r(t) = F_L^{-1}(F_H(t_H))$")
        plt.plot(btimesL, j_t, color=color[3], linestyle="-", linewidth=1, label=r"$j(t) = MR_L^{-1}(MR_H(t_H))$")

        endog_crash = t0 + tauL + n*kappa
        plt.axvline(endog_crash, linewidth=1, linestyle='-.', alpha=0.9, color='black',
                    label=r"burst time: $t_0 + \tau^* + \eta*\kappa$")

        plt.axhline(btimesH[0], linestyle='-', color='grey', linewidth=1.5)
        plt.axvline(btimesL[0], linestyle='-', color='grey', linewidth=1.5)
        plt.xlim(0, btimesL[-1])
        plt.ylim(0, btimesH[-1])


        if 0 :
            "high types on x-axis"
            r_t = [r(t, FH, FL, btimesH, btimesL) for t in btimesH]
            j_t = [j(t, J_H, J_L, btimesH, btimesL) for t in btimesH]

            plt.subplot(1, len(iter_params), num+1)
            plt.plot(btimesH, btimesL, color='black', linestyle='--', alpha=0.4, label=r"$j_1(t)$")
            plt.plot(btimesH, btimesH, color='black', linestyle=':', linewidth=1, alpha=0.99, label=r"$t_H=t_H, 45^o$")

            plt.plot(btimesH, r_t, color=color[2], linestyle="-", linewidth=1, label=r"$r(t) = F_L^{-1}(F_H(t_H))$")
            plt.plot(btimesH, j_t, color=color[3], linestyle="-", linewidth=1, label=r"$j(t) = MR_L^{-1}(MR_H(t_H))$")

            endog_crash = t0 + tauL + n*kappa
            plt.axvline(endog_crash, linewidth=1, linestyle='-.', alpha=0.9, color='black',
                        label=r"burst time: $t_0 + \tau^* + \eta*\kappa$")

            plt.axhline(btimesL[0], linestyle='-', color='grey', linewidth=1.5)
            plt.axvline(btimesH[0], linestyle='-', color='grey', linewidth=1.5)
            plt.xlim(0, btimesH[-1])
            plt.ylim(0, btimesL[-1])





    "Plot 1"
    plt.subplot(1, len(iter_params), 1)
    legend(loc='best', prop={'size':12})
    plt.xlabel(r"Low-hazard types: $t_L$")
    plt.ylabel(r"High-hazard types: $t_H$")

    if plttype=='kappa':
        plt.title(r"kappa = {}".format(k_params[0]))
    elif plttype=='tau':
        plt.title(r"$\tau_L^*$")
    else:
        plt.title(r"kappa = {}; $\tau_L^*$".format(k_params[0]))


    "Plot 2"
    plt.subplot(1, len(iter_params), 2)
    legend(loc='lower right', prop={'size':12})
    plt.xlabel(r"Low-hazard types: $t_L$")
    plt.ylabel(r"High-hazard types: $t_H$")

    if plttype=='kappa':
        plt.title(r"kappa = {}".format(k_params[1]))
    elif plttype=='tau':
        plt.title(r"$\tau_L^* + \tau_H^*$")
    else:
        plt.title(r"kappa = {}; $\tau_L^* + \tau_H^*$".format(k_params[1]))


    # "Plot 3"
    # plt.subplot(1, len(iter_params), 3)
    # legend(loc='lower right', prop={'size':12})
    # plt.xlabel(r"Low-hazard types: $t_L$")
    # plt.ylabel(r"High-hazard types: $t_H$")

    # if plttype=='kappa':
    #     plt.title(r"kappa = {}".format(k_params[2]))
    # elif plttype=='tau':
    #     plt.title(r"$\tau_L^* + \tau_H^*$")
    # else:
    #     plt.title(r"kappa = {}; $\tau_L^* + \tau_H^*$".format(k_params[2]))

    plt.suptitle(r"Comparing matched types by rank: $r(t)=F_H^{-1}(F_L(t))$ and virtual values: $j(t) = MR_H^{-1}(MR_L(t_L))$")











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
    rnorm = normalvariate
    color = ['dodgerblue', 'mediumorchid', 'palevioletred']
    dtime = linspace(0, 10, 1000)
    g = 0.12
    r = 0.02
    v = volatility = 0.025

    price_mu = [exp((g-r)*t) for t in dtime]
    price_random = [exp((g-r)*t) + rnorm(0, v) for t in dtime]
    plot(dtime, price_mu, color=color[0], linestyle='--', label=r"$p_t=e^{gt}$")
    plot(dtime, price_random, color=color[0], linestyle='-', linewidth=1, alpha=0.4)

    # (1 - (exp(-0.06) - exp(-0.15))) * exp(0.15)
    price_B_mu = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) for t in dtime]
    price_B_random = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) + rnorm(0, v) for t in dtime]
    plot(dtime, price_B_mu, color=color[2], linestyle='--', label=r"$(1-\beta(t-t_0))p_t$")
    plot(dtime, price_B_random, color=color[2], linestyle='-', linewidth=1, alpha=0.4)

    # before bubble
    pretime = linspace(-5,0, 500)
    preprice_mu = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) for t in pretime]
    preprice_random = [(exp((g-r)*t) - exp((g-r-0.04)*t) + exp(0)) + rnorm(0, v) for t in pretime]
    plot(pretime, preprice_mu, color=color[1], linestyle='--')
    plot(pretime, preprice_random, color=color[1], linestyle='-', linewidth=1, alpha=0.4)


    ylabel(r"$p_t$")
    xlabel("time")
    xlim(-5,10)

    text(7, 1.8, r"$p_t=e^{gt}$", fontsize='14')
    text(7, 1.25, r"$(1-\beta(t-t_0)) p_t$", fontsize='14')
    axvline(x=0, linewidth=1, color='k', linestyle='--',
            label=r"Bubble Start Time", alpha=0.5)
    axhline(y=0.2, xmin=1/3, xmax=2/3, linewidth=1, color='k', linestyle='-',
            label=r"Bubble Start Time")

    tick_params(axis='x', labelbottom='off')
    tick_params(axis='y', labelleft='off')




def degree_of_preemption():

    rf=0.02
    g=0.10
    fee=0
    xis = linspace(0, 20, 1000)
    lhigh = 0.08
    llow  = 0.05
    lavg  = (lhigh + llow) / 2

    def B(t, g, rf, t0=0):
        "Bubble component of price (fraction)"
        bubble = 1 - exp( -(g-rf)*(t-t0) )
        assert bubble <= 1
        return bubble


    color = ['dodgerblue', 'mediumorchid', 'palevioletred', 'steelblue', 'seagreen']

    # for a given crash date (xi), arbitrageur hastens his sellout time by preemptime(xi)
    preemptime = lambda xi: -1/l * log((g+fee-rf)/(g+fee-rf-l*B(xi, g, rf)))

    l = llow
    fee = 0.0
    plot(xis, list(map(preemptime, xis)),
            label=r"$f={},\, \lambda={}$".format(fee,l), linestyle='--', linewidth=1, color=color[0])

    l = lhigh
    fee = 0.0
    plot(xis, list(map(preemptime, xis)),
            label=r"$f={},\, \lambda={}$".format(fee,l), linestyle='-', linewidth=1, color=color[0])

    l = lhigh
    fee = 0.01
    plot(xis, list(map(preemptime, xis)),
            label=r"$f={},\, \lambda={}$".format(fee,l), linestyle='-', linewidth=1, color=color[2])

    l = lavg
    fee = 0.0
    plot(xis, list(map(preemptime, xis)),
            label=r"$f={},\, \lambda={}$ (arbitraguer)".format(fee,l), linestyle='-.', linewidth=1, color='black')


    # plot labels
    legend(loc='lower left')
    xlabel(r"$\xi$ (an exogenous bubble burst time)")
    ylabel(r"Degree of Preemption:  $\tau^* - \xi$")
    title(r"Degree of Preemption (Reduction in total waiting time) $g={}$".format(g))
    ylim(-20, 0)
    xlim(0, 20)













def plot_lambdas():
    # (h-l) / (1 + ((h-o)/(o-l)) * exp((h-l)*t) * exp(-n*log(h/l)))
    from numpy.random import poisson

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
    # PICK j=1 or j=0
    j=0


    ti=0
    n=10
    ltrue = [Decimal(0.2), Decimal(0.01)]

    for j in [1,0]:
        for i in range(50):
            color = ['dodgerblue', 'palevioletred']
            # h, l = ltrue[0], ltrue[1]
            h, l = Decimal(0.1), Decimal(0.02)
            lprior = Decimal((h + l)/2)
            # Draw random samples
            num = poisson(ltrue[j] , n*100)
            # num = list(accumulate(num))
            time = linspace(0, n, n*100)  # t0 must be before ti
            # yy = [lam(t,n) for t, n in zip(time, num)]
            # effective hazard rate is num-obs/num-periods

            ll = []
            for i, t in enumerate(time):
                lambda_post = lam(t, num[i], lp=lprior, h=h, l=l)
                lprior = lambda_post
                ll.append(lambda_post)

            plot(time, ll, color=color[j], alpha=0.1)
        # plot for labelling purpose only
        plot(time, ll, color=color[j], alpha=0.1, label=r'$\lambda={}$'.format(round(ltrue[j], 3)))


# legend(); ylabel(r'$\lambda$'); xlabel('Time Periods'); title(r"Posterior Belief of $\lambda$")
########
# For a long enough awareness window, beliefs eventually converge back to lambda=l
# Shorter awareness windows relative to hazard rate (better synchnronization) leads to more stock recalls
# More obvious arbitrage opportunities are more likely to cause coarse clocks adverse selection





# clock-games are isomorphic to a multi-unit reverse first price auction with a stochastic outside option.

# wolfram comparative statics
# d/df: swapped r with f
# \xi - 1/\lambda * log((g+r-f) / (g+r-f-\lambda*B))

# d/d\lambda
# \xi - 1/x * log((g+f-r) / (g+f-r-x*B))
# http://www.wolframalpha.com/input/?i=%5Cxi+-+1%2Fx+*+log%28%28g%2Bf-r%29+%2F+%28g%2Bf-r-x*B%29%29


# 1/(g+f-r) * log( x / (x - (g+f-r)(1 - e^(x*n*k))))

# 1/(g+x-r) * log( l / (l - (g+x-r)(1 - e^(l*n*k))))



