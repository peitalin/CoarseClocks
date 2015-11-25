

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
    return phi(l, n, ti, t0) / (1 - Phi(l, n, ti, t0))

def reverse_hazard(l, n, ti, t0):
    return phi(l, n, ti, t0) / Phi(l, n, ti, t0)





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
    # Reverse hazard rate dominance
    RH_L = [p/P for p,P in zip(phiL,PhiL)]
    RH_H = [p/P for p,P in zip(phiH,PhiH)]
    plot(tt, RH_L, color=color[0], linestyle="--", label=r"$f_L(t)/F_L(t)$")
    plot(tt, RH_H, color=color[1], linestyle="-", label=r"$f_H(t)/F_H(t)$")
    ylim(0,6)
    legend(prop={'size':14})

    # Likelihood ratio dominance
    likelihood_ratio = [l/h for l,h in zip(phiL, phiH)]
    plot(tt, likelihood_ratio, color=color[1], linestyle="-", label=r"$\phi_L(t)/\phi_H(t)$")
    legend(loc="bottom left", prop={'size':14})

    # Hazard-rate dominance
    HR_L = [p/(1-P) for p,P in zip(phiL,PhiL)]
    HR_H = [p/(1-P) for p,P in zip(phiH,PhiH)]
    plot(tt, HR_L, color=color[0], linestyle="--", label=r"$f_L(t)/(1 - F_L(t))$")
    plot(tt, HR_H, color=color[1], linestyle="-", label=r"$f_H(t)/(1 - F_H(t))$")
    ylim(0,6)
    legend(loc='bottom right', prop={'size':14})

    # Dispersive Order
    DO_L = [phiL[i]-phiH[i] for i in range(len(phiL))]
    plot(tt, DO_L, color=color[0], linestyle="--", label=r"$r(t_L) - t_L$")
    legend(loc='bottom right', prop={'size':14})

    # Star Order
    SO_L = [phiL[i]/phiH[i] for i in range(len(phiL))]
    plot(tt, SO_L, color=color[0], linestyle="--", label=r"$r(t_L)/t_L$")
    legend(loc='bottom right', prop={'size':14})

    # Convex Order
    # r(t_l) is convex on S_L

    # Virtual valuations
    # "regular" if v - (1-F)/f is increasing in v
    MR_L = [ (tt[i] - (1-PhiL[i])/phiL[i] ) for i in range(len(phiL))]
    MR_H = [ (tt[i] - (1-PhiH[i])/phiH[i] ) for i in range(len(phiH))]
    plot(tt, MR_L, color=color[0], linestyle="--", label=r"$ t_L - \frac{1-F_L(t)}{f_L(t)} $")
    plot(tt, MR_H, color=color[1], linestyle="--", label=r"$ t_H - \frac{1-F_H(t)}{f_H(t)} $")
    legend(loc='bottom right', prop={'size':14})


    # Virtual costs
    MC_L = [ (tt[i] + (PhiL[i])/phiL[i] ) for i in range(len(phiL))]
    MC_H = [ (tt[i] + (PhiH[i])/phiH[i] ) for i in range(len(phiH))]
    plot(tt, MR_L, color=color[0], linestyle="-", label=r"$ t_L + \frac{F_L(t)}{f_L(t)} $")
    plot(tt, MR_H, color=color[1], linestyle="-", label=r"$ t_H + \frac{F_H(t)}{f_H(t)} $")
    legend(loc='bottom right', prop={'size':14})






def asymmetric_auctions_plots():

    # Virtual Values: AB (2003)
    def B(t, g, rf, t0=0):
        return 1 - exp( -(g-rf)*(t-t0) )

    def match(key_value, lookup_list, return_list):
        "v -> v"
        idx = np.abs(np.array(lookup_list) - key_value).argmin()
        return return_list[idx]



    ti=1
    t0=0 # t0 = ti-n
    n=1
    nH=0.8
    kappa=0.6
    # kappa=0.4

    tau = n * kappa
    linestyle='-'
    color = ['dodgerblue', 'mediumorchid', 'palevioletred', 'steelblue', 'seagreen']

    tt  = list(linspace(t0, ti, 5001))[:-1]
    ttH = list(linspace(t0-0.1, ti, 5001))[:-1]
    # time = list(linspace(ti-n, ti)) # common support of times
    # ttH = list(linspace(ti-n, ti-0.1, 5001))[:-1]


	# Vary difference in hazard rates to see difference in MR plots (information rents)
    hazrate = 2
    phiL = [phi(hazrate, n, ti, t) for t in tt]
    PhiL = [Phi(hazrate, n, ti, t) for t in tt]

    hazrate = 4
    phiH = [phi(hazrate, n, ti, t) for t in ttH]
    PhiH = [Phi(hazrate, n, ti, t) for t in ttH]

    g, rf = 3, 0.1
    B_  =  [ B(tt[i], g, rf, t0=t0) for i in range(len(phiL))]
    Bgr =  [ B(tt[i], g, rf, t0=t0)/(g-rf) for i in range(len(phiL))]

    def inv_B():
        "B -> t - t0"
        Bval = (1-exp(-l*n*kappa))/l * (g-rf)
        return match(Bval, B_, tt)


    J_L = [ ( Bgr[i] - (1-PhiL[i])/phiL[i] ) for i in range(len(phiL))]
    J_H = [ ( Bgr[i] - (1-PhiH[i])/phiH[i] ) for i in range(len(phiH))]

    plot(tt, J_L, color=color[0], linestyle="-", label=r"$ \frac{\beta(t_L - t_0)}{g-r} - \frac{1-F_L(t)}{f_L(t)} $")
    plot(tt, J_H, color=color[1], linestyle="-", label=r"$ \frac{\beta(t_H - t_0)}{g-r} - \frac{1-F_H(t)}{f_H(t)} $")


    def r(t):
        # Takes a type_L and returns a type_H with the same rank/percentile
        # t is a time/type in the PhiL distribution
        return match(PhiL[tt.index(t)], PhiH, tt)

    def k(t):
        "J_H^-1 (J_L(t))"
        return match(J_L[tt.index(t)], J_H, tt)

    r_t = [r(t) for t in tt]
    k_t = [k(t) for t in tt]


    plot(tt, tt, color='black', linestyle='--', alpha=0.4, label=r"$k_1(t)=t$")
    # plot(tt, ttH, color='black', alpha=0.5, label=r"$k_2(t)$")
    # plot(tt, J_L, color=color[0], linestyle="--", label=r"$ \frac{\beta(t_L - t_0)}{g-r} - \frac{1-F_L(t)}{f_L(t)} $")
    # plot(tt, J_H, color=color[1], linestyle="--", label=r"$ \frac{\beta(t_H - t_0)}{g-r} - \frac{1-F_H(t)}{f_H(t)} $")
    plot(tt, r_t, color=color[2], linestyle="-", label=r"$r(t) = F_H^{-1}(F_L(t_L))$")
    plot(tt, k_t, color=color[3], linestyle="-", label=r"$k(t) = J_H^{-1}(J_L(t_L))$")

    legend(loc='bottom right', prop={'size':14})
    xlabel(r"low-hazard types: $t_L$")
    ylabel(r"hi-hazard types: $t_H$")
    title(r"Comparing matched types by rank: $r(t)=F_H^{-1}(F_L(t))$ and virtual values: $k(t) = J_H^{-1}(J_L(t_L))$")




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
    g = 0.15
    r = 0.05
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




def equilibrium_delay():

    r=0.05
    g=0.15
    f=0
    tau = linspace(0, 20, 1000)
    lhigh = 0.09
    llow  = 0.03
    lavg  = (lhigh + llow) / 2

    def B(t, g=g, f=f, r=r):
        return 1 - exp(-(g+f-r)*t)

    color = ['dodgerblue', 'palevioletred', 'black']
    l = llow
    f = 0.0
    assert l <= (g+f-r)
    plot(tau, list(map(lambda t: -log((g+f-r)/(g+f-r-l*B(t)))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='--', linewidth=1, color=color[0])

    l = lhigh
    f = 0.0
    assert l <= (g+f-r)
    plot(tau, list(map(lambda t: -log((g+f-r)/(g+f-r-l*B(t)))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='-', linewidth=1, color=color[0])

    l = lhigh
    f = 0.01
    assert l <= (g+f-r)
    plot(tau, list(map(lambda t: -log((g+f-r)/(g+f-r-l*B(t)))/l, tau)),
            label=r"$f={},\, \lambda={}$".format(f,l), linestyle='-', linewidth=1, color=color[1])

    l = lavg
    f = 0.0
    assert l <= (g+f-r)
    plot(tau, list(map(lambda t: -log((g+f-r)/(g+f-r-l*B(t)))/l, tau)),
            label=r"$f={},\, \lambda={}$ (arbitraguer)".format(f,l), linestyle='-.', linewidth=1, color=color[2])


    # endogenous crash time: \tau + nk
    # n = 20
    # k = 0.75
    # cost_benefit = [(g-r)/(n*k*t) for t in tau]
    # hazard_rate_eq = l/(1-exp(-l*n*k))
    # equilibrium_tau = (g-r)/hazard_rate_eq
    # axvline(x=equilibrium_tau, linewidth=1, color='k', linestyle='--',
    #         label=r"Endogenous Burst Time: $\tau^* + \eta\kappa; \, \kappa=3/4,\, \eta=10$", alpha=0.5)

    # plot labels
    legend(loc='left')
    xlabel(r"Elapsed time since bubble began: $t - t_0$")
    ylabel(r"Waiting Time:  $\tau^*$")
    title(r"Endogenous Waiting Times Before Exit")
    ylim(-10, 0)



    def plot_endog_crash():
        plot(tau, [(g-r)/(n*k*t) for t in tau],
                label=r"$Cost-Benefit: (g-r)/\beta(\tau); \kappa=3/4$", linestyle='--', color=color[2], alpha=0.5)

        plot(tau, [hazard_rate_eq for t in tau],
                label=r"$h^*=\lambda/(1-e^{-\lambda\eta\kappa})$", linestyle='-', color=color[0], alpha=0.5)

        plot(tau, [equilibrium_tau for t in tau],
                label=r"$\tau^*$", linestyle='--', color=color[1], alpha=0.5)








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



