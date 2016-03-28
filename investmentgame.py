
from matplotlib.pyplot import *
from numpy import sqrt, linspace, pi
from scipy.stats import norm
from seaborn import color_palette

kolor = color_palette('deep')
kolor.pop(1) # remove green
def rgb_to_hex(rgb):
    rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)

blue = rgb_to_hex(color_palette("deep")[0])
# red = rgb_to_hex(sb.color_palette("deep")[2])
# red = '#880000' # Crimson red
red =  rgb_to_hex(color_palette("deep")[4])
purple = rgb_to_hex(color_palette("deep")[3])

K = linspace(-1.5, 1.5, 1000)
varps = [0.002, 0.078, 0.142, 10.098]
vari = 1





def stdev(vari,varp):
    # outside term only
    # morris and shin original
    print('varp:', varp)
    print(vari)
    varp += 0.00000001
    vari += 0.00000001
    return vari * sqrt(varp + vari**2/varp) / (varp**2 + vari**2)


# def stdev(vari,varp):
#     #including inside term
#     print(varp)
#     print(vari)
#     varp += 0.00000001
#     vari += 0.00000001
#     return  (varp**2 + vari**2) / vari * sqrt(varp + vari**2/varp)  * (vari**2)/(varp**2)


# VA - Investor's Payoff from price amendment
def VA(theta, theta_star, vari=0.1, varp=0.1, y=0, si=0):
    "Morris and Shin benchmark, Investor Payoffs"
    "kappa = theta_star : cutoff theta"
    "theta = investor's expectation of theta"
    cdf_top = varp/vari*(y - theta_star) - theta_star - si
    f = norm.cdf(cdf_top / stdev(vari, varp))
    return theta + f - 0.5




# def plot_VA():
#     varps = [1.77, 2.906, 10.468]
#     varps = [1.77, 5, 96]

#     vari=2
#     for varp, color in zip(varps, kolor):
#         states = ([0], ['-'])
#         for sp, linestyle_ in zip(*states):
#             alpha = stdev(vari,varp)
#             label = r"$S_p: {}$ $var_p: {}$ $\alpha:{}$".format(sp, varp, round(alpha, 2))
#             plot(K, [VA(k, k, vari, varp, sp) for k in K],
#                 label=label, alpha=0.6, linestyle=linestyle_, color=color)

#     legend(loc='lower right')
#     title(r"Investor Payoffs, ($\theta=s^*$)", fontsize='14')
#     ylabel(r"$v^*(s^*)$", fontsize='16')
#     xlabel(r"$s^*$", fontsize='16')



def animations():

    ###### ENDOGENOUS SIGNALS
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib import animation

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-1, 1))
    line, = ax.plot([], [], lw=1.5)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    invest_text = ax.text(0.05, 0.51, '', transform=ax.transAxes)
    not_invest_text = ax.text(0.05, 0.46, '', transform=ax.transAxes)
    equilibrium_text = ax.text(0.76, 0.51, '', transform=ax.transAxes)


    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        invest_text.set_text('Invest')
        not_invest_text.set_text('Not Invest')
        equilibrium_text.set_text("Unique Equilibrium")
        return line, time_text, equilibrium_text


    def animate(i):
        global line
        i += 0.00000000001
        i /= 10

        x = linspace(-1.5, 1.5, 1000)
        switchpoint = 1.20
        tau_p = (i/(1+i/(2*np.pi)))**2


        if i >= switchpoint:
            y = VA(x, x, i, tau_p, 0)
            line.set_data(x, y)
            line.set_color(purple)

            taup = round(tau_p, 2)
            taui = round(i, 2)
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)
            equilibrium_text.set_text('Multiple Equilbria')

        elif i < switchpoint:
            y = VA(x, x, i, tau_p, 0)
            line.set_data(x, y)
            line.set_color(blue)
            taup = round(tau_p, 2)
            taui = round(i, 2)
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)

        if i > 0:
            legend(loc='lower right')
            title(r"Investor Payoffs, ($\theta=s^*$)", fontsize='14')
            ylabel(r"Payoffs: $v(s^*)$", fontsize='16')
            xlabel(r"Signal: $s^*$", fontsize='16')

        time_text.set_text(label)

        return line, time_text


    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=400, interval=2, blit=True, repeat_delay=200)
    anim.save('basic_animation_info_aggregation_success.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()





    ###### ENDOGENOUS SIGNALS - information aggregation failure

    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-1, 1))
    line, = ax.plot([], [], lw=1.5)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    invest_text = ax.text(0.05, 0.51, '', transform=ax.transAxes)
    not_invest_text = ax.text(0.05, 0.46, '', transform=ax.transAxes)
    equilibrium_text = ax.text(0.76, 0.51, '', transform=ax.transAxes)


    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        invest_text.set_text('Invest')
        not_invest_text.set_text('Not Invest')
        return line, time_text, equilibrium_text


    def animate(i):
        global line
        i += 0.00000000001
        i /= 10

        x = linspace(-1.5, 1.5, 1000)
        switchpoint = 1.20
        switchpointH = 8.3
        tau_p = (i/(1+(i**1.5)/(2*np.pi)))**2

        if switchpointH > i >= switchpoint:
            y = VA(x, x, i, tau_p, 0)
            line.set_data(x, y)
            line.set_color(purple)
            equilibrium_text.set_text('Multiple Equilbria')

        else:
            y = VA(x, x, i, tau_p, 0)
            line.set_data(x, y)
            line.set_color(blue)
            equilibrium_text.set_text('Unique Equilibrium')

        taup = round(tau_p, 2)
        taui = round(i, 2)
        label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)

        if i > 0:
            legend(loc='lower right')
            title(r"Investor Payoffs, ($\theta=s^*$)", fontsize='14')
            ylabel(r"Payoffs: $v(s^*)$", fontsize='16')
            xlabel(r"Signal: $s^*$", fontsize='16')

        time_text.set_text(label)

        return line, time_text


    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=400, interval=2, blit=True, repeat_delay=200)
    anim.save('basic_animation_info_aggregation_failure.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()







    ################### EXOGENOUS SIGNALS
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(-2, 2), ylim=(-1, 1))
    line, = ax.plot([], [], lw=1.5)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    invest_text = ax.text(0.05, 0.51, '', transform=ax.transAxes)
    not_invest_text = ax.text(0.05, 0.46, '', transform=ax.transAxes)
    equilibrium_text = ax.text(0.76, 0.51, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        invest_text.set_text('Invest')
        not_invest_text.set_text('Not Invest')
        equilibrium_text.set_text('Unique Equilibrium')
        return line, time_text, equilibrium_text


    def animate(i):
        global line
        x = linspace(-1.5, 1.5, 1000)
        switchpoint = 4.1
        i /= 20

        if switchpoint*2.5 > i >= switchpoint*2:
            global shift
            shift = np.sqrt((i-switchpoint)*2/1000)
            # shift in investment threshold (increase in public signal): x - shift
            y = VA(x, x-shift, 20, i + (i-switchpoint)**1.5, 0)
            line.set_data(x, y)
            line.set_color(purple)
            # line, = ax.plot(x, y, purple)
            taup = i**1
            taui = 20
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)
            equilibrium_text.set_text("Multiple equilibria")

        elif i >= switchpoint*2:
            y = VA(x, x-shift, 20, i + (i-switchpoint)**1.5, 0)
            line.set_data(x, y)
            line.set_color(purple)
            # line, = ax.plot(x, y, purple)
            taup = i**1
            taui = 20
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)
            equilibrium_text.set_text("Multiple equilibria")


        elif i >= switchpoint:
            y = VA(x, x, 20, i + (i-switchpoint)**1.5, 0)
            line.set_data(x, y)
            line.set_color(purple)
            # line, = ax.plot(x, y, purple)
            taup = i**1
            taui = 20
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)
            equilibrium_text.set_text("Multiple equilibria")

        elif i < switchpoint:
            y = VA(x, x, 20, i, 0)
            line.set_data(x, y)
            line.set_color(blue)
            # line, = ax.plot(x, y, blue)
            taup = i**1
            taui = 20
            label = r"Precision of public and private signals: $\tau_p={}, \tau_i={}$".format(taup, taui)

        if i == switchpoint*2:
            ax.plot(x, y, purple, alpha=0.5, linestyle='--')

        if i == switchpoint:
            ax.plot(x, y, blue, alpha=0.5, linestyle='--')

        if i > 0:
            legend(loc='lower right')
            title(r"Investor Payoffs, ($\theta=s^*$)", fontsize='14')
            ylabel(r"Payoffs: $v(s^*)$", fontsize='16')
            xlabel(r"Signal: $s^*$", fontsize='16')

        time_text.set_text(label)

        return line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=400, interval=2, blit=True, repeat_delay=200)

    anim.save('basic_animation_exogenous_signal.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()
















# def prob_amend(theta, K, vari, varp, sp=0.5, c=0.5):
#     # cdf_top = (3*vari/varp + 2)*sp + K
#     cdf_bottom = stdev(vari,varp)
#     cdf_top = (vari/varp) * (K - sp) - K
#     return norm.cdf(cdf_top/cdf_bottom)


# def V2B(theta, K, vari, varp, sp=0.5, c=0.5):
#     "Speculator Payoffs"
#     # cdf_top = (3*vari/varp + 2)*sp + K
#     cdf_bottom = stdev(vari,varp)

#     cdf_top = (vari/varp) * (K - sp) - K
#     pr_amend = norm.cdf(cdf_top/cdf_bottom)
#     # return pr_amend * (1-c) + (1-pr_amend) * (-c) + theta
#     ## Same as:
#     return theta + pr_amend - c














# # ##Speculator Payoffs vs Pr(Amendment)
# # ##varps = [1.77, 2.906, 10.468]
# varps = [0.01, 0.1, 0.2]
# K = linspace(-4, 4, 1000)
# for varp, color in zip(varps, kolor):
#     states = ([0.0], [0.5])
#     for sp, c in zip(*states):
#         alpha = stdev(vari,varp)
#         label = r"$c: {}$  $\sigma_p:{}$".format(c, round(varp, 1))
#         pr_a = [prob_amend(k, k, vari, varp, sp) for k in K]
#         plot(pr_a, [V2B(k, k, vari, varp, sp) for k in K],
#             label=label, alpha=0.6, color=color)
# legend(loc="lower right")
# ylim(-2, 2)


# c = 0.7
# alpha = stdev(vari,varp)
# label = r"$c: {}$  $\sigma_p:{}$".format(c, round(varp, 1))
# pr_a = [prob_amend(k, k, vari, varp, c=c) for k in K]
# plot(pr_a, [V2B(k, k, vari, varp, c=c) for k in K],
#     label=label, alpha=0.6, linestyle='--', color=color)

# legend(loc="lower right")
# title(r"Speculator Payoffs given beliefs about $Pr(T > T_{crit}) (\theta=s^*)$")
# ylabel(r"$v^*(s^*)$", fontsize='16')
# xlabel(r"$Pr(T > T_{crit})$", fontsize='16')


# # Speculator Payoffs vs K
# K = linspace(-1, 1, 1000)
# for varp, color in zip(varps, kolor):
#     states = ([0], [0.6])
#     for sp, alpha in zip(*states):
#         g = gamma(vari,varp)
#         label = r"$S_p: {}$  $\alpha:{}$".format(sp, round(g, 2))
#         plot(K, [V2B(k, k, vari, varp, sp) for k in K],
#             label=label, alpha=alpha, color=color)
# legend(loc="lower right")
# title(r"Investor Payoffs given beliefs about $Pr(T > T_{crit}) (\theta=\kappa)$")
# ylabel(r"$v^*(s^*)$", fontsize='16')
# xlabel(r"$Pr(T > T_{crit})$", fontsize='16')












# # VA - UNDERWRITER'S Payoff from price amendment

# varps = [0.002, 0.078, 0.142, 10.098]
# for varp, color in zip(varps, kolor):
#     states = ([-0.05, 0, 0.05], [0.2, 0.5, 0.8], [0.8, 0.6, 0.4])
#     for sp, c, alpha in zip(*states):
#         g = round(varp/vari, 4)
#         label = r"$S_p: {}$  $\sigma_p/\sigma_i:{}$".format(sp, g)
#         plot(K, [VA(k, k, vari, varp, sp) for k in K],
#             label=label, alpha=alpha, color=color)
# legend()
# title(r"Underwriter Payoffs, ($\theta=\kappa$)")
# ylabel(r"$v^*(\kappa,\kappa)$")
# xlabel(r"$\kappa$")


# # Varying costs over time
# # Suppose the price revise upwards, but costs increase over time.
# # cost increase over time reflects lower probability of allocations-- partially filled orders or pro-rata allocations due to oversubscription.
# varps = [0.078, 10.098]
# for varp, color in zip(varps, kolor):
#     states = ([-0.05, 0, 0.05], [0.4, 0.5, 0.6], [0.8, 0.6, 0.4])

#     for sp, c, alpha in zip(*states):
#         label = r"$S_p: {}$".format(sp) + '\n' + r"$\sigma_p:{}$".format(varp)
#         plot(K, [VA(k, k, vari, varp, sp, c) for k in K],
#             label=label, alpha=alpha, color=color)
# legend()
# title("Underwriter Payoffs")
# ylabel(r"$v^*(\kappa,\kappa)$")
# xlabel(r"$\kappa$")




# # # V2A - Investor's assessment of Pr(T>T_crit)
# # K = linspace(-2,2,1000)

# # [plot(K, [V2A(0, k, vari, varp, 0) for k in K], label="vari:{}\nvarp:{}".format(vari,varp)) for varp in [0.01, 0.1, 1, 50]]

# # legend()
# # title("Investor's Belief of Probability of Price Amendment")
# # ylabel(r"$Pr(T > T_{crit})$")
# # xlabel(r"$\kappa$")

# # Speculator's assessment of the probability of price amendment increases for any cutoff K > 0, when the precision of public signal is low. Information acquisition (attention) by speculators increase the probability of a price amendment.

# # Speculator knows greater attention and thus greater precision of private signals mean the UW is more likely to amend the price


# # V2B - Payoffs given beliefs about Pr(T > T_Crit)
# K = linspace(-2,2,1000)

# for varp, color in [(0.02, kolor[3]), (0.1,kolor[2])]:
#     states = ([-0.05, 0, 0.05], [0.8, 0.6, 0.4])
#     for sp, alpha in zip(*states):
#         label = r"$S_p: {}$".format(sp) + '\n' + r"$\sigma_p:{}$".format(varp)
#         plot(K, [V2B(k, k, vari, varp, sp) for k in K],
#             label=label, alpha=alpha, color=color)
# legend()
# title(r"Investor Payoffs given beliefs about $Pr(T > T_{crit}) (\theta=\kappa)$")
# ylabel(r"$v^*(\kappa,\kappa)$")
# xlabel(r"$\kappa$")



# Positive Sp (Sp higher than privately expected) gives positive payoffs for investors with K cutoff > 0 (Sp > K)
# Lower Sp increases the required K for a positive payoff
# Higher probability of price amendments increases the probability that a speculator invests

# The magnitude of this effect depends on the precision of the public signal. A more precise public signal amplifies this effect.

# This explains why UWs cannot consistently cheat and deliberately revise offer prices upwards (cheap talk) as this decreases the precision on the public signal thereby reducing the effectiveness of the price amendment as a coordination mechanism.





# p = linspace(0.01,0.5,500); s=p

# # Varying Thresholds wrt K
# varps2 = [0.01, 0.1, 0.5, 1]
# [plot(K, [A(T(k, vari, varp)) for k in K], label="varp: {:.3f}".format(varp)) for varp in varps2]
# legend()
# title("Proportion of investing speculators")
# ylabel("A(T(.))")
# xlabel("K")


# # Varying theta
# varps = [0.04, 0.6, 0.97, 5] # results in gammas of: 0.1, 5, 10, 113
# [plot(K, list(map(lambda k: v(k, k, 0.1, varp), K)), label="gamma: {:.1f}".format(gamma(0.1,varp)))
#     for varp in [0.04, 0.6, 0.97, 5]]

# legend()
# title("Utilities")
# ylabel("v*(K,K)")
# xlabel("K")

