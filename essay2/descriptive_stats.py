#!/usr/bin/env python


import csv, os, sys, itertools, json
import pandas as pd
import numpy as np
import arrow

from scipy.stats.mstats import kruskalwallis
from scipy.stats import mannwhitneyu

import seaborn as sb
import matplotlib.pyplot as plt
plot = plt.plot
cp = sb.color_palette
colors = sb.color_palette("muted")
colors2 = sb.color_palette("husl")

def cp_four(x):
    c = sb.color_palette(x, n_colors=16)
    return [c[1], c[7], c[9], c[12]]


def rgb_to_hex(rgb):
    rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)


def plot_var_dist(plotargs, kkey='IPO_duration', kw_xy=(20,20)):

    f, ax = plt.subplots(1,1, figsize=(12, 4), sharex=True)

    for arg in plotargs:
        df, label, color, xshift, yshift = arg
        color = sb.color_palette("muted")[color]
        label += " Obs={}".format(len(df))

        # Summary stats:
        mean = df[kkey].mean()
        mode = df[kkey].mode()
        med  = df[kkey].median()
        std  = df[kkey].std()
        skew = df[kkey].skew()
        stat = u"\nμ={:0.2f}  med={:0.2f}\nσ={:0.2f}  skew={:0.2f}".format(
                mean, med, std, skew)

        yvals, xvals, patchs = plt.hist(df[kkey].tolist(), bins=36, label=label,
                                color=color, alpha=0.6, histtype='stepfilled')

        coords = list(zip(yvals,xvals))
        coords.sort()
        y,x = coords[-3]

        ax.annotate(stat,
                    xy=(x, y),
                    xytext=(x*xshift, y*yshift),
                    arrowprops=dict(facecolor=color,
                                    width=1.6,
                                    headwidth=1.6))

    H, prob = kruskalwallis(*[x[0][kkey] for x in plotargs])
    # U, prob = mannwhitneyu(*[x[0][kkey] for x in plotargs])
    ax.annotate("Kruskal-Wallis: (H={H:.2f}, prob={p:.3f})".format(H=H, p=prob),
                xy=(kw_xy[0], kw_xy[1]))
    plt.ylabel("Frequency")
    plt.legend()


def plot_kaplan_function(duration_key):

    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt


    duration_keys = ["days_from_priced_to_listing",
                    "days_to_final_price_revision",
                    # "days_to_first_price_update",
                    "days_from_s1_to_listing",
                    "days_to_first_price_change"]
    duration_key = duration_keys[-2]

    kmf = KaplanMeierFitter()
    f, ax = plt.subplots(1,1,figsize=(12, 4), sharex=True)
    T = 1 # annotation line thickness
    xoffset = 0.4 # annotation offset (x-axis)
    yoffset = 0.04


    # Above filing price range
    kmf.fit(above[duration_key], label='Upward Price Amendment: N={}'.format(len(above)), alpha=0.9)
    kmf.plot(ax=ax, c=colors[5], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[5], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+xoffset, 0.25+yoffset),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+xoffset, 0.50+yoffset),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+xoffset, 0.75+yoffset),
                arrowprops=aprops)


    # # Within filing price range
    # kmf.fit(within[duration_key],
    #         label = 'Within filing price range: N={}'.format(len(within)),
    #         )
    # kmf.plot(ax=ax, c=colors[0], alpha=0.7)

    # quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    # aprops = dict(facecolor=colors[0], width=T, headwidth=T)

    # plt.annotate("75%: {} days".format(quartiles[0]),
    #             (quartiles[0], 0.25),
    #             xytext=(quartiles[0]+xoffset, 0.25+yoffset),
    #             arrowprops=aprops)

    # plt.annotate("50%: {} days".format(quartiles[1]),
    #             (quartiles[1], 0.50),
    #             xytext=(quartiles[1]+xoffset, 0.50+yoffset),
    #             arrowprops=aprops)

    # plt.annotate("25%: {} days".format(quartiles[2]),
    #             (quartiles[2], 0.75),
    #             xytext=(quartiles[2]+xoffset, 0.75+yoffset),
    #             arrowprops=aprops)


    # Under filing price range
    kmf.fit(under[duration_key], label='Downward Price Amendment: N={}'.format(len(under)),)
    kmf.plot(ax=ax, c=colors[2], alpha=0.7)

    quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
    aprops = dict(facecolor=colors[2], width=T, headwidth=T)

    plt.annotate("75%: {} days".format(quartiles[0]),
                (quartiles[0], 0.25),
                xytext=(quartiles[0]+xoffset, 0.25+yoffset+0.05),
                arrowprops=aprops)

    plt.annotate("50%: {} days".format(quartiles[1]),
                (quartiles[1], 0.50),
                xytext=(quartiles[1]+xoffset, 0.50+yoffset+0.05),
                arrowprops=aprops)

    plt.annotate("25%: {} days".format(quartiles[2]),
                (quartiles[2], 0.75),
                xytext=(quartiles[2]+xoffset, 0.75+yoffset+0.05),
                arrowprops=aprops)


    # log rank tests + general graph labels
    # summary, p_value, results = logrank_test(
    #                                 above[duration_key],
    #                                 within[duration_key],
    #                                 under[duration_key],
    #                                 alpha=0.95)
    # ax.annotate("Log-rank test: (prob={p:.3f})".format(p=p_value),
    #             xy=(1210, 0.08))

    plt.ylim(0,1)
    plt.xlim(0, max(np.percentile(above[duration_key], 90), np.percentile(under[duration_key],90)))
    plt.title("Kaplan-Meier Survival Functions")
    plt.xlabel("Delay (days) in {}".format(duration_key))
    plt.ylabel(r"$S(t)=Pr(T>t)$")








def uw_tier_histplots():
    sample['Underwriter Tier'] = sample['lead_underwriter_tier']
    sample['IPO Duration'] = sample['IPO_duration']
    ranks = ["-1", "0+", "7+", "9"]

    def uw_tier_duration(x):
        return sample[sample.lead_underwriter_tier==x]['IPO_duration']
    kwstat = kruskalwallis(*[uw_tier_duration(x) for x in ranks])

    # g = sb.FacetGrid(sample,
    #                 row="Underwriter Tier",
    #                 hue="Underwriter Tier",
    #                 palette=cp_four("cool_r"),
    #                 size=2, aspect=4,
    #                 hue_order=ranks, row_order=ranks,
    #                 legend=ranks, xlim=(0,1095))
    # g.map(sb.distplot, "IPO Duration")
    # plt.savefig("IPO_tiers_KP_survival.pdf", format='pdf', dpi=200)


    from lifelines.estimation import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import matplotlib.pyplot as plt

    ranks = ["-1", "0+", "7+", "9"]
    ranklabels = ['No Underwriter', 'Low Rank', 'Mid Rank', 'Rank 9 (elite)']
    kmf = KaplanMeierFitter()

    # Success
    f, ax = plt.subplots(1,1,figsize=(12, 4), sharex=True)
    T = 1 # annotation line thickness

    for rank, rlabel, color in zip(ranks, ranklabels, cp_four("cool_r")):
        uw = sample[sample.lead_underwriter_tier==rank]

        kmf.fit(uw['IPO_duration'],
                label='{} N={}'.format(rlabel, len(uw)),
                alpha=0.9)
        kmf.plot(ax=ax, c=color, alpha=0.7)

        quartiles = [int(np.percentile(kmf.durations, x)) for x in [25, 50, 75]][::-1]
        aprops = dict(facecolor=color, width=T, headwidth=T)

        if rank=="-1":
            plt.annotate("75%: {} days".format(quartiles[0]),
                        (quartiles[0], 0.25),
                        xytext=(quartiles[0]+145, 0.25+.04),
                        arrowprops=aprops)

            plt.annotate("50%: {} days".format(quartiles[1]),
                        (quartiles[1], 0.50),
                        xytext=(quartiles[1]+145, 0.50+.04),
                        arrowprops=aprops)

            plt.annotate("25%: {} days".format(quartiles[2]),
                        (quartiles[2], 0.75),
                        xytext=(quartiles[2]+145, 0.75+0.04),
                        arrowprops=aprops)
        elif rank=="9":
            plt.annotate("75%: {} days".format(quartiles[0]),
                        (quartiles[0], 0.25),
                        xytext=(quartiles[0]+415, 0.25+.1),
                        arrowprops=aprops)

            plt.annotate("50%: {} days".format(quartiles[1]),
                        (quartiles[1], 0.50),
                        xytext=(quartiles[1]+290, 0.50+.1),
                        arrowprops=aprops)

            plt.annotate("25%: {} days".format(quartiles[2]),
                        (quartiles[2], 0.75),
                        xytext=(quartiles[2]+165, 0.75+0.1),
                        arrowprops=aprops)

    plt.annotate("Kruskall Wallis\nH: {:.3f}\nprob: {:.3f}".format(*kwstat),
                (960, 0.1))
    plt.ylim(0,1)
    plt.xlim(0,1095)
    plt.title("Kaplan-Meier survival times by bank tier")
    plt.xlabel("IPO Duration (days)")
    plt.ylabel(r"$S(t)=Pr(T>t)$")
    plt.savefig("IPO_tiers_KP_survival.pdf", format='pdf', dpi=200)






def uw_tier(uw_rank):
    # top_tier = {'Citigroup', 'Credit Suisse', 'Goldman Sachs', 'Merrill Lynch'}
    if uw_rank == 9:
        return "9"
    elif uw_rank > 7:
        return "7+"
    elif uw_rank >= 0:
        return "0+"
    elif uw_rank < 0:
        return "-1"



if __name__=='__main__':

    FINALJSON = json.loads(open('/Users/peitalin/Data/IPO/NASDAQ/final_json.txt').read())
    df_file = '/Users/peitalin/Data/IPO/NASDAQ/df.csv'
    df = pd.read_csv(df_file, dtype={'cik':object, 'Year':object, 'SIC':object})
    dfu = df[df.amends != "None"]


    # under, above, within = [x[1] for x in df.groupby(['offer_in_filing_price_range'])]

    under, above = [x[1] for x in dfu.groupby(['amends'])]

    duration_keys = ["days_from_priced_to_listing",
                    "days_to_final_price_revision",
                    "days_to_first_price_update",
                    "days_from_s1_to_listing",
                    "days_to_first_price_change"]

    # # Durations plots
    # plotargs = [(above, "above", 5, 1.4, 1.1),
    #             (within, "within", 2, 1.3, 1.2)]
    # plot_var_dist(plotargs, kw_xy=(1100, 10))
    # plt.xlim(xmin=0, xmax=1460)
    # plt.xlabel("IPO Duration (days)")
    # plt.title("Book-build IPO durations")
    # plt.savefig("./succ_vs_withdraw_duration.pdf", dpi=200, format='pdf')


    # # Attention plots
    # plotargs = [(success, "Successful IPOs", 5, 1.05, 1.4),
    #             (withdraw, "Withdrawn IPOs", 2, 0.7, 9.1)]
    # plot_var_dist(plotargs, kkey='ln_CASI_all_finance', kw_xy=(7.6, 20))
    # plt.xlabel("Log Cumulative Abnormal Search Interest (all)")
    # plt.title("Abnormal Attention During Book-build IPOs")
    # plt.savefig("./succ_vs_withdraw_CASI.pdf", dpi=200, format='pdf')


    plot_kaplan_function()
    # plt.savefig("./succ_vs_withdraw_Kaplan_Meier.pdf", dpi=200, format='pdf')
    uw_tier_histplots()


