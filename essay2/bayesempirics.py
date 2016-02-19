

import os, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy
import scipy.stats as stats
import seaborn.apionly as sb
import statsmodels.api as sm
import theano.tensor as tt

from pymc3.backends import SQLite
from pymc3.backends.sqlite import load

from sklearn import preprocessing
from numpy import exp, e, log
from math import factorial

colors = sb.color_palette("muted")


BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())

df = pd.read_csv(BASEDIR + "/df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
df.set_index("cik", inplace=True)
df.drop(['1087294', '1344376'], inplace=1) # 1st update took longer than a year
df = df[df.amends != "None"]
ciks = list(df.index)

df['days_to_first_price_update'] = [-999 if np.isnan(x) else x for x in  df.days_to_first_price_update]
CASI = 'IoT_15day_CASI_weighted_finance'
df['#Syndicate Members'] = df.underwriter_syndicate_size
df['#Lead Underwriters'] = df.underwriter_num_leads
df['Underwriter Rank'] = df.underwriter_rank_avg
df['#S1A Amendments'] = df.num_SEC_amendments
df['Share Overhang'] = df.share_overhang
df['log(1+Sales)'] = [np.log(1+s) for s in df['sales']]
df['log(Proceeds)'] = [np.log(1+p) for p in df['proceeds']]
df['CASI'] = df[CASI]
df['IPO Market Returns'] = df.M3_initial_returns
df['Industry Returns'] = df.M3_indust_rets
df['Amends Down'] = [1 if x =='Down' else 0 for x in df.amends]
df['BAA Spread']= df.BAA_yield_changes
df['FF49 Industry'] = df.FF49_industry


def rgb_to_hex(rgb):
    rgb = map(lambda x: int(max(0, min(x, 255)) * 255), rgb)
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)

blue = rgb_to_hex(sb.color_palette("deep")[0])
red = rgb_to_hex(sb.color_palette("deep")[2])
purple = rgb_to_hex(sb.color_palette("deep")[3])





def mixed_effects():


    le = preprocessing.LabelEncoder()
    # Convert categorical variables to integer
    # participants_idx = le.fit_transform(messages['prev_sender'])

    classes = 'FF49_industry'
    # classes = 'underwriter_tier'
    # classes = 'amends'
    classes = 'amends'
    print("Grouping by: {}".format(classes))

    class_idx = le.fit_transform(df[classes])
    n_classes = len(le.classes_)


    NSamples = 100000
    burn = NSamples/10
    thin = 2

    covariates = [
            'Intercept',
            '#Syndicate Members',
            '#Lead Underwriters',
            'Underwriter Rank',
            'FF49 Industry',
            # 'Amends Down',
            '#S1A Amendments',
            'Share Overhang',
            'log(1+Sales)',
            'log(Proceeds)',
            'CASI',
            # 'media_1st_pricing',
            # 'VC',
            'IPO Market Returns',
            'Industry Returns',
            'BAA Spread',
            ]

    y = df['days_to_first_price_update'].values
    # y = np.ma.masked_values(list(df.days_to_first_price_update), value=-999)



    with pm.Model() as model:

        # Parameters:
        intercept = pm.Gamma('Intercept', alpha=.1, beta=.1, shape=n_classes)

        beta_underwriter_syndicate_size = pm.Normal('#Syndicate Members', mu=0, sd=100)
        beta_underwriter_num_leads = pm.Normal('#Lead Underwriters', mu=0, sd=100, shape=n_classes)
        beta_underwriter_rank_avg = pm.Normal('Underwriter Rank', mu=0, sd=100)
        beta_num_SEC_amendments = pm.Normal('#S1A Amendments', mu=0, sd=100)
        beta_FF49_industry = pm.Normal('FF49 Industry', mu=0, sd=100)
        # beta_amends_down = pm.Normal('Amends Down', mu=0, sd=100)
        beta_share_overhang = pm.Normal('Share Overhang', mu=0, sd=100)
        beta_log_sales = pm.Normal('log(1+Sales)', mu=0, sd=100)
        beta_log_proceeds = pm.Normal('log(Proceeds)', mu=0, sd=100)
        beta_CASI = pm.Normal('CASI', mu=0, sd=100)
        # beta_media_1st_pricing = pm.Normal('media_1st_pricing', mu=0, sd=100)
        # beta_VC = pm.Normal('VC', mu=0, sd=100)
        beta_BAA_spread = pm.Normal('BAA Spread', mu=0, sd=100)
        beta_M3_initial_returns = pm.Normal('IPO Market Returns', mu=0, sd=100)
        beta_M3_indust_rets = pm.Normal('Industry Returns', mu=0, sd=100)

        # Hyperparamaters
        ## alpha: hyperparameters for neg-binom distribution
        alpha = pm.Gamma('alpha', alpha=.1, beta=.1)



        # #Poisson Model Formula
        mu = tt.exp(
                intercept[class_idx]
                + beta_underwriter_syndicate_size * df.underwriter_syndicate_size
                + beta_underwriter_num_leads * df.underwriter_num_leads
                + beta_underwriter_rank_avg * df.underwriter_rank_avg
                + beta_FF49_industry * df.FF49_industry
                # + beta_amends_down * df['Amends Down']
                + beta_num_SEC_amendments * df.num_SEC_amendments
                + beta_share_overhang * df['Share Overhang']
                + beta_log_sales * df['log(1+Sales)']
                + beta_CASI * df['CASI']
                + beta_log_proceeds * df['log(Proceeds)']
                # + beta_media_1st_pricing * df.media_1st_pricing
                # + beta_VC * df.VC
                + beta_BAA_spread * df['BAA Spread']
                + beta_M3_initial_returns * df.M3_initial_returns
                + beta_M3_indust_rets * df.M3_indust_rets
                    )

        # Dependent Variable
        y_est = pm.Bound(pm.NegativeBinomial('y_est', mu=mu, alpha=alpha, observed=y), lower=1)
        y_pred = pm.Bound(pm.NegativeBinomial('y_pred', mu=mu, alpha=alpha, shape=y.shape), lower=1)
        # y_est = pm.Poisson('y_est', mu=mu, observed=data)
        # y_pred = pm.Poisson('y_pred', mu=mu, shape=data.shape)

        start = pm.find_MAP()
        # step = pm.Metropolis(start=start)
        step = pm.NUTS()
        # backend = pm.backends.Text('test')
        # trace = pm.sample(NSamples, step, start=start, chain=1, njobs=2, progressbar=True, trace=backend)
        trace = pm.sample(NSamples, step, start=start, njobs=1, progressbar=True)

        trace2 = trace
        trace = trace[-burn::thin]

        waic = pm.waic(trace)
        # dic = pm.dic(trace)



    # with pm.Model() as model:
    #     trace_loaded = pm.backends.sqlite.load('FF49_industry.sqlite')
        y_pred.dump('FF49_industry_missing/y_pred')


    ## POSTERIOR PREDICTIVE CHECKS
    y_pred = trace.get_values('y_pred')
    # see if vector of 415 samples corrsponds to 415 obs
    # ax = plt.subplot()
    # ax.axvline(y.mean())
    # sb.distplot(y, ax=ax)
    # [sb.distplot(y, kde=False, ax=ax) for y in y_pred[10:15]]

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(311)
    x_lim=90
    _ = plt.hist(y, range=[0, x_lim], bins=x_lim, histtype='stepfilled', alpha=0.6, color='black')
    _ = plt.title('Distribution of Observed Data')
    ax1.axvline(np.mean(y), linestyle='-', color='black', label='Mean')
    ax1.axvline(np.percentile(y, 50), linestyle='--', color='black', label='Median')
    ax1.axvline(np.percentile(y, 5), linestyle='--', color='black', label=r'$5^{th}$, $50^{th}$ and $95^{th}$ Percentile')
    ax1.axvline(np.percentile(y, 95), linestyle='--', color='black')


    ax2 = fig.add_subplot(312)
    ypred2 = y_pred[-20:-15]
    ax2.axvline(np.mean(ypred2), linestyle='-', color='black', label='Mean')
    ax2.axvline(np.percentile(ypred2, 50), linestyle='--', color='black', label='Median')
    ax2.axvline(np.percentile(ypred2, 5), linestyle='--', color='black', label=r'$5^{th}$, $50^{th}$ and $95^{th}$ Percentile')
    ax2.axvline(np.percentile(ypred2, 95), linestyle='--', color='black')
    _ = [plt.hist(y, range=[0, x_lim], bins=x_lim, histtype='stepfilled', alpha=0.2, color=blue) for y in ypred2]
    _ = plt.xlim(1, x_lim)
    _ = plt.ylabel('Days to Price Amendment')
    _ = plt.title('Posterior Predictive Distribution (#1)')


    ax3 = fig.add_subplot(313)
    ypred3 = y_pred[-105:-100]
    ax3.axvline(np.mean(ypred3), linestyle='-', color='black', label='Mean')
    ax3.axvline(np.percentile(ypred3, 50), linestyle='--', color='black', label='Median')
    ax3.axvline(np.percentile(ypred3, 5), linestyle='--', color='black', label=r'$5^{th}$, $50^{th}$ and $95^{th}$ Percentile')
    ax3.axvline(np.percentile(ypred3, 95), linestyle='--', color='black')
    _ = [plt.hist(y, range=[0, x_lim], bins=x_lim, histtype='stepfilled', alpha=0.2, color=blue) for y in ypred3]
    _ = plt.xlim(1, x_lim)
    _ = plt.xlabel('Days to Price Amendment')
    _ = plt.title('Posterior Predictive Distribution (#2)')
    plt.legend(fontsize='small')



    # PARAMETER POSTERIORS
    anno_kwargs = {'xycoords': 'data', 'textcoords': 'offset points',
                    'rotation': 90, 'va': 'bottom', 'fontsize': 'large'}
    anno_kwargs2 = {'xycoords': 'data', 'textcoords': 'offset points',
                    'rotation': 0, 'va': 'bottom', 'fontsize': 'large'}


    n0, n1, n2, n3 = 1, 5, 9, 14 # numbering for posterior plots
    # intercepts
    # mn = pm.df_summary(trace)['mean']['Intercept_log__0']
    # ax[0,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(0,15), color=blue, **anno_kwargs2)
    # mn = pm.df_summary(trace)['mean']['Intercept_log__1']
    # ax[0,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(0,15), color=purple, **anno_kwargs2)
    # coeffs
    # mn = pm.df_summary(trace)['mean'][2]
    # ax[1,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5, 10), color=red, **anno_kwargs)
    # mn = pm.df_summary(trace)['mean'][3]
    # ax[2,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5,10), color=red, **anno_kwargs)
    # mn = pm.df_summary(trace)['mean'][4]
    # ax[3,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5,10), color=red, **anno_kwargs)
    # plt.savefig('figure1_mixed.png')

    ax = pm.traceplot(trace, vars=['Intercept']+trace.varnames[n0:n1],
            lines={k: v['mean'] for k, v in pm.df_summary(trace).iterrows()}
            )

    for i, mn in enumerate(pm.df_summary(trace)['mean'][n0:n1]): # +1 because up and down intercept
        ax[i,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5,10), color=red, **anno_kwargs)
    plt.savefig('figure1_mixed.png')


    ax2 = pm.traceplot(trace, trace.varnames[n1:n2],
            lines={k: v['mean'] for k, v in pm.df_summary(trace).iterrows()}
            )
    for i, mn in enumerate(pm.df_summary(trace)['mean'][n1:n2]): # +1 because up and down intercept
        ax2[i,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5,10), color=red, **anno_kwargs)
    plt.savefig('figure2_mixed.png')



    ax3 = pm.traceplot(trace, trace.varnames[n2:n3],
            lines={k: v['mean'] for k, v in pm.df_summary(trace).iterrows()}
            )
    for i, mn in enumerate(pm.df_summary(trace)['mean'][n2:n3]): # +1 because up and down intercept
        ax3[i,0].annotate('{:.3f}'.format(mn), xy=(mn,0), xytext=(5,10), color=red, **anno_kwargs)
    plt.savefig('figure3_mixed.png')


    # _ = plt.figure(figsize=(5, 6))
    _ = pm.forestplot(trace, vars=['Intercept'], ylabels=le.classes_)
    plt.savefig('forestplot_intercepts.png')
    _ = pm.forestplot(trace, vars=covariates[1:], ylabels=covariates[1:])
    plt.savefig('forestplot_mixed.png')

    # pm.traceplot(trace, vars=['alpha', 'y_pred'])



    def participant_y_pred(entity_name, burn=1000, hierarchical_trace=trace):
        """Return posterior predictive for person"""
        ix = np.where(le.classes_ == entity_name)[0][0]
        return hierarchical_trace['y_pred'][burn:, ix]


    days = 7

    fig = plt.figure(figsize=(16,10))
    fig.add_subplot(221)
    entity_plotA('Hlth', days=days)
    fig.add_subplot(222)
    entity_plotB('Hlth')

    fig.add_subplot(223)
    entity_plotA('Fin', days=days)
    fig.add_subplot(224)
    entity_plotB('Fin')



    fig = plt.figure(figsize=(16,10))
    fig.add_subplot(221)
    entity_plotA('Rtail', days=days)
    fig.add_subplot(222)
    entity_plotB('Rtail')

    fig.add_subplot(223)
    entity_plotA('Softw', days=days)
    fig.add_subplot(224)
    entity_plotB('Softw')



    # messages = pd.read_csv('hangout_chat_data.csv')
    # le = preprocessing.LabelEncoder()
    # participants_idx = le.fit_transform(messages['prev_sender'])
    # participants = le.classes_
    # n_participants = len(participants)

    # with pm.Model() as model:
    #     hyper_alpha_sd = pm.Uniform('hyper_alpha_sd', lower=0, upper=50)
    #     hyper_alpha_mu = pm.Uniform('hyper_alpha_mu', lower=0, upper=10)

    #     hyper_mu_sd = pm.Uniform('hyper_mu_sd', lower=0, upper=50)
    #     hyper_mu_mu = pm.Uniform('hyper_mu_mu', lower=0, upper=60)

    #     alpha = pm.Gamma('alpha', mu=hyper_alpha_mu, sd=hyper_alpha_sd, shape=n_participants)
    #     mu = pm.Gamma('mu', mu=hyper_mu_mu, sd=hyper_mu_sd, shape=n_participants)

    #     y_est = pm.NegativeBinomial('y_est',
    #                                 mu=mu[participants_idx],
    #                                 alpha=alpha[participants_idx],
    #                                 observed=messages['time_delay_seconds'].values)

    #     y_pred = pm.NegativeBinomial('y_pred',
    #                                  mu=mu[participants_idx],
    #                                  alpha=alpha[participants_idx],
    #                                  shape=messages['prev_sender'].shape)

    #     backend = SQLite('test.sqlite')
    #     start = pm.find_MAP()
    #     step = pm.Metropolis()
    #     hierarchical_trace = pm.sample(20000, step, progressbar=True, trace=backend)


    # with pm.Model() as model:
    #     trace_loaded = pm.backends.sqlite.load('test.sqlite')



def entity_plotA(entity_name, days=7, x_lim=60):
    ix_check = participant_y_pred(entity_name) > days
    plt.hist(participant_y_pred(entity_name)[~ix_check],
            range=[0, x_lim], bins=x_lim, histtype='stepfilled',
            label='< {} days'.format(days))
    plt.hist(participant_y_pred(entity_name)[ix_check],
            range=[0, x_lim], bins=x_lim, histtype='stepfilled',
            label='> {} days'.format(days))

    plt.title('Posterior predictive \ndistribution for {}'.format(entity_name))
    plt.xlabel('Price Amendment Time (days)')
    plt.ylabel('Sample Frequency')
    plt.legend()

def entity_plotB(entity_name, x_lim=60):
    ccolor = colors[1]
    x = np.linspace(1, 60, num=60)
    num_samples = float(len(participant_y_pred(entity_name)))
    prob_lt_x = [100*sum(participant_y_pred(entity_name) < i)/num_samples for i in x]
    plt.plot(x, prob_lt_x, color=ccolor)
    plt.fill_between(x, prob_lt_x, color=ccolor, alpha=0.3)
    plt.scatter(10, float(100*sum(participant_y_pred(entity_name) < 10))/num_samples, s=180, c=ccolor)
    plt.title('Probability of amending prices {} \nwithin days (d)'.format(entity_name))
    plt.xlabel('Days')
    plt.ylabel('Cumulative probability')
    plt.ylim(ymin=0, ymax=50)
    plt.xlim(xmin=0, xmax=60)





def bino(N, k):
    return factorial(N) / (factorial(k) * factorial(N-k))

def binom(N, k, p):
    binoms = []
    for kk in range(k, N):
        binoms.append(bino(N,kk) * p**kk * (1-p)**(N-kk))
    return sum(binoms)

def rbinom(n, r, p):
    # p is probability of r
    return bino(n-1, r-1) * p**r * (1-p)**(n-r)

def rbinom2(x, r, p):
    # x is number of success, r: num failures n=r+x
    return bino(x+r-1, r-1) * p**r * (1-p)**(x)



