

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

from sklearn import preprocessing


# plt.style.use('bmh')
# colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00',
#           '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())

df = pd.read_csv(BASEDIR + "/df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
df.set_index("cik", inplace=True)
df['log_sales'] = [np.log(1+s) for s in df['sales']]
df['log_proceeds'] = [np.log(1+p) for p in df['proceeds']]

df = df[df.underwriter_num_leads > 0]
df = df[df.days_to_first_price_change >= 0]
df = df[df.amends != "None"]
CASI = 'IoT_15day_CASI_weighted_finance'
df['CASI'] = df[CASI]
ciks = list(df.index)
# uwset = set(sum([FINALJSON[c]['Experts']['Lead Underwriter'] for c in ciks],[]))
# X = messages[['is_weekend','day_of_week','message_length','num_participants']].values
# _, num_X = X.shape


messages = pd.read_csv('hangout_chat_data.csv')
X = messages[['is_weekend','day_of_week','message_length','num_participants']].values
_, num_X = X.shape


def model1():

    NSamples = 10000
    burn = NSamples/10
    thin = 10

    with pm.Model() as model:
        # parameter priors

        intercept = pm.Normal('intercept', mu=0, sd=100)
        beta_underwriter_syndicate_size = pm.Normal('underwriter_syndicate_size', mu=0, sd=100)
        beta_share_overhang = pm.Normal('share_overhang', mu=0, sd=100)
        beta_log_sales = pm.Normal('log_sales', mu=0, sd=100)
        beta_CASI = pm.Normal(CASI, mu=0, sd=100)
        beta_media_1st_pricing = pm.Normal('media_1st_pricing', mu=0, sd=100)
        beta_original_prange_size = pm.Normal('original_prange_size', mu=0, sd=100)
        beta_underwriter_rank_avg = pm.Normal('underwriter_rank_avg', mu=0, sd=100)
        beta_VC = pm.Normal('VC', mu=0, sd=100)
        beta_M3_initial_returns = pm.Normal('M3_initial_returns', mu=0, sd=100)
        beta_M3_indust_rets = pm.Normal('M3_indust_rets', mu=0, sd=100)

        # #Poisson Model Formula
        mu = tt.exp(intercept
                    + beta_underwriter_syndicate_size * df.underwriter_syndicate_size
                    + beta_share_overhang * df.share_overhang
                    + beta_log_sales * df.log_sales
                    + beta_CASI * df.CASI
                    + beta_media_1st_pricing * df.media_1st_pricing
                    + beta_original_prange_size * df.original_prange_size
                    + beta_underwriter_rank_avg * df.underwriter_rank_avg
                    + beta_VC * df.VC
                    + beta_M3_initial_returns * df.M3_initial_returns
                    + beta_M3_indust_rets * df.M3_indust_rets
                    )


        # Dependent Variable
        y_est = pm.Poisson('y_est', mu=mu, observed=df['days_to_first_price_change'].values)

        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(NSamples, step, start=start, progressbar=True)
        trace = trace[burn::thin]


    _ = pm.traceplot(trace, trace.varnames[:4])
    plt.savefig('figure1.png')
    _ = pm.traceplot(trace, trace.varnames[4:8])
    plt.savefig('figure2.png')
    _ = pm.traceplot(trace, trace.varnames[8:12])
    plt.savefig('figure3.png')
    # _ = sb.pairplot(pm.trace_to_dataframe(trace[NSamples/10:]), plot_kws={'alpha':.5})





def mixed_effects():

    le = preprocessing.LabelEncoder()
    # Convert categorical variables to integer
    # participants_idx = le.fit_transform(messages['prev_sender'])
    # participants = le.classes_
    # n_participants = len(participants)


    class_idx = le.fit_transform(df['FF49_industry'])
    # class_idx = le.fit_transform(df['underwriter_num_leads'])
    # class_idx = le.fit_transform(df['underwriter_tier'])
    # class_idx = le.fit_transform(df['amends'])
    n_classes = len(le.classes_)

    NSamples = 200000
    burn = 1000
    thin = 1

    covariates = [
            'intercept',
            'underwriter_syndicate_size',
            'share_overhang',
            'log_sales',
            CASI,
            'media_1st_pricing',
            'underwriter_rank_avg',
            'VC',
            'M3_initial_returns',
            'M3_indust_rets',
            ]



    with pm.Model() as model:

        intercept = pm.Normal('intercept', mu=0, sd=100, shape=len(le.classes_))
        beta_underwriter_syndicate_size = pm.Normal('underwriter_syndicate_size', mu=0, sd=100)
        beta_share_overhang = pm.Normal('share_overhang', mu=0, sd=100)
        beta_log_sales = pm.Normal('log_sales', mu=0, sd=100)
        beta_CASI = pm.Normal(CASI, mu=0, sd=100, shape=len(le.classes_))
        beta_media_1st_pricing = pm.Normal('media_1st_pricing', mu=0, sd=100)
        beta_underwriter_rank_avg = pm.Normal('underwriter_rank_avg', mu=0, sd=100)
        beta_VC = pm.Normal('VC', mu=0, sd=100)
        beta_M3_initial_returns = pm.Normal('M3_initial_returns', mu=0, sd=100)
        beta_M3_indust_rets = pm.Normal('M3_indust_rets', mu=0, sd=100)

        # #Poisson Model Formula
        mu = tt.exp(intercept[class_idx]
                    + beta_underwriter_syndicate_size * df.underwriter_syndicate_size
                    + beta_share_overhang * df.share_overhang
                    + beta_log_sales * df.log_sales
                    + beta_CASI[class_idx] * df.CASI
                    + beta_media_1st_pricing * df.media_1st_pricing
                    + beta_underwriter_rank_avg * df.underwriter_rank_avg
                    + beta_VC * df.VC
                    + beta_M3_initial_returns * df.M3_initial_returns
                    + beta_M3_indust_rets * df.M3_indust_rets
                    )


        # Dependent Variable
        y_est = pm.Poisson('y_est', mu=mu, observed=df['days_to_first_price_change'].values)

        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(NSamples, step, start=start, progressbar=True)
        trace = trace[burn::thin]


    _ = pm.traceplot(trace, trace.varnames[:4]);  plt.savefig('figure1_mixed.png')
    _ = pm.traceplot(trace, trace.varnames[4:8]); plt.savefig('figure2_mixed.png')
    _ = pm.traceplot(trace, trace.varnames[8:12]); plt.savefig('figure3_mixed.png')
    _ = plt.figure(figsize=(5, 6))
    _ = pm.forestplot(trace[burn:], vars=['intercept'], ylabels=le.classes_)
    # _ = pm.forestplot(trace[burn:], vars=['underwriter_syndicate_size'], ylabels=le.classes_)
    _ = pm.forestplot(trace[burn:], vars=covariates,
            ylabels=['Intercept ({})'.format(s) for s in le.classes_] + covariates[1:])






