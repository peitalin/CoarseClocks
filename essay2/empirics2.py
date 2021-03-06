
import glob, json, os, re, itertools, arrow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf
import statsmodels as sm

from functools          import partial
from pprint             import pprint
from numpy              import log, median, sign
from widgets            import as_cash, next_row, next_col

IPO_DIR = os.path.join(os.path.expanduser("~"), "Data", "IPO")
BASEDIR = os.path.join(os.path.expanduser("~"), "Data", "IPO", "NASDAQ",)
FINALJSON = json.loads(open(BASEDIR + '/final_json.txt').read())


class Tools(object):
    "Misc IPO exploratory data analysis tools for ipy REPL use"

    def __init__(self, D=FINALJSON):
        self.ciks_conames = {cik:D[cik]['Company Overview']['Company Name'] for cik in D}
        self.conames_ciks = {D[cik]['Company Overview']['Company Name']:cik for cik in D}

    def firmname(self, cik):
        return self.ciks_conames[cik]

    def get_cik(self, firm):
        ciks = [k for k in self.conames_ciks if k.lower().startswith(firm.lower())]
        print("Found => {}".format(ciks))
        return self.conames_ciks[ciks[0]]

    @staticmethod
    def aget(sdate):
        sdate = sdate if isinstance(sdate, str) else sdate.isoformat()
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', sdate):
            return arrow.get(sdate, 'M-D-YYYY') if '-' in sdate else arrow.get(sdate, 'M/D/YYYY')
        elif re.search(r'\d{4}[/-]\d{2}[/-]\d{2}', sdate):
            return arrow.get(sdate, 'YYYY-MM-DD') if '-' in sdate else arrow.get(sdate, 'YYYY/MM/DD')

    @staticmethod
    def print_pricerange(s):
        "arg s: either firmname or cik. Returns price-ranges"

        if not s.isdigit():
            cik = [k for k in FINALJSON if firmname(k).lower().startswith(s.lower())][0]
        else:
            cik = s

        filings = FINALJSON[cik]['Filing']

        print("\n{A}> Filing Price Range: {B}: {C} <{A}".format(A='='*25, B=firmname(cik), C=cik)[:91])
        print("Date\t\tFormtype\tFile\t\t\t\tPrice Range")
        for v in filings:
            filing_ashx = v[-2]
            price_range = v[-1][0]
            if price_range == 'NA':
                NA_filepath = os.path.join(BASEDIR, 'filings', cik, filing_ashx)
                filesize = os.path.getsize(NA_filepath)
                print("{}\t{}\t\t{}\t{}\t<= {:,} kbs".format(v[2], v[1], v[3], v[-1], round(filesize/1000)))
            else:
                print("{}\t{}\t\t{}\t{}".format(v[2], v[1], v[3], v[-1]))
        print("="*90+'\n')

T = Tools()







VNAME = {
    'Intercept': 'Intercept',
    '(Intercept)': "Intercept",
    'priceupdate_up': 'Price Update (up)',
    'priceupdate_down': 'Price Update (down)',
    'pct_final_revision_up': 'Final Price Revision (up)',
    'pct_final_revision_down': 'Final Price Revision (down)',
    'pct_first_price_change_up': 'First Price Update (up)',
    'pct_first_price_change_down': 'First Price Update (down)',
    'prange_change_plus': 'Price Range Change',
    'original_prange_size': 'Price Range',
    'number_of_price_updates': 'No. Price Amendments',
    'number_of_price_updates_up': 'No. Price Updates (up)',
    'number_of_price_updates_down': 'No. Price Updates (down)',
    'delay_in_price_update' : 'Delay in 1st price update',
    'log(days_from_s1_to_listing)': 'log(Days from S-1 to Listing)',
    'underwriter_rank_avg': 'Underwriter Rank',
    'underwriter_collective_rank': 'Underwriter Rank (Group)',
    'underwriter_num_leads': '#Lead Underwriters',
    'underwriter_syndicate_size': '#Syndicate Members',
    'VC': 'Venture Capital',
    'confidential_IPO': 'Confidential IPO',
    'share_overhang': 'Share Overhang',
    'log(proceeds)': 'log(Proceeds)',
    'log(market_cap)': 'log(Market Cap)',
    'log(1 + sales)': 'log(1 + Sales)',
    'liab_over_assets': 'Liab/Assets',
    'EPS': 'EPS+',
    'M3_indust_rets': '3-Month Industry Returns',
    'M3_initial_returns': '3-Month Average IPO Returns',
    'CASI': 'CASI',
    'np.square(CASI)': 'CASI^2',
    'I(CASI^2)': 'CASI^2',
    'foreign': 'foreign',
    'media_listing': 'Media Count',
    'media_1st_pricing': 'Media Count'
}


if __name__=='__main__':
    ciks = sorted(FINALJSON.keys())
    # cik = '1439404' # Zynga         # 8.6
    # cik = '1418091' # Twitter       # 7.4
    # cik = '1271024' # LinkedIn      # 9.6
    # cik = '1500435' # GoPro         # 8.1
    # cik = '1318605' # Tesla Motors  # 8
    # cik = '1326801' # Facebook      # 8.65
    # cik = '1564902' # SeaWorld      # 9.54
    # cikfb = '1326801' # Facebook
    # ciks1 = ['1439404', '1418091', '1271024', '1500435', '1318605', '1594109', '1326801', '1564902']



    df = pd.read_csv(BASEDIR + "/df.csv", dtype={'cik':object, 'Year':object, 'SIC':object})
    df.set_index("cik", inplace=1)
    dup = df[df['size_of_first_price_update'].notnull()]
    dnone = df[df['prange_change_first_price_update'].isnull()]



def xls_empirics(lm, column='C', sheet='15dayCASI', model_type='lm', cluster=('FF49_industry',), sigstars=True):

    from xlwings import Workbook, Range, Sheet
    wb = Workbook(os.path.join(os.path.realpath(os.path.curdir), 'empirics2.xlsx'))
    Sheet(sheet).activate()
    column = column.upper()

    def roundn(i, n=2):
        coef = str(round(i, n))
        for i in range(n-len(coef.split('.')[-1])):
            coef += '0'
        return coef

    def is_logit(lm):
        cmp_model_type = sm.discrete.discrete_model.MultinomialResultsWrapper
        return isinstance(lm, cmp_model_type)

    def write_coefs_tvals(varnames, coefs, tvals, pvals, column=column):

        for v, coef, tval, pval in zip(varnames, coefs, tvals, pvals):

            if v.startswith('Year') or v.startswith('FF49_industry'):
                continue

            if ':' in v:
                v1, v2 = v.split(':')
                v1 = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v1)
                v2 = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v2)
                # Renames IOTKEY, np.square(IOTKEY), and log(IOTKEY) only.
                if 'CASI' in v1:
                    v = v1 + ":" + v2
                else:
                    v = v2 + ":" + v1
                Range('B' + VROW[v]).value = VNAME[v1] + " x " + VNAME[v2]
            else:
                v = re.sub(r"IoT_\d\dday_(?=CASI_)CASI_[a-zA-Z_]*", 'CASI', v)
                Range('B' + VROW[v]).value = VNAME[v]

            coef = roundn(coef, n=2)
            if sigstars:
                if pval <= 0.01:
                    coef += '***'
                elif pval <= 0.05:
                    coef += '**'
                # elif pval <= 0.05:
                #     coef += '*'

            Range(column + VROW[v]).value = coef
            Range(column + str(int(VROW[v])+1)).value = tval

    def rpy_square_terms(eq, Rpy=True):
        """Convert squared covariates syntax between R and Python.
        Rpy=True => Python to R: np.square(X) -> I(X^2).
        RPY=False => R to Python"""

        eq = re.sub(r'\(Intercept\)', 'Intercept', eq)
        if Rpy:
            var = re.findall('(?<=np.square\()[\w_]*(?=\))', eq)
            for v in var:
                eq = eq.replace('np.square({})'.format(v), 'I({}^2)'.format(v))
        else:
            var = re.findall('(?<=I\()[\w_]*(?=\^2\))', eq)
            for v in var:
                eq = eq.replace('I({}^2)'.format(v), 'np.square({})'.format(v))
        return eq

    def lme(eq):
        r("M <- lme({eq}, data=dfR)".format(eq=eq))
        r("Summ <- summary(M)")

        varnames = r("names(coef(M))")
        coefs = r("Summ$tTable[,1]")
        pvals = r("Summ$tTable[,5]")
        tvalues = r("Summ$tTable[,4]")
        tvals = ['(%s)' % roundn(t, n=2) for t in tvalues]
        return varnames, coefs, pvals, tvals

    def lmer(eq):
        r("M <- lmer({eq}, data=dfR)".format(eq=eq))
        r("Summ <- summary(M)")
        varnames = r("rownames(coef(Summ))")
        coefs = r("coef(Summ)[,1]")
        pvals = r("coef(Summ)[,5]")
        tvalues = r("coef(Summ)[,4]")
        tvals = ['(%s)' % roundn(t, n=2) for t in tvalues]
        return varnames, coefs, pvals, tvals


    Range("{col}4:{col}60".format(col=column)).value = [[None]]*60

    if isinstance(lm, str):

        eq = lm
        if 'np.square' in eq:
            eq = rpy_square_terms(eq)


        # STANDARD OLS + Cluster Robust Errors
        if model_type=='lm':
            r("M <- lm({eq}, data=dfR)".format(eq=eq))
            r("src('clmclx.R')")
            if len(cluster) == 2:
                r("M_cluster <- mclx(M, 1, dfR$%s, dfR$%s)" % cluster)
            elif len(cluster) == 1 and "HC1" not in cluster:
                r("M_cluster <- clx(M, 1, dfR$%s)" % cluster)
            elif len(cluster) == 1:
                r('M_cluster <- coeftest(M, vcov=vcovHC(M, type="HC1"))')
            else:
                r('M_cluster <- summary(M)$coefficients')
                cluster = "Standard Errors"

            varnames = r("names(M$coefficients)".format(eq=eq))
            varnames = [rpy_square_terms(v, Rpy=False) for v in varnames]
            coefs = r("M_cluster[,1]".format(eq=eq))
            pvals = r("M_cluster[,4]".format(eq=eq))
            tvalues = r("M_cluster[,3]".format(eq=eq))
            tvals = ['(%s)' % roundn(t, n=2) for t in tvalues]

            Range('B' + VROW['Nobs']).value = 'No. Obs'
            Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
            Range(column + VROW['Nobs']).value = r("M$df.residual + M$rank")[0]
            Range(column + str(int(VROW['Rsq'])-1)).value = r("summary(M)$r.squared")[0]
            Range(column + str(int(VROW['Rsq'])+2)).value = [cluster]

            write_coefs_tvals(varnames, coefs, tvals, pvals)
            return None


        # Linear Mixed Model (lme4) + (lmerTest)
        if model_type=="lmer":
            varnames, coefs, pvals, tvals = lmer(eq)
            N_OBS = r("length(Summ$residuals)")[0]
            AIC = r("Summ$AICtab")[0]

            ## GET RANDOM EFFECTS ESTIMATES - LMER
            _varcorr = r("as.data.frame(VarCorr(M))")
            _top_n = int(len(_varcorr[0]) / 2)
            _random_effects = list(zip(*[list(x) for x in _varcorr]))[:_top_n]
            rand = {x[1]:x[-1] for x in _random_effects}
            ranef = r("ranef(M)$FF49_industry")


        # Linear Mixed Model (nlme)
        if model_type=="lme":
            varnames, coefs, pvals, tvals = lme(eq)
            N_OBS = r("M$dims$N")[0]
            AIC = r("summary(M)$AIC")[0]

            ## GET RANDOM EFFECTS ESTIMATES - LME
            _random_effects = dict(zip(r("rownames(VarCorr(M))"), r("VarCorr(M)[, 'StdDev']")))
            rand = {k:v for k,v in _random_effects.items() if v}
            _ = rand.pop("Residual")
            if len(rand) == 2:
                _ = r("randf <- as.data.frame(ranef(M)[2:2])")
                _ = r("names(randf) <- names(ranef(M)[2:2]$FF49_industry)")
            elif len(rand) == 1:
                _ = r("randf <- as.data.frame(ranef(M)[1])")
            else:
                print("unimplemented 2+ random effects")
            ranef = r("randf")


        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['AIC'])-1)).value = 'AIC'
        # VROW['LogLik'] - 1 because VROW is double spaced for coefficients + tvalues.
        Range(column + VROW['Nobs']).value = N_OBS
        Range(column + str(int(VROW['AIC'])-1)).value = AIC

        # Random Effects
        Range('B' + VROW['RandEffects']).value = 'Random Effects StDev'
        startrow, endrow = VROW['RandEffects'], str(int(VROW['RandEffects'])+2)
        rand_coefs = [x for x in Range("B{}:B{}".format(startrow, endrow)).value if x]
        rand_coefs = ["CASI" if x.startswith("IoT") else x for x in rand_coefs]
        rowlookup = dict(zip(rand_coefs, list(range(int(startrow), int(endrow)+1))))

        for k,v in rand.items():
            k = "CASI" if k.startswith("IoT") else k
            if k in rowlookup.keys():
                Range(  'B%s' % rowlookup[k]            ).value = k
                Range(  column + '%s' % rowlookup[k]    ).value = v
            else:
                Range(  'B%s' % (int(startrow) + len(rand_coefs))            ).value = k
                Range(  column + "%s" % (int(startrow) + len(rand_coefs))    ).value = v


        # Individual random effects for each FF49 industry
        randf = pd.DataFrame(list(zip(*ranef)),
                            columns=ranef.colnames,
                            index= ranef.rownames)

        for n, tup in enumerate(randf.itertuples(), 4):
            for alpha, value in zip(['B', column, chr(ord(column)+1)] , tup):
                Range('{A}{N}'.format(A=alpha, N=int(endrow)+n)).value = value


        write_coefs_tvals(varnames, coefs, tvals, pvals)
        return None



    # Python statsmodels
    if not is_logit(lm):
        varnames = tuple(lm.params.keys())
        rlm = lm.get_robustcov_results()
        coefs = rlm.params
        pvals = rlm.pvalues
        tvals = ['(%s)' % roundn(t, n=2) for t in rlm.tvalues]

        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
        Range(column + VROW['Nobs']).value = rlm.nobs
        Range(column + str(int(VROW['Rsq'])-1)).value = rlm.rsquared

        write_coefs_tvals(varnames, coefs, tvals, pvals)

    elif is_logit(lm):
        varnames = tuple(mnl.params.index)
        coefs = mnl.params
        pvals = mnl.pvalues
        tvals = mnl.tvalues

        for k in coefs:
            # k number of choice variables
            col = chr(ord(column) + k)
            print(tvals)
            t_stats = ['(%s)' % roundn(t, n=2) for t in tvals[k]]
            write_coefs_tvals(varnames, coefs[k], t_stats, pvals[k], column=col)

        Range('B' + VROW['Nobs']).value = 'No. Obs'
        Range('B' + str(int(VROW['Rsq'])-1)).value = 'R^2'
        Range(column + VROW['Nobs']).value = mnl.nobs
        Range(column + str(int(VROW['Rsq'])-1)).value = mnl.prsquared











def OLS_initial_returns():

    ##############################################
    # Initial Returns regression
    # Interaction: CASI * Final_price_revision
    days=15
    VORDER = [
        'Intercept',
        'CASI',
        'np.square(CASI)',

        'CASI:priceupdate_up',
        'CASI:priceupdate_down',
        'priceupdate_up',
        'priceupdate_down',

        'CASI:pct_final_revision_up',
        'CASI:pct_final_revision_down',
        'pct_final_revision_up',
        'pct_final_revision_down',

        'delay_in_price_update',
        'log(days_from_s1_to_listing)',
        # 'CASI:number_of_price_updates_up',
        # 'CASI:number_of_price_updates_down',
        'number_of_price_updates_up',
        'number_of_price_updates_down',

        'underwriter_rank_avg',
        'VC',
        # 'confidential_IPO',
        'media_listing',
        'share_overhang',
        'log(proceeds)',
        'log(1 + sales)',
        'EPS',
        'M3_indust_rets',
        'M3_initial_returns',
        'Nobs',
        'Rsq',
    ]
    VARS = VORDER
    VROW = dict(zip(VARS, [str(n) for n in range(4,100,2)]))



    CONTROLS = [
        'Year',
        'FF49_industry',
        'log(days_from_s1_to_listing)',
        'number_of_price_updates_up',
        'number_of_price_updates_down',
        'underwriter_rank_avg',
        'VC',
        'share_overhang',
        'log(proceeds)',
        'log(1 + sales)',
        'media_listing',
        'EPS',
        'M3_indust_rets',
        'M3_initial_returns', # 13
        ]


    from rpy2.robjects import r
    list(map(r, """
        require(sandwich)
        require(lmtest)
        require(plm)
        source("clmclx.R")
        df <- data.table::fread("/Users/peitalin/Data/IPO/NASDAQ/df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
        df <- df[df$close_return < 200]
        ## df <- df[!is.na(df$percent_first_price_update)]
        dfR <- df[df$amends != "None"]

        # dfR <- df
        # dfR <- dfR[dfR$IoT_15day_CASI_weighted_finance != 0]

        #### CENTERING VARIABLES
        dfR$IoT_15day_CASI_weighted_finance <- dfR$IoT_15day_CASI_weighted_finance - mean(dfR$IoT_15day_CASI_weighted_finance)
        dfR$IoT_30day_CASI_weighted_finance <- dfR$IoT_30day_CASI_weighted_finance - mean(dfR$IoT_30day_CASI_weighted_finance)

        dfR$priceupdate_up <- dfR$priceupdate_up - mean(dfR$priceupdate_up)
        dfR$priceupdate_down <- dfR$priceupdate_down - mean(dfR$priceupdate_down)

        dfR$pct_final_revision_up <- dfR$pct_final_revision_up - mean(dfR$pct_final_revision_up)
        dfR$pct_final_revision_down <- dfR$pct_final_revision_down - mean(dfR$pct_final_revision_down)

        # Partialling out media_listing effect on IOT
        # Regress IOT ~ media_listing and take residuals to use in regression
        dfR$IoT_15day_CASI_weighted_finance <- lm(dfR$IoT_15day_CASI_weighted_finance ~ dfR$media_listing )$residuals
        dfR$IoT_30day_CASI_weighted_finance <- lm(dfR$IoT_30day_CASI_weighted_finance ~ dfR$media_listing )$residuals

        dfR$IoT_15day_CASI_all <- lm(dfR$IoT_15day_CASI_all ~ dfR$media_listing )$residuals
        dfR$IoT_30day_CASI_all <- lm(dfR$IoT_30day_CASI_all ~ dfR$media_listing )$residuals

        dfR$IoT_15day_CASI_news <- lm(dfR$IoT_15day_CASI_news ~ dfR$media_listing )$residuals
        dfR$IoT_30day_CASI_news <- lm(dfR$IoT_30day_CASI_news ~ dfR$media_listing )$residuals

        ##""".split('\n')))

    # clusterby = ('FF49_industry', 'Year')
    clusterby = ('FF49_industry',)
    # clusterby = ('underwriter_rank_avg',)
    # clusterby = ('Year',)
    # clusterby = ('HC1',)
    # clusterby = ''
    # days = 15
    # days = 30
    IOTKEY = 'IoT_{days}day_CASI_weighted_finance'.format(days=days)
    # IOTKEY = 'IoT_{days}day_CASI_all'.format(days=days)
    # IOTKEY = 'IoT_{days}day_CASI_news'.format(days=days)


    # Fit controls only


    col = 'D'
    XVAR = [IOTKEY,
    'np.square({})'.format(IOTKEY),
    '%s:%s' % (IOTKEY, 'priceupdate_up'),
    '%s:%s' % (IOTKEY, 'priceupdate_down'),
    'priceupdate_up',
    'priceupdate_down',
    ]
    X = " + ".join(CONTROLS + XVAR)
    lm = 'close_return ~ ' + X
    xls_empirics(lm, column=col, sheet='initial_returns'+str(days), cluster=clusterby)

    col = 'E'
    XVAR = [IOTKEY,
    'np.square({})'.format(IOTKEY),
    '%s:%s' % (IOTKEY, 'priceupdate_up'),
    '%s:%s' % (IOTKEY, 'priceupdate_down'),
    'priceupdate_up',
    'priceupdate_down',
    'delay_in_price_update',
    ]
    X = " + ".join(CONTROLS + XVAR)
    lm = 'close_return ~ ' + X
    xls_empirics(lm, column=col, sheet='initial_returns'+str(days), cluster=clusterby)




    col = 'F'
    XVAR = [IOTKEY,
    'np.square({})'.format(IOTKEY),
    '%s:%s' % (IOTKEY, 'pct_final_revision_up'),
    '%s:%s' % (IOTKEY, 'pct_final_revision_down'),
    'pct_final_revision_up',
    'pct_final_revision_down',
    ]
    X = " + ".join(CONTROLS + XVAR)
    lm = 'close_return ~ ' + X
    xls_empirics(lm, column=col, sheet='initial_returns'+str(days), cluster=clusterby)



    col = 'G'
    XVAR = [IOTKEY,
    'np.square({})'.format(IOTKEY),
    '%s:%s' % (IOTKEY, 'pct_final_revision_up'),
    '%s:%s' % (IOTKEY, 'pct_final_revision_down'),
    'pct_final_revision_up',
    'pct_final_revision_down',
    'delay_in_price_update',
    ]
    X = " + ".join(CONTROLS + XVAR)
    lm = 'close_return ~ ' + X
    xls_empirics(lm, column=col, sheet='initial_returns'+str(days), cluster=clusterby)
















###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
###################################################################


def xls_price_updates():


    ###############################################
    # First price update regression

    """
    delay_in_price_update measures the length of time left to subscribe in the IPO (expressed as a fraction of total time from initial pricing to listing).
    This information is known at all times because the listing date is set ahead of time and common knowledge.

    """


    ## THIS SETS THE XLS Variable Order
    VARS = [
        'Intercept',
        'CASI',
        'np.square(CASI)',
        # 'CASI:delay_in_price_update',
        # 'CASI:pct_final_revision_up',
        # 'CASI:log(days_from_s1_to_listing)',
        'delay_in_price_update',
        'log(days_from_s1_to_listing)',
        'underwriter_rank_avg',
        # 'underwriter_collective_rank',
        'underwriter_num_leads',
        'underwriter_syndicate_size',
        # 'pct_final_revision_up',
        'VC',
        'share_overhang',
        'log(proceeds)',
        'log(1 + sales)',
        'EPS',
        'M3_indust_rets',
        'M3_initial_returns',
        'Nobs',
        'Rsq',
    ]
    VROW = dict(zip(VARS, [str(n) for n in range(4,100,2)]))



    ###############################################
    # First price update regression

    # dup = df[df['prange_change_first_price_update'].notnull()]
    dup = df[df['size_of_first_price_update'].notnull()]


    CONTROLVAR = [
            'Year',
            'log(days_from_s1_to_listing)',
            'underwriter_rank_avg',
            # 'underwriter_collective_rank',
            'underwriter_num_leads',
            'underwriter_syndicate_size',
            # 'pct_final_revision_up',
            'VC',
            'share_overhang',
            'log(proceeds)',
            'log(1 + sales)',
            'EPS',
            'M3_indust_rets',
            'M3_initial_returns', # 13
            'delay_in_price_update', # 14
        ]


    from rpy2.robjects import r
    list(map(r, """
        require(sandwich)
        require(lmtest)
        require(plm)
        source("clmclx.R")
        df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
        dfR <- df[!is.na(df$size_of_first_price_update)]""".split('\n')))



    YVAR = 'percent_first_price_update'
    # YVAR = 'close_return'
    # Fit controls only
    for i, col in zip([11,12], 'CD'):
        X = " + ".join(CONTROLVAR[:i])
        lm = YVAR + ' ~ ' + X
        xls_empirics(lm, column=col, sheet='updates_CASI', cluster='FF49_industry')


    models = list(itertools.product([15,30], [2,3]))
    for model, col in zip(models, 'FGIJ'):
        days, i = model
        IOTKEY = 'IoT_{days}day_CASI_weighted_finance'.format(days=days)
        XVAR = [IOTKEY,
                'np.square({})'.format(IOTKEY),
                # '{}:{}'.format(IOTKEY, 'delay_in_price_update'),
                # '{}:{}'.format(IOTKEY, 'pct_final_revision_up'),
                # '{}:{}'.format(IOTKEY, 'log(days_from_s1_to_listing)')
                ]
        X = " + ".join(CONTROLVAR + XVAR[:i])
        lm = YVAR + ' ~ ' + X
        xls_empirics(lm, column=col, sheet='updates_CASI', cluster='FF49_industry')



    """
    I make sure CASI is consistent with the date of the event, and sum CASI in the D-days before an event of interest (dependent variables such as filing price range amendments, or the final price revision).
    """



    """ I look at a subsample of firms whic update offer prices early.
    Which kind of firms are more likely to amend offer prices upwards?
    It appears that delayed issues are strongly less likely to update prices upwards,
    issues that attract more attention are more likely to experience upwards amendments
    """



    """
    delay_in_price_update measures the length of time left to subscribe in the IPO (expressed as a fraction of total time from initial pricing to listing).
    This information is known at all times because the listing date is set ahead of time and common knowledge.

    """








# if "__regularization__":
#     ## Regularization - LASSO and Ridge regression
#     import patsy
#     y, x = patsy.dmatrices(Y + " ~ " + X, data=df)
#     from sklearn import linear_model

#     ridge = linear_model.Ridge(alpha=0.5)
#     ridge.fit(x,y)
#     list(zip(['intercept']+X.split('+'), ridge.coef_[0]))


#     lasso = linear_model.Lasso(alpha=0.5)
#     lasso.fit(x,y)
#     list(zip(['intercept']+X.split('+'), lasso.coef_))

#     """ LASSO Coefs as a function of regularization
#     alphas = np.linspace(0,2,100)
#     coefs = []
#     for a in alphas:
#         lasso.set_params(alpha=a)
#         lasso.fit(x,y)
#         coefs.append(lasso.coef_)

#     ax = plt.gca()
#     ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

#     ax.plot(alphas, coefs)
#     ax.set_xscale('log')
#     ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
#     plt.xlabel('alpha')
#     plt.ylabel('weights')
#     plt.title('Ridge coefficients as a function of the regularization')
#     plt.axis('tight')
#     plt.show()
#     """






















def corrmatrix():
    iotkeys = [
            'IoT_15day_CASI_weighted_finance',
            'IoT_30day_CASI_weighted_finance',
            'IoT_15day_CASI_all',
            'IoT_30day_CASI_all',
            'IoT_15day_CASI_news',
            'IoT_30day_CASI_news'
        ]
    df[iotkeys].corr()

    df['log(days_from_s1_to_listing)'] = log(df.days_from_s1_to_listing)
    df['log(proceeds)'] = log(df.proceeds)

    CASI = df['IoT_15day_CASI_weighted_finance']
    CASI30 = df['IoT_30day_CASI_weighted_finance']
    P = df['priceupdate_up']
    FRP = df['pct_final_revision_up']

    df['CASIxP'] =  CASI * P
    df['CASIxFRP'] = CASI * FRP

    df['CASIxP'] =  (CASI - CASI.mean()) * (P - P.mean())
    df['CASIxFRP'] = (CASI - CASI.mean()) * (FRP - FRP.mean())

    df['CASIxP_30'] =  (CASI30 - CASI30.mean()) * (P - P.mean())
    df['CASIxFRP_30'] = (CASI30 - CASI30.mean()) * (FRP - FRP.mean())

    df['log(1+sales)'] = log(df.sales + 1)

    design_vars = [
        'IoT_15day_CASI_weighted_finance',
        'IoT_30day_CASI_weighted_finance',
        'CASIxP',
        'CASIxFRP',
        'priceupdate_up',
        'pct_final_revision_up',
        'number_of_price_updates_up',
        'number_of_price_updates_down',
        'EPS',
        'VC',
        'underwriter_rank_avg',
        'share_overhang',
        'log(proceeds)',
        'log(1+sales)',
        'log(days_from_s1_to_listing)',
        # 'confidential_IPO',
        'media_listing',
        'M3_indust_rets',
        'M3_initial_returns'
        ]

    df[design_vars].corr()[['IoT_15day_CASI_weighted_finance',
        'IoT_30day_CASI_weighted_finance', 'CASIxP', 'CASIxFRP', 'priceupdate_up', 'pct_final_revision_up',]].to_csv("corr_matrix_interactions.csv")

    df[iotkeys + design_vars].corr().to_csv("corr_matrix.csv")


