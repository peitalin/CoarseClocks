
require(stargazer)
require(glmnet)
require(glmnetUtils)
require(nnet)
require(sandwich)
source("clmclx.R")


# df <- data.table::fread("~/Data/IPO/NASDAQ/df.csv", colClasses=c(cik="character", SIC="character", Year="Factor"))
df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="Factor", amends="Factor"))
df <- df[df$days_to_first_price_change > 0]
df <- df[df$days_to_first_price_change < 300]
df <- df[df$cik != '1368308']
df <- df[df$cik != '1087294']
df[['Amendment Delay']] <- df$delay_in_price_update
df[['FF49 Industry']] <- df$FF49_industry

df$FF49.Industry <- df$FF49_industry
df$Amends <- df$amends
df$Days.to.Amendment <- df$days_to_first_price_update
df$Days.to.Amendment[df$Days.to.Amendment == -999] <- 0
df[['Days to Amendment']] <- df$Days.to.amendment
df$Lead.Underwriters <- df$underwriter_num_leads
df$Underwriter.Rank <- df$underwriter_rank_avg
df$Syndicate.Members <- df$underwriter_syndicate_size
df$Share.Overhang <- df$share_overhang
df$log.Sales <- df$'log(1+Sales)'
df$log.Proceeds <- df$'log(Proceeds)'
df$S1A.Amendments <- df$'#S1A Amendments'
df$Ipo.Market.Returns <- df$'IPO Market Returns'
df$Industry.Returns <- df$'Industry Returns'
df$BAA.Spread <- df$'BAA Spread'
# 1st update took longer than a year

# amends
dfa <- df[df$amends != "None"]
dfn <- df[df$amends == "None"]


####### Stargazer Descriptive Stats
# covar <- c('Amendment Delay', '#Syndicate Members', '#Lead Underwriters', 'Underwriter Rank', '#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'IPO Market Returns', 'Industry Returns', 'BAA Spread')
# df <- df[, c('amends', covar), with=F]
# dfa <- dfa[, covar, with=F]

# stargazer(subset(df, amends=="None"), type='text', digits=2)
# stargazer(subset(df, amends!="None"), type='text', digits=2)
###############################################################


# y1 <- df$pct_first_price_change
# eq1 <- y1 ~ df$'Amendment Delay' + I(df[['Amendment Delay']]^2) + df$'#Syndicate Members' + df$'#Lead Underwriters' + df$'Underwriter Rank' + df$'#S1A Amendments' + df$'Share Overhang' + df$'log(1+Sales)' + df$'log(Proceeds)' + df$'CASI' + df$'IPO Market Returns' + df$'Industry Returns' + df$'BAA Spread' + df$'FF49_industry' + df$'Year'

# m1 <- glm(eq1, data=df)
# covar <- c('Amendment Delay', 'Amendment Delay^2', '#Syndicate Members', '#Lead Underwriters', 'Underwriter Rank', '#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'IPO Market Returns', 'Industry Returns', 'BAA Spread')
# stargazer(m1, omit=c("FF49_industry", "Year"), omit.labels=c("Industry Dummies", "Year Dummies"), star.cutoffs=c(0.1, 0.05, 0.01), covariate.labels=covar, type='text')


eq1 <- Days.to.Amendment ~
    underwriter_num_leads +
    underwriter_syndicate_size +
    underwriter_rank_avg +
    amendment +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    Ipo.Market.Returns +
    Industry.Returns +
    BAA.Spread +
    Year

library(pscl)
m1 <- hurdle(eq1, data=df, dist='negbin')
# m11 <- m1
# m11$coefficients$count <- m1$coefficients$zero

stargazer(m1, m1, zero.component=FALSE, omit=c("Year"), omit.labels=c("Year Dummies"), type='text')
stargazer(m1, zero.component=TRUE, omit=c("Year"), omit.labels=c("Year Dummies"), type='text')

covar <- c('\\#Lead Underwriters', '\\#Syndicate Members', 'Underwriter Rank', 'Amends Up', '\\#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'IPO Market Returns', 'Industry Returns', 'BAA Spread')
stargazer(m1, m1, zero.component=TRUE, omit=c("Year"), omit.labels=c("Year Dummies"), covariate.labels=covar)
stargazer(m1, zero.component=FALSE, omit=c("Year"), omit.labels=c("Year Dummies"), covariate.labels=covar, type='text')
# just latex the 1st table and copy real figures from 2nd text table




y2 <- amendment <- factor(df$amends)
y2 <- relevel(y2, ref="None")
eq2 <- y2 ~ df$'Amendment Delay' + df$'#Lead Underwriters' + df$'#Syndicate Members' + df$'Underwriter Rank' + df$'#S1A Amendments' + df$'Share Overhang' + df$'log(1+Sales)' + df$'log(Proceeds)' + df$'CASI' + df$'IPO Market Returns' + df$'Industry Returns' + df$'BAA Spread' + df$'FF49_industry' + df$'Year'


# m2 <- glm(eq2, family=binomial, data=df)
m2 <- multinom(eq2, data=df)

covar <- c('Amendment Delay', '\\#Lead Underwriters', '\\#Syndicate Members', 'Underwriter Rank', '\\#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'IPO Market Returns', 'Industry Returns', 'BAA Spread')
stargazer(m2, omit=c("FF49_industry", "Year"), omit.labels=c("Industry Dummies", "Year Dummies"), star.cutoffs=c(0.1, 0.05, 0.01), covariate.labels=covar, type='text')










library(rstan)
require(brms)
rstan_options (auto_write=TRUE)
options (mc.cores=parallel::detectCores ()) # Run on multiple cores

# NegBinomial

eq5 <- Days.to.Amendment  ~
    (1|FF49.Industry) +
    Amends +
    Lead.Underwriters +
    Syndicate.Members +
    Underwriter.Rank +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    BAA.Spread +
    Ipo.Market.Returns +
    Industry.Returns

attach(dfa)

eq6 <- Days.to.Amendment | trunc(lb=1) ~
    (1|FF49.Industry) +
    Amends +
    Lead.Underwriters +
    Syndicate.Members +
    Underwriter.Rank +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    BAA.Spread +
    Ipo.Market.Returns +
    Industry.Returns

eq7 <- Days.to.Amendment | trunc(lb=1) ~
    (1|FF49.Industry) +
    Amends +
    Lead.Underwriters +
    Syndicate.Members +
    Underwriter.Rank +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    BAA.Spread +
    Ipo.Market.Returns +
    Industry.Returns +
    Year


m4 <- brm(eq5, data=df, family=hurdle_poisson(link='log'), inits=0, n.chains=2, n.iter=8000, n.warmup=2000)
# attach(df) # hurdle: estimates 2 processes: 1 for zero-counts and 1 for positive counts
m5 <- brm(eq5, data=df, family=hurdle_negbinomial(link='log'), inits=0, n.chains=2, n.iter=8000, n.warmup=2000)
# shit fit predicts all 1's and zeros. Maybe this is the logit model?
ypred_m5 <- predict(m5, summary=TRUE)
ypred_m55 <- predict(m5, summary=FALSE)
MASS::write.matrix(ypred_m5, file='ypred_m5.rdat')


make_stancode(eq5, family=hurdle_negbinomial(link='log'),
			  prior=c(set_prior("normal(0,8)", class='shape')))

# make_stancode(eq5, family=negbinomial, prior=c(set_prior("normal(0,8)", class='shape')))
m6 <- brm(eq6, data=dfa, family=negbinomial(link='log'), inits=0, n.chains=2, n.iter=9000, n.warmup=2000)
# m7 <- brm(eq7, data=dfa, family=negbinomial(link='log'), inits=0, n.chains=2, n.iter=9000, n.warmup=2000)


# LOO: compare hurdle-negbin vs. zero-truncated negbin
# LOO(m6, m7)

# summary(m6, waic=TRUE)
hypothesis(m6, "Lead.Underwriters < 0")
"The one-sided 95% credibility interval does not contain zero, thus indicating that the standard
deviations differ from each other in the expected direction. In accordance with this finding,
the Evid.Ratio shows that the hypothesis being tested (i.e., Lead.Underwriters < 0) is about
1076 times more likely than the alternative hypothesis underwriter_num_leads > 0."


## predicted responses # summary=FALSE -> SxN matrix, where S is no. samples
ypred_m6 <- predict(m6, summary=FALSE)
MASS::write.matrix(ypred_m6, file='ypred_m6.rdat')


# extract model stan code and rerun to create stan object for ggmcmc plotting
m66 <- m6$fit

covar <- c('Intercept', 'Amends Up', '#Lead Underwriters', '#Syndicate Members', 'Underwriter Rank', '#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')

require(ggmcmc)
# ggs_histogram(S2, family='b')
P <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b') }, names(m66)), Label = covar)
S2 <- ggs(m66, par_labels=P, family="^b")

ci(S2, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))

source("ggs_custom.R")
pdf("ggs_caterpillar_fixedeffects.pdf")
ggs_caterpillar(S2) + xlab("Parameter Estimates") + theme_tufte(base_size=12, ticks=FALSE)
dev.off()



# ggmcmc(S2, file="ggmcmc-output.pdf", plot=c("density", "traceplot"), param_page=5)


indust <- c("Aero", "Agric", "Autos", "Banks", "Beer", "BldMt", "BusSv", "Chems", "Chips", "Clths",
            "Cnstr", "Coal", "Drugs", "ElcEq", "Fin", "Food", "Fun", "Hardw", "Hlth", "Hshld",
            "Insur", "LabEq", "Mach", "Meals", "MedEq", "Mines", "Oil", "Paper", "PerSv", "RlEst",
            "Rtail", "Rubbr", "Softw", "Steel", "Telcm", "Trans", "Txtls", "Util", "Whlsl")
R <- data.frame(Parameter = paste("r_FF49.Industry.", indust, ".Intercept.", sep=""),
               Label = indust)
S3 <- ggs(m66, par_labels=R, family="^r")


source("ggs_custom.R")
pdf("ggs_caterpillar_industries.pdf")
ggs_caterpillar(S3, greek=TRUE) + xlab("Random Industry Effects Estimates") + theme_tufte(base_size=12)
ggs_caterpillar(S3, greek=TRUE) + xlab("Random Industry Effects Estimates")
# greek=TRUE -> y_scale_discrete (scales y axis)
dev.off()









### TRACEPLOTS
library(gridExtra)
library(ggthemes)

source("ggs_custom.R")

pdf("traceplots1.pdf")
SS <- ggs(m66, par_labels=P, family="AmendsUp")
f1 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f2 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Lead.Underwriters")
f3 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f4 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Syndicate.Members")
f5 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f6 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Underwriter.Rank")
f7 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f8 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="S1A.Amendments")
f9 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f10 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Share.Overhang")
f11 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f12 <- ggs_density(SS) + theme_pander(base_size=9)

grid.arrange(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, ncol=2, nrow=6)
dev.off()



pdf("traceplots2.pdf")
SS <- ggs(m66, par_labels=P, family="log.Sales")
f1 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f2 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="log.Proceeds")
f3 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f4 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="CASI")
f5 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f6 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="BAA.Spread")
f7 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f8 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Ipo.Market.Returns")
f9 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f10 <- ggs_density(SS) + theme_pander(base_size=9)

SS <- ggs(m66, par_labels=P, family="Industry.Returns")
f11 <- ggs_traceplot(SS) + theme_pander(base_size=9)
f12 <- ggs_density(SS) + theme_pander(base_size=9)

grid.arrange(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, ncol=2, nrow=6)
dev.off()





#### rstanarm, no truncated negbinom
# eq7 <- df[['Days to Amendment']] ~
#     (1|df[['FF49 Industry']]) +
#     df$'Amends' +
#     df$'#Lead Underwriters' +
#     df$'#Syndicate Members' +
#     df$'Underwriter Rank' +
#     df$'#S1A Amendments' +
#     df$'Share Overhang' +
#     df$'log(1+Sales)' +
#     df$'log(Proceeds)' +
#     df$'CASI' +
#     df$'IPO Market Returns' +
#     df$'Industry Returns' +
#     df$'BAA Spread'

# require(rstanarm)
# mm7 <- stan_glmer(eq7, data=dfa, family=neg_binomial_2(link='log'))
# # ci95 <- round(posterior_interval(mm5, prob=0.95), 2)
# # launch_shinystan(mm7)

# ypred_mm7 <- posterior_predict(mm7)
# MASS::write.matrix(ypred_mm7, file='ypred_mm7.rdat')























