
require(stargazer)
# require(glmnet)
# require(glmnetUtils)
require(nnet)
library(pscl)
require(ggmcmc)
library(gridExtra)
library(ggthemes)

# df <- data.table::fread("~/Data/IPO/NASDAQ/df.csv", colClasses=c(cik="character", SIC="character", Year="Factor"))
df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="Factor", amends="Factor"))
df <- df[df$days_to_first_price_change > 0]
df <- df[df$days_to_first_price_change < 300]
# 1st update took longer than a year
df <- df[df$cik != '1368308']
df <- df[df$cik != '1087294']
# Two bad ipos
df$FF49.Industry <- df$FF49_industry
df$Amends <- df$amends
df$Days.to.Amendment <- df$days_to_first_price_update
df$Days.to.Amendment[df$Days.to.Amendment == -999] <- 0
df$Lead.Underwriters <- df$underwriter_num_leads
df$Underwriter.Rank <- df$underwriter_rank_avg
df$Syndicate.Members <- df$underwriter_syndicate_size
df$Share.Overhang <- df$share_overhang
df$log.Sales <- df$'log(1+Sales)'
df$log.Proceeds <- df$'log(Proceeds)'
df$S1A.Amendments <- df$'#S1A Amendments'/(1+df$Days.to.Amendment)
# df$S1A.Amendments <- df$'#S1A Amendments'
df$Ipo.Market.Returns <- df$'IPO Market Returns'
df$Industry.Returns <- df$'Industry Returns'
df$BAA.Spread <- df$'BAA Spread'
df$Amendment.Delay <- df$delay_in_price_update
df$AmendsUp <- ifelse(df$amends == 'Up', 1, 0)
df$underwriter_first_lead <- sapply(df$underwriter_first_lead, function(x) { gsub(' ', '.', x) })
df$underwriter_groups <- sapply(df$underwriter_groups, function(x) { gsub(' ', '.', x) })
# amends
dfa <- df[df$amends != "None"]

# create variable that is Days.to.Amendment iff amends=="Down"
df$Days.to.Amend.Down <- df$Days.to.Amendment
df$Days.to.Amend.Down[df$amends != "Down"] <- 0


####### Stargazer Descriptive Stats
covar <- c('Days.to.Amendment', 'Amendment.Delay', '#Syndicate Members', '#Lead Underwriters', 'Underwriter Rank', '#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'IPO Market Returns', 'Industry Returns', 'BAA Spread')
dfa <- df[df$amends != "None"]
dfa <- dfa[, c('amends', covar), with=F]

dfn <- df[df$amends == "None"]
dfn <- dfn[, c('amends', covar), with=F]

# stargazer(dfa, type='text', digits=2)
# stargazer(dfn, type='text', digits=2)
###############################################################



eq1 <- Days.to.Amendment ~ Lead.Underwriters +
    Syndicate.Members +
    Underwriter.Rank +
    AmendsUp +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    Ipo.Market.Returns +
    Industry.Returns +
    BAA.Spread +
    Year |  Lead.Underwriters + Syndicate.Members + Underwriter.Rank + S1A.Amendments + Share.Overhang + log.Sales + log.Proceeds + CASI + Ipo.Market.Returns + Industry.Returns + BAA.Spread + FF49.Industry + Amendment.Delay

m1 <- hurdle(eq1, data=df, dist='negbin')
summary(m1)
# m11 <- m1
# m11$coefficients$count <- m1$coefficients$zero


stargazer(m1, m1, zero.component=FALSE, omit=c("Year", "FF49.Industry"), omit.labels=c("Year Dummies", "Industry Dummies"), type='text')
stargazer(m1, zero.component=TRUE, omit=c("Year", "FF49.Industry"), omit.labels=c("Year Dummies", "Industry Dummies"), type='text')

covar <- c('\\textit{\\#Lead Underwriters}',
           '\\textit{\\#Syndicate Members}',
           '\\textit{Underwriter Rank}',
           '\\textit{Amends Up}',
           '\\textit{\\#S1A Amendments}',
           '\\textit{Share Overhang}',
           '\\textit{log(1+Sales)}',
           '\\textit{log(Proceeds)}',
           '\\textit{CASI}',
           '\\textit{IPO Market Returns}',
           '\\textit{Industry Returns}',
           '\\textit{BAA Spread}',
           '\\textit{Intercept}')
stargazer(m1, m1, zero.component=TRUE, omit=c("Year", "FF49.Industry"), omit.labels=c("Year Dummies", "Industry Dummies"), covariate.labels=covar)
stargazer(m1, zero.component=FALSE, omit=c("Year", "FF49.Industry"), omit.labels=c("Year Dummies", "Industry Dummies"), covariate.labels=covar, type='text')
# just latex the 1st table and copy real figures from 2nd text table



df$amends <- relevel(factor(df$amends), 'None')
eq2 <- amends ~ Amendment.Delay +
    Lead.Underwriters +
    Syndicate.Members +
    S1A.Amendments +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    CASI +
    Ipo.Market.Returns +
    Industry.Returns +
    BAA.Spread +
    FF49.Industry +
    Year

m2 <- multinom(eq2, data=df)

covar <- c('\\textit{Amendment Delay}',
           '\\textit{#Lead Underwriters}',
           '\\textit{#Syndicate Members}',
           '\\textit{#S1A Amendments}',
           '\\textit{Underwriter Rank}',
           '\\textit{Share Overhang}',
           '\\textit{log(1+Sales)}',
           '\\textit{log(Proceeds)}',
           '\\textit{CASI}',
           '\\textit{IPO Market Returns}',
           '\\textit{Industry Returns}',
           '\\textit{BAA Spread}',
           '\\textit{Intercept}'
           )

stargazer(m2, omit=c("FF49.Industry", "Year"), omit.labels=c("Industry Dummies", "Year Dummies"), star.cutoffs=c(0.1, 0.05, 0.01), covariate.labels=covar, type='text')

stargazer(m2, omit=c("FF49.Industry", "Year"), omit.labels=c("Industry Dummies", "Year Dummies"), star.cutoffs=c(0.1, 0.05, 0.01), type='text')








library(rstan)
require(brms)
rstan_options (auto_write=TRUE)
options (mc.cores=parallel::detectCores ()) # Run on multiple cores

# NegBinomial



attach(dfa)
eq6 <- Days.to.Amendment | trunc(lb=1) ~
    (1|FF49.Industry) +
    (1|Year) +
    AmendsUp +
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


# trait: amends at all
# traithu:
# AmendsUp in count model, omitted from zero model:
eq7 <- Days.to.Amendment  ~ 0 + AmendsUp + (0 + trait|FF49.Industry) + (0 + trait|Year) +
    trait * (
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
    Industry.Returns)


#. Sometimes, predictors should only influence the ZIH part but not the actual response (or
# vice versa). As this cannot be modeled with the trait variable, two other internally generated and
# reserved (numeric) variables, namely main and spec, are supported. main is 1 for the response part
# and 0 for the ZIH part of the model. For spec it is the other way round. Suppose that x1 should
# only influence the actual response, and x2 only the ZIH process. We can write this as follows:
# formula = y ~ 0 + main + spec + main:x1 + spec:x2. The main effects of main or spec
# serve as intercepts, while the interaction terms main:x1 and spec:x2 ensure that x1 and x2 only
# predict one part of the model, respectively.
### spec: zero process only
### main: count process only

attach(df)
eq8 <- Days.to.Amendment  ~ 0 + main + spec +
    (0 + main|FF49.Industry) +
    # (0 + main|Year) +
    main:AmendsUp +
    main:Lead.Underwriters +
    main:Syndicate.Members +
    main:S1A.Amendments +
        main:Underwriter.Rank +
        main:Share.Overhang +
        main:log.Sales +
        main:log.Proceeds +
        main:CASI +
        main:BAA.Spread +
        main:Ipo.Market.Returns +
        main:Industry.Returns +
    spec:Amendment.Delay +
        # spec:S1A.Amendments +
        spec:Underwriter.Rank +
        spec:Share.Overhang +
        spec:log.Sales +
        spec:log.Proceeds +
        spec:CASI +
        spec:BAA.Spread +
        spec:Ipo.Market.Returns +
        spec:Industry.Returns

eq9 <- Days.to.Amendment  ~ 0 + main + spec +
    (0 + main|FF49.Industry) +
    # (0 + main|Year) +
    main:AmendsUp +
    main:Lead.Underwriters +
    main:Syndicate.Members +
        main:S1A.Amendments +
        main:Underwriter.Rank +
        main:Share.Overhang +
        main:log.Sales +
        main:log.Proceeds +
        main:CASI +
        main:BAA.Spread +
        main:Ipo.Market.Returns +
        main:Industry.Returns +
    spec:Amendment.Delay +
        spec:S1A.Amendments +
        spec:Underwriter.Rank +
        spec:Share.Overhang +
        spec:log.Sales +
        spec:log.Proceeds +
        spec:CASI +
        spec:BAA.Spread +
        spec:Ipo.Market.Returns +
        spec:Industry.Returns



eq11 <- Days.to.Amend.Down  ~ 0 + main + spec +
    (0 + main|FF49.Industry) +
    # (0 + main|Year) +
    main:Lead.Underwriters +
    main:Syndicate.Members +
    main:S1A.Amendments +
        main:Underwriter.Rank +
        main:Share.Overhang +
        main:log.Sales +
        main:log.Proceeds +
        main:CASI +
        main:BAA.Spread +
        main:Ipo.Market.Returns +
        main:Industry.Returns +
    spec:Amendment.Delay +
    spec:S1A.Amendments +
        spec:Underwriter.Rank +
        spec:Share.Overhang +
        spec:log.Sales +
        spec:log.Proceeds +
        spec:CASI +
        spec:BAA.Spread +
        spec:Ipo.Market.Returns +
        spec:Industry.Returns

# m8 <- brm(eq8, data=df, family=hurdle_negbinomial(link='log'), inits=0, n.chains=1, n.iter=50, n.warmup=20)

# m9 <- brm(eq9, data=df, family=hurdle_negbinomial(link='log'), inits=0, n.chains=2, n.iter=5000, n.warmup=2000)
# ypred_m9 <- predict(m9, summary=FALSE)
# MASS::write.matrix(ypred_m9, file='ypred_m9.rdat')

m11 <- brm(eq11, data=df, family=hurdle_negbinomial(link='log'), inits=0, n.chains=2, n.iter=5000, n.warmup=2000)
ypred_m11 <- predict(m11, summary=FALSE)
MASS::write.matrix(ypred_m11, file='ypred_m11.rdat')


# # make_stancode(eq5, family=negbinomial, prior=c(set_prior("normal(0,8)", class='shape')))
# m6 <- brm(eq6, data=dfa, family=negbinomial(link='log'), inits=0, n.chains=2, n.iter=5000, n.warmup=1000)
# ## predicted responses # summary=FALSE -> SxN matrix, where S is no. samples
# ypred_m6 <- predict(m6, summary=FALSE)
# MASS::write.matrix(ypred_m6, file='ypred_m6.rdat')

# # attach(df) # hurdle: estimates 2 processes: 1 for zero-counts and 1 for positive counts
# # make_stancode(eq7, family=hurdle_negbinomial(link="log"))
# m7 <- brm(eq7, data=df, family=hurdle_negbinomial(link='log'), inits=0, n.chains=2, n.iter=5000, n.warmup=1000)
# ypred_m7 <- predict(m7, summary=FALSE)
# MASS::write.matrix(ypred_m7, file='ypred_m7.rdat')



# LOO: compare hurdle-negbin vs. zero-truncated negbin
# LOO(m6, m7)
# Incompatible since different number of observations

hypothesis(m6, "Lead.Underwriters < 0")
"The one-sided 95% credibility interval does not contain zero, thus indicating that the standard
deviations differ from each other in the expected direction. In accordance with this finding,
the Evid.Ratio shows that the hypothesis being tested (i.e., Lead.Underwriters < 0) is about
1076 times more likely than the alternative hypothesis underwriter_num_leads > 0."

# extract model stan code and rerun to create stan object for ggmcmc plotting
m66 <- m6$fit
m77 <- m7$fit


covar <- c('Intercept', 'Amends Up', '#Lead Underwriters', '#Syndicate Members', 'Underwriter Rank', '#S1A Amendments', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
P <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b') }, names(m66)), Label = covar)
S2 <- ggs(m66, par_labels=P, family="^b")
ci(S2, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))
ci(ggs(m66, family='shape'), thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))





# year <- c("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014")
# Ryear <- data.frame(Parameter=paste("r_Year.", year, ".Intercept.", sep=""), Label=year)
# S4 <- ggs(m66, par_labels=Ryear, family="^r_Year")
# pdf("ggs_caterpillar_year.pdf")
# ggs_caterpillar(S4, greek=TRUE) + xlab("Random Year Effects Estimates") + theme_tufte(base_size=12)
# # greek=TRUE -> y_scale_discrete (scales y axis)
# dev.off()




covar.main <- c('Intercept', 'Amends Up', '#Lead Underwriters', '#Syndicate Members', '#S1A Amendments', 'Underwriter Rank', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
# covar.main <- c('Intercept', '#Lead Underwriters', '#Syndicate Members', '#S1A Amendments', 'Underwriter Rank', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
covar.spec <- c('Intercept', 'Amendment Delay','#S1A Amendments', 'Underwriter Rank', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
Pmain <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b_main') }, names(m99)), Label=covar.main)
Pspec <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b_spec') }, names(m99)), Label=covar.spec)

Pmain$Parameter <- as.factor(sapply(Pmain$Parameter, function(x) { gsub(':', '.', x) }))
Pspec$Parameter <- as.factor(sapply(Pspec$Parameter, function(x) { gsub(':', '.', x) }))


m9main <- ggs(m99, par_labels=Pmain, family="^b_main")
m9spec <- ggs(m99, par_labels=Pspec, family="^b_spec")

ci(m9main, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))
ci(m9spec, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))
ci(ggs(m99, family='shape'), thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))





indust <- c("Aero", "Agric", "Autos", "Banks", "Beer", "BldMt", "Books", "BusSv", "Chems", "Chips", "Clths",
            "Cnstr", "Coal", "Drugs", "ElcEq", "FabPr", "Fin", "Food", "Fun", "Hardw", "Hlth", "Hshld",
            "Insur", "LabEq", "Mach", "Meals", "MedEq", "Mines", "Oil", "Other", "Paper", "PerSv", "RlEst",
            "Rtail", "Rubbr", "Ships", "Softw", "Steel", "Telcm", "Toys", "Trans", "Txtls", "Util", "Whlsl")
# Rindust <- data.frame(Parameter = paste("r_FF49.Industry.", indust, ".Intercept.", sep=""), Label = indust)
Rindust <- data.frame(Parameter = paste("r_FF49.Industry.", indust, ".main.", sep=""), Label = indust)
S3 <- ggs(m99, par_labels=Rindust, family="^r_FF49")

source("ggs_custom.R")

pdf("ggs_caterpillar_industries.pdf")
ggs_caterpillar(S3, greek=TRUE) + xlab("Random Industry Effects Estimates") + theme_tufte(base_size=12)
# greek=TRUE -> y_scale_discrete (scales y axis)
dev.off()












### TRACEPLOTS

source("ggs_custom.R")
theme_blank1 = theme_pander(base_size=9) + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
theme_blank2 = theme_pander(base_size=9) + theme(strip.background=element_blank(), strip.text.x=element_blank(), axis.title.y=element_blank())


pdf("traceplots1.pdf")
SS <- ggs(m66, par_labels=P, family="AmendsUp")
f1 <- ggs_traceplot(SS) + theme_blank1
f4 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Lead.Underwriters")
f2 <- ggs_traceplot(SS) + theme_blank1
f5 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Syndicate.Members")
f3 <- ggs_traceplot(SS) + theme_blank1
f6 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Underwriter.Rank")
f7 <- ggs_traceplot(SS) + theme_blank1
f10 <- ggs_density(SS) + theme_blank2
#######################

SS <- ggs(m66, par_labels=P, family="S1A.Amendments")
f8 <- ggs_traceplot(SS) + theme_blank1
f11 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Share.Overhang")
f9 <- ggs_traceplot(SS) + theme_blank1
f12 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="log.Sales")
f13 <- ggs_traceplot(SS) + theme_blank1
f16 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="log.Proceeds")
f14 <- ggs_traceplot(SS) + theme_blank1
f17 <- ggs_density(SS) + theme_blank2
###############################

SS <- ggs(m66, par_labels=P, family="CASI")
f15 <- ggs_traceplot(SS) + theme_blank1
f18 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="BAA.Spread")
f19 <- ggs_traceplot(SS) + theme_blank1
f22 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Ipo.Market.Returns")
f20 <- ggs_traceplot(SS) + theme_blank1
f23 <- ggs_density(SS) + theme_blank2

SS <- ggs(m66, par_labels=P, family="Industry.Returns")
f21 <- ggs_traceplot(SS) + theme_blank1
f24 <- ggs_density(SS) + theme_blank2
#######################

grid.arrange(f1, f2, f3, f4, f5, f6,
             f7, f8, f9, f10, f11, f12,
             f13, f14, f15, f16, f17, f18,
             f19, f20, f21, f22, f23, f24, ncol=3, nrow=8)
dev.off()
#########################







source("ggs_custom.R")
theme_blank1 = theme_pander(base_size=9) + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
theme_blank2 = theme_pander(base_size=9) + theme(strip.background=element_blank(), strip.text.x=element_blank(), axis.title.y=element_blank())

# m99
pdf("traceplots99_main-hurdle.pdf")
SS <- ggs(m99, par_labels=Pmain, family="b_main.AmendsUp")
f1 <- ggs_traceplot(SS) + theme_blank1
f4 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Lead.Underwriters")
f2 <- ggs_traceplot(SS) + theme_blank1
f5 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Syndicate.Members")
f3 <- ggs_traceplot(SS) + theme_blank1
f6 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Underwriter.Rank")
f7 <- ggs_traceplot(SS) + theme_blank1
f10 <- ggs_density(SS) + theme_blank2
#######################

SS <- ggs(m99, par_labels=Pmain, family="b_main.S1A.Amendments")
f8 <- ggs_traceplot(SS) + theme_blank1
f11 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Share.Overhang")
f9 <- ggs_traceplot(SS) + theme_blank1
f12 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.log.Sales")
f13 <- ggs_traceplot(SS) + theme_blank1
f16 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.log.Proceeds")
f14 <- ggs_traceplot(SS) + theme_blank1
f17 <- ggs_density(SS) + theme_blank2
###############################

SS <- ggs(m99, par_labels=Pmain, family="b_main.CASI")
f15 <- ggs_traceplot(SS) + theme_blank1
f18 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.BAA.Spread")
f19 <- ggs_traceplot(SS) + theme_blank1
f22 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Ipo.Market.Returns")
f20 <- ggs_traceplot(SS) + theme_blank1
f23 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pmain, family="b_main.Industry.Returns")
f21 <- ggs_traceplot(SS) + theme_blank1
f24 <- ggs_density(SS) + theme_blank2
#######################

grid.arrange(f1, f2, f3, f4, f5, f6,
             f7, f8, f9, f10, f11, f12,
             f13, f14, f15, f16, f17, f18,
             f19, f20, f21, f22, f23, f24, ncol=3, nrow=8)
dev.off()
#########################


pdf("traceplots99_spec-hurdle.pdf")
SS <- ggs(m99, par_labels=Pspec, family="b_spec.Amendment.Delay")
f1 <- ggs_traceplot(SS) + theme_blank1
f4 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.Underwriter.Rank")
f2 <- ggs_traceplot(SS) + theme_blank1
f5 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.S1A.Amendments")
f3 <- ggs_traceplot(SS) + theme_blank1
f6 <- ggs_density(SS) + theme_blank2
#########################

SS <- ggs(m99, par_labels=Pspec, family="b_spec.Share.Overhang")
f7 <- ggs_traceplot(SS) + theme_blank1
f10 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.log.Sales")
f8 <- ggs_traceplot(SS) + theme_blank1
f11 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.log.Proceeds")
f9 <- ggs_traceplot(SS) + theme_blank1
f12 <- ggs_density(SS) + theme_blank2
###############################

SS <- ggs(m99, par_labels=Pspec, family="b_spec.CASI")
f13 <- ggs_traceplot(SS) + theme_blank1
f16 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.BAA.Spread")
f14 <- ggs_traceplot(SS) + theme_blank1
f17 <- ggs_density(SS) + theme_blank2

SS <- ggs(m99, par_labels=Pspec, family="b_spec.Ipo.Market.Returns")
f15 <- ggs_traceplot(SS) + theme_blank1
f18 <- ggs_density(SS) + theme_blank2
################################

SS <- ggs(m99, par_labels=Pspec, family="b_spec.Industry.Returns")
f19 <- ggs_traceplot(SS) + theme_blank1
f22 <- ggs_density(SS) + theme_blank2


grid.arrange(f1, f2, f3, f4, f5, f6,
             f7, f8, f9, f10, f11, f12,
             f13, f14, f15, f16, f17, f18,
             f19, f22, ncol=3, nrow=8)
dev.off()
#########################









# M11 #########
############
source("ggs_custom.R")
theme_blank1 = theme_pander(base_size=9) + theme(axis.title.x=element_blank(), axis.title.y=element_blank())
theme_blank2 = theme_pander(base_size=9) + theme(strip.background=element_blank(), strip.text.x=element_blank(), axis.title.y=element_blank())


m111 <- m11$fit

indust <- c("Aero", "Agric", "Autos", "Banks", "Beer", "BldMt", "Books", "BusSv", "Chems", "Chips", "Clths",
            "Cnstr", "Coal", "Drugs", "ElcEq", "FabPr", "Fin", "Food", "Fun", "Hardw", "Hlth", "Hshld",
            "Insur", "LabEq", "Mach", "Meals", "MedEq", "Mines", "Oil", "Other", "Paper", "PerSv", "RlEst",
            "Rtail", "Rubbr", "Ships", "Softw", "Steel", "Telcm", "Toys", "Trans", "Txtls", "Util", "Whlsl")
# Rindust <- data.frame(Parameter = paste("r_FF49.Industry.", indust, ".Intercept.", sep=""), Label = indust)
Rindust <- data.frame(Parameter = paste("r_FF49.Industry.", indust, ".main.", sep=""), Label = indust)
S3 <- ggs(m111, par_labels=Rindust, family="^r_FF49")


pdf("ggs_caterpillar_industries-m11.pdf")
ggs_caterpillar(S3, greek=TRUE) + xlab("Random Industry Effects Estimates") + theme_tufte(base_size=12)
# greek=TRUE -> y_scale_discrete (scales y axis)
dev.off()




covar.main <- c('Intercept', '#Lead Underwriters', '#Syndicate Members', '#S1A Amendments', 'Underwriter Rank', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
covar.spec <- c('Intercept', 'Amendment Delay','#S1A Amendments', 'Underwriter Rank', 'Share Overhang', 'log(1+Sales)', 'log(Proceeds)', 'CASI', 'BAA Spread', 'IPO Market Returns', 'Industry Returns')
Pmain <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b_main') }, names(m111)), Label=covar.main)
Pspec <- data.frame(Parameter = Filter(function(x) { gdata::startsWith(x, 'b_spec') }, names(m111)), Label=covar.spec)

Pmain$Parameter <- as.factor(sapply(Pmain$Parameter, function(x) { gsub(':', '.', x) }))
Pspec$Parameter <- as.factor(sapply(Pspec$Parameter, function(x) { gsub(':', '.', x) }))


m11main <- ggs(m111, par_labels=Pmain, family="^b_main")
m11spec <- ggs(m111, par_labels=Pspec, family="^b_spec")

ci(m11main, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))
ci(m11spec, thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))
ci(ggs(m111, family='shape'), thick_ci = c(0.05, 0.95), thin_ci = c(0.025, 0.975))



# m111
pdf("traceplots11_main-hurdle.pdf")
SS <- ggs(m111, par_labels=Pmain, family="b_main.Lead.Underwriters")
f1 <- ggs_traceplot(SS) + theme_blank1
f4 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.Syndicate.Members")
f2 <- ggs_traceplot(SS) + theme_blank1
f5 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.Underwriter.Rank")
f3 <- ggs_traceplot(SS) + theme_blank1
f6 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.S1A.Amendments")
f7 <- ggs_traceplot(SS) + theme_blank1
f10 <- ggs_density(SS) + theme_blank2
#######################

SS <- ggs(m111, par_labels=Pmain, family="b_main.Share.Overhang")
f8 <- ggs_traceplot(SS) + theme_blank1
f11 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.log.Sales")
f9 <- ggs_traceplot(SS) + theme_blank1
f12 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.log.Proceeds")
f13 <- ggs_traceplot(SS) + theme_blank1
f16 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.CASI")
f14 <- ggs_traceplot(SS) + theme_blank1
f17 <- ggs_density(SS) + theme_blank2
###############################

SS <- ggs(m111, par_labels=Pmain, family="b_main.BAA.Spread")
f15 <- ggs_traceplot(SS) + theme_blank1
f18 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.Ipo.Market.Returns")
f19 <- ggs_traceplot(SS) + theme_blank1
f22 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pmain, family="b_main.Industry.Returns")
f20 <- ggs_traceplot(SS) + theme_blank1
f23 <- ggs_density(SS) + theme_blank2


#######################

grid.arrange(f1, f2, f3, f4, f5, f6,
             f7, f8, f9, f10, f11, f12,
             f13, f14, f15, f16, f17, f18,
             f19, f20, f22, f23, ncol=3, nrow=8)
dev.off()
#########################


pdf("traceplots11_spec-hurdle.pdf")
SS <- ggs(m111, par_labels=Pspec, family="b_spec.Amendment.Delay")
f1 <- ggs_traceplot(SS) + theme_blank1
f4 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.Underwriter.Rank")
f2 <- ggs_traceplot(SS) + theme_blank1
f5 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.S1A.Amendments")
f3 <- ggs_traceplot(SS) + theme_blank1
f6 <- ggs_density(SS) + theme_blank2
#########################

SS <- ggs(m111, par_labels=Pspec, family="b_spec.Share.Overhang")
f7 <- ggs_traceplot(SS) + theme_blank1
f10 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.log.Sales")
f8 <- ggs_traceplot(SS) + theme_blank1
f11 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.log.Proceeds")
f9 <- ggs_traceplot(SS) + theme_blank1
f12 <- ggs_density(SS) + theme_blank2
###############################

SS <- ggs(m111, par_labels=Pspec, family="b_spec.CASI")
f13 <- ggs_traceplot(SS) + theme_blank1
f16 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.BAA.Spread")
f14 <- ggs_traceplot(SS) + theme_blank1
f17 <- ggs_density(SS) + theme_blank2

SS <- ggs(m111, par_labels=Pspec, family="b_spec.Ipo.Market.Returns")
f15 <- ggs_traceplot(SS) + theme_blank1
f18 <- ggs_density(SS) + theme_blank2
################################

SS <- ggs(m111, par_labels=Pspec, family="b_spec.Industry.Returns")
f19 <- ggs_traceplot(SS) + theme_blank1
f22 <- ggs_density(SS) + theme_blank2


grid.arrange(f1, f2, f3, f4, f5, f6,
             f7, f8, f9, f10, f11, f12,
             f13, f14, f15, f16, f17, f18,
             f19, f22, ncol=3, nrow=8)
dev.off()
#########################






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



source("clmclx.R")
attach(df)
eq9 <- close_return ~ Underwriter.Rank +
    # Lead.Underwriters +
    # Syndicate.Members +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    Industry.Returns +
    Ipo.Market.Returns +
    BAA.Spread +
    CASI +
    # I(CASI^2) +
    number_of_price_updates_up +
    number_of_price_updates_down +
    pct_final_revision_up +
    pct_final_revision_down +
    CASI:pct_final_revision_up +
    CASI:pct_final_revision_down +
    Amendment.Delay +
    Year + FF49.Industry


######## Cluster Robust OLS
# m09 <- lm(eq9, data=df)
# clx(m09, 1, FF49_industry)


require(lme4)
# library(lmerTest)
# m9.lmer1 <- lmer(update(eq9, ~ . + (1 | FF49_industry) + (1 | Year)))
m9.lmer2 <- lmer(update(eq9, ~ . + (1 | underwriter_groups)))
summary(m9.lmer2)
dotplot(ranef(m9.lmer2, condVar=TRUE))

# require(nlme)
# # m9.lme1 <- lme(eq9, random = ~ 1 | FF49_industry/underwriter_groups)
# m9.lme1 <- lme(eq9, random = ~ 1 | underwriter_groups)

covar <- c('\\textit{Underwriter Rank}', '\\textit{Share Overhang}', '\\textit{log(1+Sales)}', '\\textit{log(Proceeds)}', '\\textit{Industry Returns}', '\\textit{IPO Market Returns}', '\\textit{BAA Spread}', '\\textit{CASI}', '\\textit{$#\\Delta P_{up}$}', '\\textit{$#\\Delta P_{down}$}', '\\textit{$FPR_{up}$}', '\\textit{$FPR_{down}$}',  '\\textit{$CASI \\times FPR_{up}$}', '\\textit{$CASI \\times FPR_{down}$}',  '\\textit{Intercept}')
stargazer(m9.lme1, omit=c("Year", "FF49.Industry"), omit.labels=c("Year Dummies", "Industry Dummies"), covariate.labels=covar, type='text')





eq22 <- close_return ~ Underwriter.Rank +
    (1|underwriter_first_lead) +
    # (1|underwriter_groups) +
    # Lead.Underwriters +
    # Syndicate.Members +
    Share.Overhang +
    log.Sales +
    log.Proceeds +
    Industry.Returns +
    Ipo.Market.Returns +
    BAA.Spread +
    CASI +
    # I(CASI^2) +
    number_of_price_updates_up +
    number_of_price_updates_down +
    pct_final_revision_up +
    pct_final_revision_down +
    CASI:pct_final_revision_up +
    CASI:pct_final_revision_down +
    Year + FF49.Industry


m22 <- brm(eq22, data=df, family=gaussian, inits=0, n.chains=1, n.iter=2000, n.warmup=1000)
m222 <- m22$fit


# year <- c("2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014")
# R <- data.frame(Parameter=paste("r_.", year, ".Intercept.", sep=""), Label=year)
source("ggs_custom.R")
S6 <- ggs(mm22, family="^r_underwriter")
# pdf("ggs_caterpillar_year.pdf")
ggs_caterpillar(S6, greek=TRUE) + xlab("Random Year Effects Estimates")
# greek=TRUE -> y_scale_discrete (scales y axis)
# dev.off()








