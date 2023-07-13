library(JuliaCall)
library(BayesianTools)

prepare_julia <- function(){
    julia_eval('if isdir("Grassland_data_analysis") cd("Grassland_data_analysis"); end')
    julia_eval('import Pkg; Pkg.activate(".")')
    julia_command('import RegionalGrasslandSim as sim')
    julia_command('using RegionalGrasslandValid')
    julia_call("set_datapath", "../RegionalGrasslandData")
}

#--------------------------------------------
df <- expand.grid(id=1:9, explo=c("HEG", "AEG", "SEG"))
selected_plots <- paste0(df$explo, "0", df$id)

parameter_names <- c(
    "sigma_biomass",
    "sigma_evaporation",
    "sigma_soilmoisture",
    "moisture_conv",
    "senescence_intercept",
    "senescence_rate",
    "below_competition_strength",
    "trampling_factor", 
    "grazing_factor",
    "mowing_factor",
    "max_SRSA_water_reduction",
    "max_SLA_water_reduction",
    "max_AMC_nut_reduction",
    "max_SRSA_nut_reduction")

prior <- createTruncatedNormalPrior(
            # σ_bio   σ_ev σ_mo m_c  s_i   s_r below  tram   graz  mow  SRSA SLA  AMC  SRSA_n
    mean =  c(10000,  10,  10,  0.8, 0.1,  1,  0.001, 1,     1,    1,   0.2, 0.5, 0.5, 0.5),
    sd =    c(100,    10,  10,  0.2, 5,    5,  0.01,  0.2,   0.2,  0.2, 0.2, 0.2, 0.2, 0.2),
    lower = c(0.1,    0.1, 0.1, 0.1, 0,    0,  0.0,   0.1,   0.1,  0.1, 0.0, 0.0, 0.0, 0.0), 
    upper = c(100000, 100, 100, 1.2, 10,   10, 0.01,  5,     5,    5,   1.0, 1.0, 1.0, 1.0))

ll <- function(param){

    prepared <- julia_eval('@isdefined RegionalGrasslandValid')
    if (! prepared){
        prepare_julia()
    } 

    param_list <- split(param, parameter_names)
    loglik = 0
    for (plotID in selected_plots){
        loglik_plot = julia_do.call(
            "loglikelihood_model", 
            list(
                julia_eval("sim"),
                plotID=plotID,
                inf_p=param_list,
                startyear=2012,
                endyear=2021))
        loglik = loglik + loglik_plot
    }

    return(loglik)
}

# test_p <- prior$sampler()
# start_time <- Sys.time()
# ll(test_p)
# end_time <- Sys.time()
# end_time - start_time


bayesianSetup = createBayesianSetup(
    likelihood = ll, 
    prior = prior,
    names = parameter_names,
    parallel = T,
    parallelOptions = list(
        packages = list("BayesianTools", "JuliaCall"),
        variables = list(
            "parameter_names", 
            "selected_plots", 
            "prepare_julia")))

out <- runMCMC(
    bayesianSetup = bayesianSetup, 
    sampler = "DREAMzs", 
    settings = list(
        iterations = 20000, 
        message = T, 
        nrChains = 2,
        burnin = 0))
# stopParallel(bayesianSetup)

# summary(out)
plot(out)
# correlationPlot(out)
marginalPlot(out, prior = TRUE)
# marginalLikelihood(out)
MAP(out)

