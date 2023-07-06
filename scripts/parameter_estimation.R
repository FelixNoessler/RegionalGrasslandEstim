library(JuliaCall)
julia_setup()

julia_command('import Pkg; Pkg.activate(".");')
julia_source("4_scripts/run model/validation.jl")


parameter_names <- c(
    "sigma_biomass",
    "sigma_evaporation",
    "sigma_soilmoisture",
    "moisture_conv",
    "senescence_intercept",
    "senescence_rate")


ll <- function(param){
    param_list <- split(param, parameter_names)

    julia_do.call(
        "ModelValidation.loglikelihood_model",
        list(
            plotID="AEG01",
            inf_p=param_list,
            startyear=2012,
            endyear=2021))
}

ll(c(1000,2,3, 100, 0.005, 0.05))

# start_time <- Sys.time()
# ll(c(1000,2,3, 100, 0.005, 0.05))
# end_time <- Sys.time()
# end_time - start_time



#--------------------------------------------
library(BayesianTools)

bayesianSetup = createBayesianSetup(
    likelihood = ll, 
    lower = c(0.1, 0.1, 0.1, 0.1, 0.000001, 0.000001), 
    upper = c(100000000, 100, 100, 5000, 0.1, 0.01),
    names = parameter_names,
    parallel = F)

settings = list(
    iterations = 10000, 
    message = T, 
    nrChains = 1,
    burnin= 0)

out <- runMCMC(
    bayesianSetup = bayesianSetup, 
    sampler = "DREAMzs", 
    settings = settings)

summary(out)
plot(out)
correlationPlot(out)
marginalPlot(out, prior = TRUE)
marginalLikelihood(out)
MAP(out)

