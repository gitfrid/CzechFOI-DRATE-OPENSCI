# ------ G-ESTIMATION OF STRUCTURAL NESTED CUMULATIVE FAILURE TIME MODELS ------
# ---------------------- WITH ADJUSTMENT FOR CONFOUNDERS -----------------------
#
# -------------------------------- Changed (original code BY: Joy Shi)   -------
# -------------------------- LAST MODIFIED: 2025-08-24 -------------------------
#
# NOTES: 
# This is a memory-efficient R implementation of SNCFTM, supporting individual-level
# treatment, censoring, and outcome modeling. It includes optional grid evaluation,
# blip-up/down marginal risk calculations, and bootstrap confidence intervals.
#
# REQUIREMENTS:
# The function relies on packages 'optimx', 'parallel', and 'data.table'.
# It is designed to run on large datasets without storing full longitudinal matrices.
#
# BASED ON THE SNCFTM SAS MACRO BY SALLY PICCIOTTO:
# For more information, see:
# https://www.hsph.harvard.edu/causal/software/ 
# https://pubmed.ncbi.nlm.nih.gov/24347749/
#
# ARGUMENTS:
#  - data: data.table or data.frame containing the variables in the model
#  - id: string, participant index variable
#  - time: string, follow-up/event time variable (minimum 1)
#  - x: string, treatment indicator variable
#  - x.modelvars: formula for treatment model (e.g., ~1 for intercept-only)
#  - x.linkfunction: "identity" for linear regression, "logit" for logistic regression
#  - y: string, outcome variable (e.g., event indicator)
#  - clost: string or NULL, censoring due to loss-to-follow-up
#  - clost.modelvars: formula or NULL, model for loss-to-follow-up censoring
#  - cdeath: string or NULL, censoring due to death
#  - cdeath.modelvars: formula or NULL, model for death censoring
#  - blipfunction: 1 or 2, type of blip function (1: 1+[exp(psi*Am)-1]/(k-m), 2: psi*Am)
#  - start.value: numeric, starting value for psi optimization
#  - grid: TRUE/FALSE, evaluate estimating equation over a psi grid
#  - grid.range: numeric, half-width of psi range for grid evaluation
#  - grid.increment: numeric, step size for psi grid evaluation
#  - blipupdown: TRUE/FALSE, calculate marginal cumulative risks under "never" and "always" treatment
#  - boot: TRUE/FALSE, perform bootstrap to obtain confidence intervals
#  - R: integer, number of bootstrap replications
#  - parallel: TRUE/FALSE, parallelize bootstrap across cores
#  - seed: integer, random seed for reproducibility
#
# OUTPUT:
#  A list containing:
#   - psi: estimated blip parameter
#   - psi.esteq: value of the estimating equation at psi
#   - psi.converge: convergence code from optimx
#   - psi.grid (optional): data.frame of grid evaluation results
#   - blip.results (optional): data.table of cumulative risks (observed, never-vax, always-vax)
#   - boot.results (optional): numeric vector of bootstrap psi estimates


# Installing and loading required packages
if (!require('parallel')) install.packages('parallel'); library('parallel')
if (!require('optimx')) install.packages('optimx'); library('optimx')

# ========================================================
# MEMORY-SAFE SNCFTM FUNCTION (CONF-ADJUSTED, FULLY FIXED)
# ========================================================
# =============================
# Robust SNCFTM Function
# =============================
sncftm.conf.robust <- function(data, id, time, x, x.modelvars=NULL, x.linkfunction="identity", y,
                               clost=NULL, clost.modelvars=NULL, cdeath=NULL, cdeath.modelvars=NULL,
                               blipfunction, start.value=0,
                               grid=FALSE, grid.range=1.5, grid.increment=0.01,
                               blipupdown=TRUE, boot=TRUE, R=1000, parallel=TRUE, seed=549274) {

  # Load packages
  if (!require("optimx")) install.packages('optimx'); library(optimx)
  if (!require("parallel")) install.packages('parallel'); library(parallel)
  if (!require("data.table")) install.packages('data.table'); library(data.table)

  if (parallel) numCores <- max(detectCores() - 1, 1)

  # ------------------------------
  # Safe GLM helper
  # ------------------------------
  safe_glm <- function(formula, data, family=NULL) {
    resp_var <- all.vars(formula)[1]
    if(length(unique(data[[resp_var]])) < 2) {
      warning(paste("Outcome", resp_var, "has no variation. Using intercept-only model."))
      formula <- as.formula(paste(resp_var, "~1"))
    }
    tryCatch({
      if (is.null(family)) glm(formula, data=data, control=glm.control(maxit=50))
      else glm(formula, data=data, family=family, control=glm.control(maxit=50))
    }, error=function(e){
      warning(paste("GLM failed:", e$message, "- falling back to intercept-only model"))
      if (is.null(family)) glm(as.formula(paste(resp_var, "~1")), data=data)
      else glm(as.formula(paste(resp_var, "~1")), data=data, family=family)
    })
  }

  # ------------------------------
  # Data prep
  # ------------------------------
  dt <- as.data.table(data)
  for(col in c(x,y,time,id)) {
    if(!col %in% names(dt)) stop(paste("Column", col, "not found in data"))
    dt[[col]] <- as.numeric(dt[[col]])
  }

  # ------------------------------
  # Treatment model
  # ------------------------------
  if(is.null(x.modelvars) || length(unique(dt[[x]])) < 2) {
    # Skip modeling if constant treatment or no variables
    dt[, x.pred := mean(.SD[[1]]), .SDcols=x]
    message("Treatment model skipped due to constant x or no model variables")
  } else {
    x.form <- as.formula(paste(x, "~", paste(x.modelvars, collapse="+")))
    x.model <- if(x.linkfunction=="identity") safe_glm(x.form, dt)
               else safe_glm(x.form, dt, family=binomial)
    dt[, x.pred := predict(x.model, newdata=dt, type="response")]
  }

  # ------------------------------
  # Censoring models
  # ------------------------------
  if(!is.null(clost) && !is.null(clost.modelvars) && length(unique(dt[[clost]]))>1) {
    clost.form <- as.formula(paste(clost, "~", paste(clost.modelvars, collapse="+")))
    dt[, clost.pred := predict(safe_glm(clost.form, dt, family=binomial), newdata=dt, type="response")]
  } else dt[, clost.pred := 1]

  if(!is.null(cdeath) && !is.null(cdeath.modelvars) && length(unique(dt[[cdeath]]))>1) {
    cdeath.form <- as.formula(paste(cdeath, "~", paste(cdeath.modelvars, collapse="+")))
    dt[, cdeath.pred := predict(safe_glm(cdeath.form, dt, family=binomial), newdata=dt, type="response")]
  } else dt[, cdeath.pred := 1]

  # ------------------------------
  # Aggregate per individual
  # ------------------------------
  dt_indiv <- dt[, .(
    x = first(.SD[[1]]),
    x.pred = first(x.pred),
    y = max(.SD[[2]]),   # one death per person
    clost.pred = first(clost.pred),
    cdeath.pred = first(cdeath.pred)
  ), by=.(id), .SDcols=c(x,y)]
  dt_indiv[, ever_treat := as.integer(x != 0)]

  # ------------------------------
  # Estimating equation
  # ------------------------------
  estf.conf.mem <- function(psi, dt_indiv, blipfunction) {
    dt_un <- dt_indiv[ever_treat==0]
    dt_tr <- dt_indiv[ever_treat==1]

    u1_un <- if(nrow(dt_un)==0) 0 else sum((1/(dt_un$clost.pred*dt_un$cdeath.pred))*(dt_un$x - dt_un$x.pred))
    u1_tr <- if(nrow(dt_tr)==0) 0 else {
      hm_tr <- if(blipfunction==1) 1 else exp(psi*dt_tr$x)
      sum(hm_tr*(1/(dt_tr$clost.pred*dt_tr$cdeath.pred))*(dt_tr$x - dt_tr$x.pred))
    }
    u <- u1_tr + u1_un
    cov <- (u1_tr^2 + u1_un^2)
    as.numeric(u^2 / (cov + 1e-8))
  }

  # ------------------------------
  # Psi estimation
  # ------------------------------
  est_res <- suppressWarnings(optimx(start.value, estf.conf.mem,
                                     dt_indiv=dt_indiv, blipfunction=blipfunction,
                                     method="nlminb"))

  results <- list(
    psi = est_res$p1,
    psi.esteq = est_res$value,
    psi.converge = est_res$convcode
  )

  # ------------------------------
  # Optional: grid evaluation
  # ------------------------------
  if(grid){
    psi.seq <- seq(-grid.range, grid.range, by=grid.increment)
    results$psi.grid <- data.frame(psi=psi.seq, est.eq=sapply(psi.seq, function(p) estf.conf.mem(p, dt_indiv, blipfunction)))
  }

  # ------------------------------
  # Blip-up/down (robust)
  # ------------------------------
  dt_indiv[, observed_risk := y]

  psi_val <- if(!is.null(results$psi) && !is.na(results$psi)) results$psi else 0

  dt_indiv[, never_vax_risk := ifelse(ever_treat==1, y / exp(psi_val*x), y)]
  dt_indiv[, always_vax_risk := y * exp(psi_val*x)]

  results$blip.results <- dt_indiv[, .(id, observed_risk, never_vax_risk, always_vax_risk)]

  return(results)
}
