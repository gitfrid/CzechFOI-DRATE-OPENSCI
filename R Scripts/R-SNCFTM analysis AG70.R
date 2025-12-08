
# ========================================================
# R-SNCFTM ANALYSIS AG70 (CONF-ADJUSTED) - FIXED
# ========================================================

# ========================================================
# R-SNCFTM ANALYSIS AG70 (CONF-ADJUSTED) - FIXED
# ========================================================

library(data.table)
library(survival)
library(optimx)
library(parallel)
library(ggplot2)

# Load SNCFTM function
source("C:/CzechFOI-DRATE-REALBIAS/R Scripts/sncftm-function-confounding-adjustment-ramsave.R")

# ========================================================
# 1. Load and preprocess dataset
# ========================================================
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/Vesely_106_202403141131_AG70.csv")
setnames(df, tolower(names(df)))

# Convert relevant columns to Date
date_cols <- c("datumumrti", paste0("datum_", 1:7))
df[, (date_cols) := lapply(.SD, function(x) as.Date(x, "%Y-%m-%d")), .SDcols = date_cols]

# Baseline date
START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)

# Numeric day columns
df[, death_day := to_day(datumumrti)]
df[, dose_day := to_day(do.call(pmin, c(.SD, na.rm=TRUE))), .SDcols = paste0("datum_", 1:7)]
df[is.na(dose_day), dose_day := NA_integer_]

# Maximum follow-up
MAX_FOLLOW_UP <- 1095
df[, end_day := ifelse(is.na(death_day), MAX_FOLLOW_UP, pmin(death_day, MAX_FOLLOW_UP))]

# ========================================================
# 2. Prepare minimal dataset for SNCFTM
# ========================================================
df_snc <- df[, .(id = .I, death_day, dose_day)]
df_snc[, x := ifelse(!is.na(dose_day), 1L, 0L)]  # treatment indicator
df_snc[, y := 1L]                                  # one row per individual
df_snc[, clost := 0L]
df_snc[, cdeath := 0L]

# ========================================================
# 3. Run SNCFTM confounder-adjusted analysis
# ========================================================
results <- sncftm.conf.robust(
  data = df_snc,
  id = "id",
  time = "death_day",
  x = "x",
  x.modelvars = "1",
  x.linkfunction = "identity",
  y = "y",
  clost = "clost",
  clost.modelvars = "1",
  cdeath = "cdeath",
  cdeath.modelvars = "1",
  blipfunction = 2,
  start.value = 0,
  grid = TRUE,
  grid.range = 1.5,
  grid.increment = 0.05,
  blipupdown = TRUE,
  boot = TRUE,
  R = 200,
  parallel = TRUE,
  seed = 12345
)

# ========================================================
# 4. Debug info
# ========================================================
total_deaths <- nrow(df_snc)
vx_deaths <- sum(df_snc$x)
uvx_deaths <- total_deaths - vx_deaths
cat("DEBUG INFO:\n")
cat("Total deaths:", total_deaths, "\n")
cat("Deaths among vaccinated:", vx_deaths, "\n")
cat("Deaths among unvaccinated:", uvx_deaths, "\n")

# ========================================================
# 5. Compute cumulative risk over follow-up days (FIXED)
# ========================================================
dt_indiv <- results$blip.results

# Compute cumulative sum per individual and normalize to 0–1
dt_indiv[, observed_risk := cumsum(observed_risk)/sum(observed_risk)]
dt_indiv[, never_vax_risk := cumsum(never_vax_risk)/sum(never_vax_risk)]
dt_indiv[, always_vax_risk := cumsum(always_vax_risk)/sum(always_vax_risk)]

# Create plotting dataframe
blip_df <- dt_indiv[, .(day = 1:.N,
                        observed_risk,
                        never_vax_risk,
                        always_vax_risk)]

plot_df <- melt(blip_df, id.vars="day",
                measure.vars=c("observed_risk","never_vax_risk","always_vax_risk"),
                variable.name="regime", value.name="cumulative_risk")
plot_df[, regime := factor(regime,
                           levels=c("observed_risk","never_vax_risk","always_vax_risk"),
                           labels=c("Observed","Never Vaccinated","Always Vaccinated"))]

# ========================================================
# 6. Compute bootstrap confidence intervals (optional)
# ========================================================
boot_df <- as.data.table(results$boot.results)

never_cols <- grep("^Y0\\.t", names(boot_df), value = TRUE)
always_cols <- grep("^Yg\\.t", names(boot_df), value = TRUE)

if(length(never_cols) > 0) {
  ci_never <- t(apply(boot_df[, ..never_cols], 2, quantile, probs = c(0.025,0.975), na.rm=TRUE))
} else ci_never <- data.frame(lower=numeric(0), upper=numeric(0))

if(length(always_cols) > 0) {
  ci_always <- t(apply(boot_df[, ..always_cols], 2, quantile, probs = c(0.025,0.975), na.rm=TRUE))
} else ci_always <- data.frame(lower=numeric(0), upper=numeric(0))

ci_dt <- data.table(
  day = 1:max(nrow(ci_never), nrow(ci_always)),
  never_vax_lower = if(nrow(ci_never)>0) ci_never[,1] else NA,
  never_vax_upper = if(nrow(ci_never)>0) ci_never[,2] else NA,
  always_vax_lower = if(nrow(ci_always)>0) ci_always[,1] else NA,
  always_vax_upper = if(nrow(ci_always)>0) ci_always[,2] else NA
)

# ========================================================
# 7. Plot cumulative risk with 95% CI
# ========================================================
output_plot_file <- "C:/CzechFOI-DRATE-REALBIAS/Plot Results/AG70_sncftm_conf_plot.png"

ci_exists <- !all(is.na(ci_dt$never_vax_lower)) & !all(is.na(ci_dt$always_vax_lower))

p <- ggplot()

if(ci_exists){
  p <- p +
    geom_ribbon(data=ci_dt, aes(x=day, ymin=never_vax_lower, ymax=never_vax_upper), fill="blue", alpha=0.2) +
    geom_ribbon(data=ci_dt, aes(x=day, ymin=always_vax_lower, ymax=always_vax_upper), fill="red", alpha=0.2)
}

p <- p +
  geom_line(data=plot_df, aes(x=day, y=cumulative_risk, color=regime, linetype=regime), linewidth=1) +
  scale_color_manual(values=c("black","blue","red")) +
  scale_linetype_manual(values=c("dotted","solid","solid")) +
  labs(title="Cumulative Risk of Death for AG70 (Confounder-adjusted SNCFTM)",
       x="Follow-up Day", y="Cumulative Risk", color="Regime", linetype="Regime") +
  theme_minimal() +
  theme(text=element_text(size=14))

ggsave(output_plot_file, plot=p, width=12, height=6)
# ========================================================
