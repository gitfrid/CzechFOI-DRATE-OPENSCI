# ========================================================
# FULL SCRIPT: SNCFTM IV ANALYSIS FOR AG70
# ========================================================

# --- 0. Load required packages ---
library(data.table)
library(survival)
library(optimx)
library(parallel)
library(ggplot2)
# Load SNCFTM IV function
source("C:/CzechFOI-DRATE-REALBIAS/R Scripts/sncftm-function-iv.R")


# ========================================================
# 1. Load and preprocess dataset
# ========================================================
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")

# Convert all column names to lowercase
setnames(df, tolower(names(df)))

# Check for required columns
required_cols <- c("datumumrti", paste0("datum_", 1:7))
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

# Convert relevant columns to Date
date_cols <- c("datumumrti", paste0("datum_", 1:7))
df[, (date_cols) := lapply(.SD, function(x) as.Date(x, format = "%Y-%m-%d")), .SDcols = date_cols]

# Baseline date and conversion function
START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)

# Numeric day columns
df[, death_day := to_day(datumumrti)]
df[, dose_days := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
df[, dose_day := ifelse(is.na(dose_days), NA_integer_, to_day(dose_days))]

# Follow-up time
max_observed_day <- max(c(df$death_day, df$dose_day), na.rm = TRUE)
MAX_FOLLOW_UP <- min(max_observed_day, 1095)
df[, end_day := ifelse(is.na(death_day), MAX_FOLLOW_UP, death_day)]

# ========================================================
# 2. Prepare data for SNCFTM function
# ========================================================

# Create "long" dataset for time-to-event
df_long <- df[, .(id = .I, day = 1:MAX_FOLLOW_UP), by = 1:nrow(df)]
df_long <- merge(df_long, df[, .(id = .I, death_day, dose_day)], by = "id", all.x = TRUE)

# Outcome (Y) and treatment (X) indicators
df_long[, y := as.integer(day == death_day)]
df_long[, x := as.integer(!is.na(dose_day) & day >= dose_day)]

# Instrument (Z) – here we can use e.g., eligibility or random assignment proxy
# For simulation, we'll just use random binary instrument
set.seed(123)
df_long[, z := sample(0:1, .N, replace = TRUE)]

# No censoring indicators for simplicity
df_long[, clost := 0]
df_long[, cdeath := 0]

# ========================================================
# 3. SNCFTM IV function
# ========================================================
# (Insert sncftm.iv() function here; use the full function from your previous R code)

# ========================================================
# 4. Run SNCFTM IV analysis
# ========================================================
results <- sncftm.iv(
  data = df_long,
  id = "id",
  time = "day",
  z = "z",
  z.modelvars = ~1,
  z.family = "binomial",
  x = "x",
  y = "y",
  clost = "clost",
  clost.modelvars = ~1,
  cdeath = "cdeath",
  cdeath.modelvars = ~1,
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
# 5. Prepare cumulative risk data for plotting
# ========================================================
blip_df <- as.data.table(results$blip.results)
setnames(blip_df, old = c("y", "y.0", "y.g"), new = c("observed_risk", "never_vax_risk", "always_vax_risk"))
blip_df[, day := 1:.N]

# Reshape for ggplot
plot_df <- melt(
  blip_df,
  id.vars = "day",
  measure.vars = c("observed_risk", "never_vax_risk", "always_vax_risk"),
  variable.name = "regime",
  value.name = "cumulative_risk"
)
plot_df[, regime := factor(regime,
                           levels = c("observed_risk", "never_vax_risk", "always_vax_risk"),
                           labels = c("Observed", "Never Vaccinated", "Always Vaccinated"))]

# ========================================================
# 6. Compute 95% CI from bootstrap results
# ========================================================
boot_df <- as.data.table(results$boot.results)
never_cols <- grep("^Y0\\.t", names(boot_df), value = TRUE)
always_cols <- grep("^Yg\\.t", names(boot_df), value = TRUE)

ci_never <- t(apply(boot_df[, ..never_cols], 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE))
ci_always <- t(apply(boot_df[, ..always_cols], 2, quantile, probs = c(0.025, 0.975), na.rm = TRUE))

ci_dt <- data.table(
  day = 1:nrow(ci_never),
  never_vax_lower = ci_never[,1],
  never_vax_upper = ci_never[,2],
  always_vax_lower = ci_always[,1],
  always_vax_upper = ci_always[,2]
)

# ========================================================
# 7. Plot cumulative risk curves with 95% CI
# ========================================================
ggplot() +
  # Confidence intervals
  geom_ribbon(data = ci_dt, aes(x = day, ymin = never_vax_lower, ymax = never_vax_upper), fill = "blue", alpha = 0.2) +
  geom_ribbon(data = ci_dt, aes(x = day, ymin = always_vax_lower, ymax = always_vax_upper), fill = "red", alpha = 0.2) +
  # Point estimate lines
  geom_line(data = plot_df, aes(x = day, y = cumulative_risk, color = regime, linetype = regime), size = 1) +
  scale_color_manual(values = c("black", "blue", "red")) +
  scale_linetype_manual(values = c("dotted", "solid", "solid")) +
  labs(
    title = "Cumulative Risk of Death for AG70 with 95% CI",
    x = "Follow-up Day",
    y = "Cumulative Risk",
    color = "Regime",
    linetype = "Regime"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 14))
