# ======================================================
# AG70 Time-Dependent Cox Model (Vectorized, Stable)
# ======================================================

library(data.table)
library(survival)
library(ggplot2)
library(survminer)
library(plotly)
library(htmlwidgets)

# 1. Load Data
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")
setnames(df, tolower(names(df)))

date_cols <- c("datumumrti", paste0("datum_", 1:7))
df[, (date_cols) := lapply(.SD, as.Date), .SDcols = date_cols]

# 2. Convert to day numbers relative to 2020-01-01
START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)
df[, death_day := to_day(datumumrti)]

# 3. Determine first dose
df[, first_dose := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
df[, dose_day := ifelse(is.na(first_dose), NA_integer_, as.integer(first_dose - START_DATE))]

# 4. Truncate before first dose (study start)
df <- df[is.na(death_day) | death_day >= dose_day]

# 5. Recenter days relative to first population dose
first_population_dose <- min(df$dose_day, na.rm = TRUE)
df[, death_day := ifelse(is.na(death_day), NA_integer_, death_day - first_population_dose)]
df[, dose_day := ifelse(is.na(dose_day), NA_integer_, dose_day - first_population_dose)]

# 6. Define end of follow-up
END_MEASURE <- 1095
df[, end_day := pmin(death_day, END_MEASURE)]
df[is.na(end_day), end_day := END_MEASURE]

# ======================================================
# 7. Create time-dependent exposure dataset (vectorized)
# ======================================================

# Unvaccinated intervals
td_unvax <- df[is.na(dose_day), .(
  id = .I,
  tstart = 0,
  tstop = end_day,
  event = as.integer(!is.na(death_day) & death_day <= end_day),
  vx = 0
)]

# Vaccinated intervals
df_vax <- df[!is.na(dose_day)]
td_vax <- rbind(
  df_vax[, .(
    id = .I,
    tstart = 0,
    tstop = dose_day,
    event = 0,
    vx = 0
  )],
  df_vax[, .(
    id = .I,
    tstart = dose_day,
    tstop = end_day,
    event = as.integer(!is.na(death_day) & death_day <= end_day),
    vx = 1
  )]
)

# Combine
td_df <- rbind(td_unvax, td_vax)

# -----------------------------
# REMOVE zero-length intervals
# -----------------------------
td_df <- td_df[tstop > tstart]

# -----------------------------
# CHECK events per group
# -----------------------------
table(td_df$vx, td_df$event)

# ======================================================
# 8. Fit time-dependent Cox model
# ======================================================
fit <- coxph(Surv(tstart, tstop, event) ~ vx, data = td_df)
summary(fit)

# ======================================================
# 9. Bootstrap Hazard Ratio
# ======================================================
set.seed(123)
B <- 100
ids <- unique(td_df$id)
boot_hr <- replicate(B, {
  sample_ids <- sample(ids, replace = TRUE)
  boot_data <- td_df[id %in% sample_ids]
  fit_boot <- coxph(Surv(tstart, tstop, event) ~ vx, data = boot_data)
  coef(fit_boot)[["vx"]]
})
hr_est <- exp(boot_hr)
cat("Bootstrap HR (vx vs uvx):", round(median(hr_est),2),
    "95% CI [", round(quantile(hr_est,0.025),2), ",", round(quantile(hr_est,0.975),2), "]\n")

# ======================================================
# 10. Kaplan-Meier plot by vaccination status
# ======================================================
km_fit <- survfit(Surv(tstart, tstop, event) ~ vx, data = td_df)
km_plot <- ggsurvplot(
  km_fit,
  data = td_df,
  risk.table = TRUE,
  conf.int = TRUE,
  palette = c("#E7B800", "#2E9FDF"),
  legend.labs = c("Unvaccinated", "Vaccinated"),
  xlab = paste0("Days since first population dose (day ", first_population_dose, ")"),
  ylab = "Survival probability",
  ggtheme = theme_minimal()
)

# ======================================================
# 11. Save KM plot as interactive HTML
# ======================================================
html_folder <- "C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-CCW bootstrap"
if (!dir.exists(html_folder)) dir.create(html_folder, recursive = TRUE)
plotly_obj <- ggplotly(km_plot$plot)
saveWidget(
  plotly_obj,
  file = file.path(html_folder, "KM_tdcox_AG70.html"),
  selfcontained = TRUE
)

print(km_plot)
