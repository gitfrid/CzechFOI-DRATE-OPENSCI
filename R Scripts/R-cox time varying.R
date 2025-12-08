# Load required packages
library(data.table)
library(survival)
library(survminer)  # For prettier survival plots
library(dplyr)
library(ggplot2)
library(splines)

# --- 1. Load data ---
# df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/Vesely_106_202403141131_AG70.csv")
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")

# Convert column names to lowercase
setnames(df, tolower(names(df)))

# --- 2. Prepare date columns ---
required_cols <- c("datumumrti", paste0("datum_", 1:7))
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) stop("Missing columns: ", paste(missing_cols, collapse = ", "))

# Convert relevant columns to Date format
date_cols <- required_cols
df[, (date_cols) := lapply(.SD, function(x) as.Date(x, format = "%Y-%m-%d")), .SDcols = date_cols]

# --- 3. Convert to day numbers relative to 2020-01-01 ---
START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)
df[, death_day := to_day(datumumrti)]
#df[, dose_days := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
#df[, dose_day := ifelse(is.na(dose_days), NA_integer_, to_day(dose_days))]
df[, dose_days := do.call(pmax, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
df[, dose_day := ifelse(is.na(dose_days), NA_integer_, to_day(dose_days))]


# --- 4. Define censoring day (end of follow-up) ---
max_observed_day <- max(c(df$death_day, df$dose_day), na.rm = TRUE)
MAX_FOLLOW_UP <- min(max_observed_day, 1095)
df[, end_day := ifelse(is.na(death_day), MAX_FOLLOW_UP, death_day)]

# --- 5. Expand into time-varying format (1 or 2 rows per subject) ---
tv_df <- rbindlist(lapply(1:nrow(df), function(i) {
  row <- df[i]
  
  died <- !is.na(row$death_day) && row$death_day == row$end_day
  status <- ifelse(died, 1L, 0L)
  
  if (!is.na(row$dose_day)) {
    data.table(
      id = i,
      start = c(0, row$dose_day),
      stop = c(row$dose_day, row$end_day),
      status = c(0L, status),
      vaccinated = c(0L, 1L)
    )
  } else {
    data.table(
      id = i,
      start = 0,
      stop = row$end_day,
      status = status,
      vaccinated = 0L
    )
  }
}))

# --- 5.1 Filter invalid intervals ---
invalid_intervals <- tv_df[is.na(stop) | is.na(start) | stop <= start]
num_invalid <- nrow(invalid_intervals)
cat(sprintf("Number of invalid intervals (stop <= start or NA): %d\n", num_invalid))

tv_df_clean <- tv_df[!(is.na(stop) | is.na(start) | stop <= start)]
cat(sprintf("Number of valid intervals after filtering: %d\n", nrow(tv_df_clean)))

# --- 6. Fit Cox proportional hazards model ---
cox_model <- coxph(Surv(start, stop, status) ~ vaccinated + cluster(id), data = tv_df_clean)
print(summary(cox_model))

# --- 7. Check proportional hazards assumption ---
ph_test <- cox.zph(cox_model)
print(ph_test)

# Plot proportional hazards test results
p3 <- ggcoxzph(ph_test)
png("Proportional_Hazards_Assumption.png", width = 800, height = 600)
print(p3)
dev.off()

# --- 8. Kaplan-Meier survival curves ---
fit_km <- survfit(Surv(start, stop, status) ~ vaccinated, data = tv_df_clean)

p1 <- ggsurvplot(
  fit_km, 
  data = tv_df_clean,
  conf.int = TRUE,
  risk.table = TRUE,
  legend.title = "Vaccination Status",
  legend.labs = c("Unvaccinated", "Vaccinated"),
  xlab = "Time (days)",
  ylab = "Survival Probability",
  title = "Kaplan-Meier Survival Curves: Vaccinated vs Unvaccinated",
  palette = c("#E7B800", "#2E9FDF"),
  ggtheme = theme_minimal()
)
ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-cox time varying/KM_Survival_Curves_Vx_vs_Uvx.png", p1$plot, width = 8, height = 6)
print(p1)

# --- 9. Cumulative incidence plot ---
p2 <- ggsurvplot(
  fit_km,
  data = tv_df_clean,
  fun = "event",
  conf.int = TRUE,
  risk.table = TRUE,
  legend.title = "Vaccination Status",
  legend.labs = c("Unvaccinated", "Vaccinated"),
  xlab = "Time (days)",
  ylab = "Cumulative Incidence",
  title = "Cumulative Incidence of Death: Vaccinated vs Unvaccinated",
  palette = c("#E7B800", "#2E9FDF"),
  ggtheme = theme_minimal()
)
ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-cox time varying/Cumulative_Incidence_Vx_vs_Uvx.png", p2$plot, width = 8, height = 6)
print(p2)


# --- 10. Estimate and plot smoothed hazard rates for each group ---

# Function to estimate smoothed hazard from survival data using kernel smoothing
smoothed_hazard <- function(fit, group_label) {
  times <- fit$time
  surv_probs <- fit$surv
  
  # Calculate discrete hazard for intervals between times
  d_surv <- surv_probs[-length(surv_probs)] - surv_probs[-1]
  hazard <- d_surv / surv_probs[-length(surv_probs)]
  
  # Align times with hazard vector length
  hazard_times <- times[-1]
  
  df_hazard <- data.frame(
    time = hazard_times,
    hazard = hazard,
    group = group_label
  )
  
  # Smooth hazard with smoothing spline
  spline_fit <- smooth.spline(df_hazard$time, df_hazard$hazard, spar = 0.5)
  df_hazard$smoothed_hazard <- predict(spline_fit, df_hazard$time)$y
  
  df_hazard
}


# Get KM fits per vaccination group
fit_km_unvax <- survfit(Surv(start, stop, status) ~ 1, data = tv_df_clean[vaccinated == 0])
fit_km_vax <- survfit(Surv(start, stop, status) ~ 1, data = tv_df_clean[vaccinated == 1])

haz_unvax <- smoothed_hazard(fit_km_unvax, "Unvaccinated")
haz_vax <- smoothed_hazard(fit_km_vax, "Vaccinated")

haz_df <- rbind(haz_unvax, haz_vax)

# Plot smoothed hazards
p_hazard <- ggplot(haz_df, aes(x = time, y = smoothed_hazard, color = group)) +
  geom_line(size = 1) +
  labs(title = "Smoothed Hazard Rates by Vaccination Status",
       x = "Time (days)",
       y = "Hazard rate",
       color = "Group") +
  theme_minimal() +
  scale_color_manual(values = c("Unvaccinated" = "#E7B800", "Vaccinated" = "#2E9FDF"))

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-cox time varying/Smoothed_Hazard_Rates.png", p_hazard, width = 8, height = 6)
print(p_hazard)


# --- 11. Time-varying vaccine efficacy (VE) plot using time-stratified Cox model ---

# Stratify follow-up into intervals (e.g., 0-90, 90-180, 180-365, 365+ days)
tv_df_clean[, interval := cut(stop, breaks = c(0, 90, 180, 365, MAX_FOLLOW_UP), right = FALSE)]

# Fit Cox model with interaction between vaccination and interval to estimate HR by interval
cox_model_strat <- coxph(Surv(start, stop, status) ~ vaccinated * interval + cluster(id), data = tv_df_clean)

# Extract coefficients and covariance matrix
coef_sum <- summary(cox_model_strat)$coefficients

# Base effect of vaccinated (vaccinated coefficient) is for baseline interval [0,90)
# Interaction terms are for other intervals, we calculate HR for each interval as:
# HR(interval) = exp(beta_vaccinated + beta_interaction) where beta_interaction = 0 for first interval

# Extract intervals
interval_levels <- levels(tv_df_clean$interval)

# Calculate HR and confidence intervals per interval
ve_data <- data.frame(
  interval = interval_levels,
  HR = NA_real_,
  lower = NA_real_,
  upper = NA_real_
)

# Vaccinated main effect row index
idx_vaccinated <- which(rownames(coef_sum) == "vaccinated")

for (i in seq_along(interval_levels)) {
  int_name <- interval_levels[i]
  # Interaction term name: vaccinated:interval[...]
  if (i == 1) {
    # Baseline interval, no interaction term
    beta <- coef_sum[idx_vaccinated, "coef"]
    se <- coef_sum[idx_vaccinated, "se(coef)"]
  } else {
    # Interaction term name format may differ, inspect:
    # They look like: vaccinated:interval[90,180)
    inter_term_name <- paste0("vaccinated:interval", int_name)
    inter_term_idx <- which(rownames(coef_sum) == inter_term_name)
    if (length(inter_term_idx) == 1) {
      beta <- coef_sum[idx_vaccinated, "coef"] + coef_sum[inter_term_idx, "coef"]
      se <- sqrt(coef_sum[idx_vaccinated, "se(coef)"]^2 + coef_sum[inter_term_idx, "se(coef)"]^2)  # Approximate
    } else {
      # Sometimes interaction terms are named differently due to factor levels:
      # Try escaped names
      inter_term_idx <- grep(paste0("vaccinated:interval", gsub("[\\(\\)\\[\\]]", "", int_name)), rownames(coef_sum))
      if (length(inter_term_idx) == 1) {
        beta <- coef_sum[idx_vaccinated, "coef"] + coef_sum[inter_term_idx, "coef"]
        se <- sqrt(coef_sum[idx_vaccinated, "se(coef)"]^2 + coef_sum[inter_term_idx, "se(coef)"]^2)
      } else {
        beta <- coef_sum[idx_vaccinated, "coef"]
        se <- coef_sum[idx_vaccinated, "se(coef)"]
      }
    }
  }
  
  hr <- exp(beta)
  lower <- exp(beta - 1.96 * se)
  upper <- exp(beta + 1.96 * se)
  
  ve_data$HR[i] <- hr
  ve_data$lower[i] <- lower
  ve_data$upper[i] <- upper
}

ve_data$VE <- 1 - ve_data$HR
ve_data$VE_lower <- 1 - ve_data$upper
ve_data$VE_upper <- 1 - ve_data$lower

# Plot VE with 95% CI by time interval
p_ve <- ggplot(ve_data, aes(x = interval, y = VE)) +
  geom_point(size = 3, color = "#2E9FDF") +
  geom_errorbar(aes(ymin = VE_lower, ymax = VE_upper), width = 0.2, color = "#2E9FDF") +
  ylim(-0.5, 1) +
  labs(
    title = "Estimated Vaccine Efficacy (VE) Over Time Intervals",
    x = "Follow-up Time Interval (days)",
    y = "Vaccine Efficacy (1 - HR)"
  ) +
  theme_minimal()

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-cox time varying/Vaccine_Efficacy_Over_Time.png", p_ve, width = 8, height = 6)
print(p_ve)
