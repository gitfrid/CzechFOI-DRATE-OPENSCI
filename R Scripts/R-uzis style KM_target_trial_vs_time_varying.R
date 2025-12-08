library(data.table)
library(dplyr)
library(ggplot2)
library(pbapply)
library(survival)
library(survminer)

#---------------------------
# 1. Load data
#---------------------------
# t <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")
t <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/Vesely_106_202403141131_AG70.csv")

#---------------------------
# 2. Convert dates and age
#---------------------------
date_cols <- grep("^Datum_|DatumUmrti", names(t), value = TRUE)
t[, (date_cols) := lapply(.SD, function(x) as.IDate(x, format = "%Y-%m-%d")), .SDcols = date_cols]

if (!"Age" %in% names(t)) {
  t[, Age := 70L]
} else {
  t[, Age := as.integer(Age)]
}

#---------------------------
# 3. Study start/end like survivalCCW
#---------------------------
START_DATE <- min(unlist(t[, date_cols, with = FALSE]), na.rm = TRUE)
END_DATE   <- max(unlist(t[, date_cols, with = FALSE]), na.rm = TRUE)
MAX_FOLLOW_UP <- 1095L

to_day <- function(x) as.integer(x - START_DATE)

t[, death_day := to_day(DatumUmrti)]
dose_cols <- grep("^Datum_", names(t), value = TRUE)
t[, first_dose_date := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = dose_cols]
t[, dose_day := ifelse(is.infinite(first_dose_date), NA_integer_, to_day(first_dose_date))]

# End of follow-up: min(death, administrative censoring)
t[, end_day := pmin(ifelse(is.na(death_day), MAX_FOLLOW_UP, death_day), MAX_FOLLOW_UP)]

#---------------------------
# 4. Define immunity lag
#---------------------------
IMMUNITY_LAG <- 0L  # set as you like

# Helper: status at a specific day (time-varying)
get_status <- function(d, dose_day, end_day) {
  if (!is.na(dose_day) && (dose_day + IMMUNITY_LAG <= d) && d <= end_day) {
    return("vax")
  } else if (d <= end_day) {
    return("unvax")
  } else {
    return(NA_character_)
  }
}

#---------------------------
# 4b. Baseline (time-fixed) status for daily/weekly comparison
#     -> vaccinated at baseline ONLY if dose_day + lag <= 0
#     (This is a different estimand than the KM target-trial below.)
#---------------------------
t[, baseline_status := ifelse(!is.na(dose_day) & (dose_day + IMMUNITY_LAG) <= 0, "vax", "unvax")]

#---------------------------
# 5. Daily aggregation (time-varying vs time-fixed)
#---------------------------
days <- 0:(MAX_FOLLOW_UP - 1L)  # 0..1094

get_daily_summary <- function(status_mode = c("time_varying", "time_fixed")) {
  mode <- match.arg(status_mode)
  pblapply(days, function(d) {
    if (mode == "time_varying") {
      t[, status := sapply(1:.N, function(i) get_status(d, dose_day[i], end_day[i]))]
    } else {
      # time-fixed: baseline assignment, contribute until end_day
      t[, status := ifelse(d <= end_day, baseline_status, NA_character_)]
    }
    t[, .(
      deaths = sum(death_day == d, na.rm = TRUE),
      population = sum(!is.na(status))
    ), by = status][, day := d][]
  }) %>% rbindlist()
}

daily_summary_tv <- get_daily_summary("time_varying")
daily_summary_tf <- get_daily_summary("time_fixed")

# Calculate mortality rate & 95% CI
add_rate_ci <- function(df) {
  df[, mortality_rate := deaths / pmax(population, 1)]
  df[, se := sqrt(pmax(mortality_rate * (1 - mortality_rate) / pmax(population, 1), 0))]
  df[, lower := pmax(0, mortality_rate - 1.96 * se)]
  df[, upper := pmin(1, mortality_rate + 1.96 * se)]
  df[]
}

daily_summary_tv <- add_rate_ci(daily_summary_tv)
daily_summary_tf <- add_rate_ci(daily_summary_tf)

#---------------------------
# 6. Weekly aggregation (both modes)
#---------------------------
aggregate_weekly <- function(daily_df) {
  daily_df %>%
    mutate(week = ceiling((day + 1)/7)) %>%  # +1 because days start at 0
    group_by(week, status) %>%
    summarise(
      deaths = sum(deaths),
      population = sum(population),
      mortality_rate = deaths / pmax(population, 1),
      se = sqrt(pmax(mortality_rate * (1 - mortality_rate) / pmax(population, 1), 0)),
      lower = pmax(0, mortality_rate - 1.96 * se),
      upper = pmin(1, mortality_rate + 1.96 * se),
      .groups = "drop"
    )
}

weekly_summary_tv <- aggregate_weekly(daily_summary_tv)
weekly_summary_tf <- aggregate_weekly(daily_summary_tf)

#---------------------------
# 7. Save results (daily/weekly)
#---------------------------
fwrite(daily_summary_tv, "AG70_daily_summary_timevarying.csv")
fwrite(weekly_summary_tv, "AG70_weekly_summary_timevarying.csv")
fwrite(daily_summary_tf, "AG70_daily_summary_timefixed.csv")
fwrite(weekly_summary_tf, "AG70_weekly_summary_timefixed.csv")

#---------------------------
# 8. Plot: Daily UZIS-style (time-varying)
#---------------------------
p_daily_tv <- ggplot(daily_summary_tv, aes(x = day, y = mortality_rate, color = status, fill = status)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("unvax" = "red", "vax" = "blue")) +
  scale_fill_manual(values = c("unvax" = "red", "vax" = "blue")) +
  labs(
    title = "AG70 Mortality: Vaccinated vs Unvaccinated (Daily, Time-Varying)",
    x = "Day since study start", y = "Mortality rate",
    color = "Vaccination status", fill = "Vaccination status"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_daily_timevarying.png",
       plot = p_daily_tv, width = 10, height = 6, dpi = 300)

#---------------------------
# 9. Plot: Weekly UZIS-style (time-varying)
#---------------------------
p_weekly_tv <- ggplot(weekly_summary_tv, aes(x = week, y = mortality_rate, color = status, fill = status)) +
  geom_line(size = 1.2) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2, color = NA) +
  scale_color_manual(values = c("unvax" = "red", "vax" = "blue")) +
  scale_fill_manual(values = c("unvax" = "red", "vax" = "blue")) +
  labs(
    title = "AG70 Mortality: Vaccinated vs Unvaccinated (Weekly, Time-Varying)",
    x = "Week since study start", y = "Mortality rate",
    color = "Vaccination status", fill = "Vaccination status"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_weekly_timevarying.png",
       plot = p_weekly_tv, width = 10, height = 6, dpi = 300)

#---------------------------
# 10. Plot: Daily overlay (time-varying solid vs time-fixed dashed)
#---------------------------
p_daily_compare <- ggplot() +
  geom_line(data = daily_summary_tv, aes(x = day, y = mortality_rate, color = status), size = 1) +
  geom_line(data = daily_summary_tf, aes(x = day, y = mortality_rate, color = status), linetype = "dashed", size = 1) +
  scale_color_manual(values = c("unvax" = "red", "vax" = "blue")) +
  labs(
    title = "AG70 Mortality: Time-Varying (solid) vs Time-Fixed Baseline (dashed)",
    x = "Day since study start", y = "Mortality rate", color = "Vaccination status"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_daily_compare_tv_vs_tf.png",
       plot = p_daily_compare, width = 10, height = 6, dpi = 300)
	   
#---------------------------
# 10A. Plot: Weekly overlay (time-varying solid vs time-fixed dashed)
#---------------------------
p_weekly_compare <- ggplot() +
  geom_line(data = weekly_summary_tv, aes(x = week, y = mortality_rate, color = status), size = 1.2) +
  geom_line(data = weekly_summary_tf, aes(x = week, y = mortality_rate, color = status), linetype = "dashed", size = 1.2) +
  scale_color_manual(values = c("unvax" = "red", "vax" = "blue")) +
  labs(
    title = "AG70 Mortality: Time-Varying (solid) vs Time-Fixed Baseline (dashed) - Weekly",
    x = "Week since study start", y = "Mortality rate",
    color = "Vaccination status"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "top", plot.title = element_text(hjust = 0.5))

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_weekly_overlay.png",
       plot = p_weekly_compare, width = 10, height = 6, dpi = 300)

#============================================================
# 11. KM CURVES
#============================================================

#---------- 11A. KM (bias-free target-trial style; time-fixed per person)
# Group assignment: a person is "vax" if they are vaccinated before censoring
t[, km_group := ifelse(!is.na(dose_day) & (dose_day + IMMUNITY_LAG) <= end_day, "vax", "unvax")]

# Entry time (left truncation) and exit time
t[, start_day := ifelse(km_group == "vax", dose_day + IMMUNITY_LAG, 0L)]
t[, stop_day  := end_day]

# Event indicator occurs only if death_day is within [start_day, stop_day]
t[, event := as.integer(!is.na(death_day) & death_day >= start_day & death_day <= stop_day)]

# Keep only with positive follow-up
km_tf <- t[(stop_day - start_day) > 0, .(start_day, stop_day, event, group = km_group)]

surv_tf <- survfit(Surv(time = start_day, time2 = stop_day, event = event) ~ group, data = km_tf)

p_km_tf <- ggsurvplot(
  surv_tf, data = km_tf,
  conf.int = TRUE, pval = TRUE, risk.table = TRUE,
  palette = c("red", "blue"),
  xlab = "Days since study start", ylab = "Survival probability",
  title = "Kaplan–Meier (Target-Trial Style, Bias-Free)",
  legend.title = "Vaccination status"
)

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_KM_bias_free.png",
       plot = p_km_tf$plot, width = 10, height = 6, dpi = 300)
ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_KM_bias_free_risktable.png",
       plot = p_km_tf$table, width = 10, height = 3, dpi = 300)

#---------- 11B. KM (time-varying/switching illustration; contributes to both groups)
# Unvaccinated interval: [0, min(switch, end_day)) with event if death before switch
t[, switch_day := ifelse(is.na(dose_day), NA_integer_, dose_day + IMMUNITY_LAG)]

unvax_iv <- t[, .(
  start_day = 0L,
  stop_day  = pmin(end_day, ifelse(is.na(switch_day), end_day, switch_day)),
  event     = as.integer(!is.na(death_day) & (is.na(switch_day) | death_day < switch_day) & death_day <= end_day),
  group     = "unvax"
)]

# Keep positive length
unvax_iv <- unvax_iv[(stop_day - start_day) > 0]

# Vaccinated interval: [switch, end_day] with event if death after switch
vax_iv <- t[!is.na(switch_day), .(
  start_day = switch_day,
  stop_day  = end_day,
  event     = as.integer(!is.na(death_day) & death_day >= switch_day & death_day <= end_day),
  group     = "vax"
)]
vax_iv <- vax_iv[(stop_day - start_day) > 0]

km_tv_switch <- rbindlist(list(unvax_iv, vax_iv), use.names = TRUE, fill = TRUE)

surv_tv <- survfit(Surv(time = start_day, time2 = stop_day, event = event) ~ group, data = km_tv_switch)

p_km_tv <- ggsurvplot(
  surv_tv, data = km_tv_switch,
  conf.int = TRUE, pval = TRUE, risk.table = TRUE,
  palette = c("red", "blue"),
  xlab = "Days since study start", ylab = "Survival probability",
  title = "Kaplan–Meier (Time-Varying/Switching Illustration)",
  legend.title = "Vaccination status"
)

ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_KM_timevarying_switch.png",
       plot = p_km_tv$plot, width = 10, height = 6, dpi = 300)
ggsave("C:/CzechFOI-DRATE-REALBIAS/Plot Results/R-uzis style/AG70_KM_timevarying_switch_risktable.png",
       plot = p_km_tv$table, width = 10, height = 3, dpi = 300)

#============================================================
# Notes:
# - KM (bias-free) emulates a target trial: one group per person; vaccinated enter at switch.
# - KM (time-varying/switching) is for illustration; people can appear in both groups.
#   For causal inference with time-varying exposure, prefer time-dependent Cox/Poisson models.
#============================================================
