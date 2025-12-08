# ========================================================
# SIMPLE SURVIVAL ANALYSIS: AG70 (NO IV)
# ========================================================

library(data.table)
library(survival)
library(ggplot2)

# ========================================================
# 1. Load and preprocess dataset
# ========================================================
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")

setnames(df, tolower(names(df)))
date_cols <- c("datumumrti", paste0("datum_", 1:7))
df[, (date_cols) := lapply(.SD, as.Date, format = "%Y-%m-%d"), .SDcols = date_cols]

START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)

df[, death_day := to_day(datumumrti)]
df[, dose_days := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
df[, dose_day := ifelse(is.na(dose_days), NA_integer_, to_day(dose_days))]

MAX_FOLLOW_UP <- min(max(c(df$death_day, df$dose_day), na.rm = TRUE), 1095)
df[, end_day := ifelse(is.na(death_day), MAX_FOLLOW_UP, death_day)]

# ========================================================
# 2. Build survival dataset in counting-process format
# ========================================================
# Each person contributes two intervals: before vax and after vax (if vaccinated)

df_surv <- rbindlist(lapply(1:nrow(df), function(i) {
  row <- df[i]
  id <- i
  
  # Interval before vaccination (if exists)
  pre <- if (!is.na(row$dose_day)) {
    data.table(
      id = id,
      start = 0,
      stop = row$dose_day,
      event = 0,
      vax = 0
    )
  } else NULL
  
  # Interval after vaccination (or entire follow-up if never vax)
  post <- data.table(
    id = id,
    start = ifelse(is.na(row$dose_day), 0, row$dose_day),
    stop = row$end_day,
    event = ifelse(!is.na(row$death_day) && row$death_day == row$end_day, 1, 0),
    vax = ifelse(is.na(row$dose_day), 0, 1)
  )
  
  rbind(pre, post, fill = TRUE)
}))

# ========================================================
# 3. Fit Cox model with time-dependent vaccination
# ========================================================
cox_fit <- coxph(Surv(start, stop, event) ~ vax + cluster(id), data = df_surv)
summary(cox_fit)

# ========================================================
# 4. Predict survival curves (Observed / Never / Always)
# ========================================================
fit_never <- survfit(cox_fit, newdata = data.frame(vax = 0))
fit_always <- survfit(cox_fit, newdata = data.frame(vax = 1))

# Observed Kaplan-Meier (empirical)
km_fit <- survfit(Surv(stop, event) ~ 1, data = df_surv)

# ========================================================
# 5. Prepare for plotting
# ========================================================
plot_df <- rbind(
  data.table(time = km_fit$time, surv = km_fit$surv, regime = "Observed"),
  data.table(time = fit_never$time, surv = fit_never$surv, regime = "Never Vaccinated"),
  data.table(time = fit_always$time, surv = fit_always$surv, regime = "Always Vaccinated")
)

plot_df[, risk := 1 - surv]  # cumulative risk

# ========================================================
# 6. Plot cumulative risk curves
# ========================================================
ggplot(plot_df, aes(x = time, y = risk, color = regime, linetype = regime)) +
  geom_line(size = 1) +
  scale_color_manual(values = c("black", "blue", "red")) +
  labs(
    title = "Cumulative Risk of Death for AG70 (No IV)",
    x = "Follow-up Day",
    y = "Cumulative Risk",
    color = "Regime",
    linetype = "Regime"
  ) +
  theme_minimal() +
  theme(text = element_text(size = 14))
