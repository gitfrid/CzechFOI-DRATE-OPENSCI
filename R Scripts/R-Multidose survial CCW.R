# Load required packages
library(data.table)
library(survival)

# --- 1. Load data ---
# df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/Vesely_106_202403141131_AG70.csv")
df <- fread("C:/CzechFOI-DRATE-REALBIAS/Terra/FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")

# Convert all column names to lowercase
setnames(df, tolower(names(df)))

# Check for required columns
required_cols <- c("datumumrti", paste0("datum_", 1:7))
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop("Missing columns: ", paste(missing_cols, collapse = ", "))
}

# Convert relevant columns to Date format
date_cols <- c("datumumrti", paste0("datum_", 1:7))
df[, (date_cols) := lapply(.SD, function(x) as.Date(x, format = "%Y-%m-%d")), .SDcols = date_cols]

# Define baseline date and conversion function
START_DATE <- as.Date("2020-01-01")
to_day <- function(x) as.integer(x - START_DATE)

# Add numeric day columns
df[, death_day := to_day(datumumrti)]
df[, dose_days := do.call(pmin, c(.SD, na.rm = TRUE)), .SDcols = paste0("datum_", 1:7)]
df[, dose_day := ifelse(is.na(dose_days), NA_integer_, to_day(dose_days))]

# Define follow-up time
max_observed_day <- max(c(df$death_day, df$dose_day), na.rm = TRUE)
MAX_FOLLOW_UP <- min(max_observed_day, 1095)
df[, end_day := ifelse(is.na(death_day), MAX_FOLLOW_UP, death_day)]

# --- 2. Long format intervals ---
dt_list <- lapply(1:nrow(df), function(i) {
  row <- df[i]
  if (!is.na(row$dose_day) && row$dose_day > row$end_day) return(NULL)

  intervals <- list()

  if (is.na(row$dose_day) || row$dose_day == row$end_day) {
    if (row$end_day > 0) {
      intervals[[1]] <- data.table(
        id = i,
        tstart = 0,
        tstop = row$end_day,
        event = as.integer(row$death_day == row$end_day),
        vaccinated = 0
      )
    }
  } else {
    if (row$dose_day > 0) {
      intervals[[1]] <- data.table(
        id = i,
        tstart = 0,
        tstop = row$dose_day,
        event = 0,
        vaccinated = 0
      )
    }
    if (row$end_day > row$dose_day) {
      intervals[[2]] <- data.table(
        id = i,
        tstart = row$dose_day,
        tstop = row$end_day,
        event = as.integer(row$death_day == row$end_day),
        vaccinated = 1
      )
    }
  }
  rbindlist(intervals)
})

dt_long <- rbindlist(dt_list, fill = TRUE)
dt_long <- dt_long[tstart < tstop]
dt_long <- dt_long[complete.cases(dt_long)]

# --- 4. Cox Model (Time-varying Vaccination) ---
cox_model <- coxph(Surv(tstart, tstop, event) ~ vaccinated, data = dt_long)
summary(cox_model)

# Proportional hazards test
cox.zph(cox_model)

# Calculate Vaccine Efficacy
hr <- exp(coef(cox_model)["vaccinated"])
ve <- (1 - hr) * 100
cat("\nVaccine Efficacy (Time-dependent):", round(ve, 2), "%\n")

# --- 5. Survival Plot ---
newdata <- data.frame(vaccinated = c(0, 1))
sv <- survfit(cox_model, newdata = newdata)

pdf("C:/CzechFOI-DRATE-REALBIAS/Results/survival_plot_time_dep.pdf")
plot(sv, col = c("blue", "red"), lty = 1:2,
     xlab = "Days since baseline", ylab = "Survival probability",
     main = "Survival Curves by Time-varying Vaccination Status")
legend("topright", legend = c("Unvaccinated", "Vaccinated"), col = c("blue", "red"), lty = 1:2)
dev.off()