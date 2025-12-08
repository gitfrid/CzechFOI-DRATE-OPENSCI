library(data.table)
library(lubridate)

# === CONFIGURABLE CONSTANTS ===
INPUT_CSV <- "C:/CzechFOI-DRATE-NOBIAS/Terra/Vesely_106_202403141131_AG70.csv"
OUTPUT_FOLDER <- "C:/CzechFOI-DRATE-NOBIAS/Terra"
START_DATE <- ymd("2020-01-01")
DOSE_DATE_COLS <- paste0("Datum_", 1:7)
NEEDED_COLS <- c("Rok_narozeni", "DatumUmrti", DOSE_DATE_COLS)

RETRIES <- 10000
BASE_RNG_SEED <- 42
set.seed(BASE_RNG_SEED)

# === UTILITIES ===
to_day_number <- function(date_vec) {
  date_vec <- ymd(date_vec)
  as.integer(difftime(date_vec, START_DATE, units = "days"))
}

parse_dates <- function(dt) {
  for (col in c(DOSE_DATE_COLS, "DatumUmrti")) {
    dt[[col]] <- ymd(dt[[col]])
  }
  dt
}

estimate_death_rate <- function(dt) {
  deaths <- dt[["DatumUmrti"]]
  death_rate <- sum(!is.na(deaths)) / nrow(dt)
  death_rate <- min(max(death_rate, 1e-4), 0.999)
  death_rate
}

simulate_deaths <- function(dt, end_measure, death_rate) {
  dt <- copy(dt)
  dt[, DatumUmrti := as.Date(NA)]
  dt[, death_day := as.numeric(NA)]

  n <- nrow(dt)
  will_die <- runif(n) < death_rate
  death_days <- rep(NA_real_, n)
  death_days[will_die] <- sample(0:end_measure, sum(will_die), replace = TRUE)

  dt[, death_day := death_days]
  dt[will_die == TRUE, DatumUmrti := START_DATE + death_days[will_die]]

  dt
}

# === DOSE ASSIGNMENT ===
assign_doses_real_curve_random <- function(df_target, df_source, retries = RETRIES) {
  df_target <- copy(df_target)
  for (col in DOSE_DATE_COLS) {
    df_target[[col]] <- as.Date(NA)
  }

  dose_sets <- df_source[, ..DOSE_DATE_COLS]
  dose_sets <- dose_sets[rowSums(!is.na(dose_sets)) > 0]

  death_day_arr <- df_target$death_day
  assigned_flag <- integer(nrow(df_target))

  skip_count <- 0
  updates <- list()

  for (i in seq_len(nrow(dose_sets))) {
    dose_dates <- as.Date(unlist(dose_sets[i, ]))
    valid_dates <- dose_dates[!is.na(dose_dates)]
    if (length(valid_dates) == 0) next

    valid_days <- to_day_number(valid_dates)
    last_dose_day <- max(valid_days)

    eligible_indices <- which(assigned_flag == 0L)
    if (length(eligible_indices) == 0) {
      skip_count <- skip_count + 1
      next
    }

    eligible_indices <- sample(eligible_indices, length(eligible_indices))
    trial_pool <- head(eligible_indices, min(retries, length(eligible_indices)))

    selected_pos <- NA_integer_
    for (pos in trial_pool) {
      if (is.na(death_day_arr[pos]) || death_day_arr[pos] > last_dose_day) {
        selected_pos <- pos
        break
      }
    }

    if (!is.na(selected_pos)) {
      updates[[length(updates) + 1]] <- list(pos = selected_pos, doses = dose_dates)
      assigned_flag[selected_pos] <- 1L
    } else {
      skip_count <- skip_count + 1
    }
  }

  for (upd in updates) {
    pos <- upd$pos
    doses <- upd$doses
    for (j in seq_along(DOSE_DATE_COLS)) {
      if (!is.na(doses[j])) {
        df_target[[DOSE_DATE_COLS[j]]][pos] <- doses[j]
      }
    }
  }

  cat(sprintf("Assigned %d doses, Skipped %d\n", length(updates), skip_count))
  df_target
}

# === OUTPUT ===
format_and_save <- function(dt, out_path) {
  # Format dates as "YYYY-MM-DD" or empty
  for (col in c("DatumUmrti", DOSE_DATE_COLS)) {
    dt[[col]] <- ifelse(is.na(dt[[col]]), "", format(as.Date(dt[[col]]), "%Y-%m-%d"))
  }
  # Format death_day as x.0 if not NA, else empty
  dt[, death_day := ifelse(is.na(death_day), "", sprintf("%.1f", as.numeric(death_day)))]

  # Save only required columns in correct order
  fwrite(dt[, .(Rok_narozeni, DatumUmrti, Datum_1, Datum_2, Datum_3, Datum_4, Datum_5, Datum_6, Datum_7, death_day)],
         out_path, quote = FALSE, na = "")
  cat(sprintf("Saved: %s\n", out_path))
}

# === MAIN ===
run_all_cases <- function() {
  if (!dir.exists(OUTPUT_FOLDER)) dir.create(OUTPUT_FOLDER, recursive = TRUE)

  cat("📥 Loading data...\n")
  dt <- fread(INPUT_CSV, select = NEEDED_COLS, colClasses = "character")
  dt <- parse_dates(dt)

  max_death_day <- max(to_day_number(dt[["DatumUmrti"]]), na.rm = TRUE)
  END_MEASURE <- ifelse(is.finite(max_death_day), max_death_day, 1533)
  cat(sprintf("Measurement window (END_MEASURE): %d days\n", END_MEASURE))

  death_rate <- estimate_death_rate(dt)
  dt_sim_deaths <- simulate_deaths(dt, end_measure = END_MEASURE, death_rate = death_rate)

  dt_case3 <- assign_doses_real_curve_random(dt_sim_deaths, dt)
  output_file <- file.path(OUTPUT_FOLDER, "FG) case3_sim_deaths_sim_real_doses_with_constraint.csv")
  format_and_save(dt_case3, output_file)

  cat("✅ All cases processed and saved.\n")
}

# === RUN ===
run_all_cases()
