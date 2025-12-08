library(data.table)  # fast fread/fwrite and data handling
library(lubridate)   # for date parsing and formatting

# === CONFIGURATION ===
input_csv <- "C:/CzechFOI-DRATE-NOBIAS/Terra/Vesely_106_202403141131.csv"
output_csv <- "C:/CzechFOI-DRATE-NOBIAS/Terra/Vesely_106_202403141131_AG70.csv"
reference_year <- 2023
dose_date_cols <- paste0("Datum_", 1:7)
needed_cols <- c("Rok_narozeni", "DatumUmrti", dose_date_cols)

# === FUNCTIONS ===
parse_dates <- function(dt) {
  for (col in c(dose_date_cols, "DatumUmrti")) {
    dt[[col]] <- ymd(dt[[col]])
  }
  return(dt)
}

calculate_age <- function(dt) {
  dt <- copy(dt)  # deep copy to avoid shallow copy warning
  dt[, Age := reference_year - as.integer(Rok_narozeni)]
  return(dt)
}

format_dates_for_csv <- function(dt) {
  for (col in c(dose_date_cols, "DatumUmrti")) {
    dt[[col]] <- ifelse(is.na(dt[[col]]), "", format(dt[[col]], "%Y-%m-%d"))
  }
  return(dt)
}

# === MAIN ===
filter_and_save_age_70 <- function() {
  cat("📥 Loading input CSV...\n")
  dt <- fread(input_csv, select = needed_cols, colClasses = "character")

  cat("📆 Parsing dates and calculating age...\n")
  dt <- parse_dates(dt)
  dt <- calculate_age(dt)

  cat("🔎 Filtering to Age == 70...\n")
  dt_ag70 <- dt[Age == 70]

  cat(sprintf("💾 Saving %d rows to output...\n", nrow(dt_ag70)))
  dt_ag70 <- format_dates_for_csv(dt_ag70)

  fwrite(dt_ag70, output_csv, na = "", quote = FALSE)
  

  cat(sprintf("✅ Done. Saved to %s\n", output_csv))
}

# === RUN ===
filter_and_save_age_70()
