# Load libraries
library(NetworkInference)
library(dplyr)
library(lubridate)

# Read your data (CHANGE THIS PATH)
cat("Loading data...\n")
df <- read.delim("your_file.tsv", sep = "\t")
cat("Loaded", nrow(df), "rows\n")

# Sample 2000 rows
set.seed(123)
df <- df[sample(nrow(df), 50000), ]
cat("Sampled 2000 rows\n")

# Convert timestamp
cat("Processing timestamps...\n")
df$TIMESTAMP <- mdy_hms(df$TIMESTAMP)

# Create time-based cascades
cat("Creating cascades...\n")
cascade_data <- df %>%
  mutate(time_window = floor_date(TIMESTAMP, "30 minutes")) %>%
  group_by(time_window) %>%
  mutate(cascade_id = cur_group_id()) %>%
  ungroup() %>%
  select(node = SOURCE_SUBREDDIT, infection_time = TIMESTAMP, cascade_id) %>%
  bind_rows(
    df %>%
      mutate(time_window = floor_date(TIMESTAMP, "30 minutes")) %>%
      group_by(time_window) %>%
      mutate(cascade_id = cur_group_id()) %>%
      ungroup() %>%
      select(node = TARGET_SUBREDDIT, infection_time = TIMESTAMP, cascade_id)
  )
cat("Created", n_distinct(cascade_data$cascade_id), "cascades\n")

# Convert to NetInf format
cat("Converting to NetInf format...\n")
cascades <- as_cascade_long(cascade_data, 
                            cascade_node_name = "node", 
                            event_time = "infection_time", 
                            cascade_id = "cascade_id")

# Run NetInf with progress
cat("Running NetInf algorithm (this may take a few minutes)...\n")
cat("Status: Initializing...\n")

results <- netinf(cascades, 
                  trans_mod = "exponential", 
                  p_value_cutoff = 0.05,
                  quiet = FALSE)  # FALSE shows progress

cat("Done! Inferred", nrow(results), "edges\n")

# Show results
print(head(results, 100))

# Save results
write.csv(results, "netinf_results.csv", row.names = FALSE)
cat("Results saved to netinf_results.csv\n")



library(igraph)

# Create graph from results
g <- graph_from_data_frame(results, directed = TRUE)

# simple plot
plot(g, 
     vertex.size = 6,
     vertex.label.cex = 0.8,
     edge.arrow.size = 0.3,
     main = "Inferred Information Diffusion Network")

