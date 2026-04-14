# MOCHA

## Evaluation Results Data Format

To ensure seamless interoperability between Python and R, and to handle large-scale spatial transcriptomics data efficiently, this pipeline outputs evaluation results using **Apache Parquet (`.parquet`)**. 

Parquet is a heavily compressed, columnar storage format that preserves data types and reads in fractions of a second in both languages. The results are split into two clean, relational "tidy" tables:

### 1. Predictions Table (`predictions.parquet`)
This table stores the spot-level clustering/classification results for downstream metric calculations (e.g., ARI, NMI, accuracy). 

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `cohort` | String | The dataset identifier. | `"DLPFC"` |
| `tissue` | String | The specific tissue or sample. | `"Sample_151673"` |
| `spot_id` | String | The unique barcode/name of the spot. | `"AAACAAGTATCTCCCA-1"` |
| `method` | String | The multi-sample method evaluated. | `"SpaGCN"` |
| `true_label` | String | Pathologist annotation or ground truth. | `"Layer_1"` |
| `predicted_label` | String | The method's output cluster/label. | `"Cluster_3"` |

*(Note: `spot_id` combined with `tissue` forms a globally unique identifier for each spot across cohorts).*

### 2. Performance Metrics Table (`performance.parquet`)
Because multi-sample methods process multiple tissues simultaneously, hardware metrics are recorded per *run* rather than per spot.

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `cohort` | String | The dataset identifier. | `"DLPFC"` |
| `method` | String | The multi-sample method evaluated. | `"SpaGCN"` |
| `tissues_processed`| String | Comma-separated list of tissues in this run. | `"Sample_1, Sample_2"` |
| `runtime_seconds` | Float | Total wall-clock time in seconds. | `345.67` |
| `memory_peak_mb` | Float | Peak RAM usage in Megabytes. | `8042.5` |

---

### Loading the Data

You can easily load these results in your preferred language without needing complex spatial container objects to evaluate performance.

**In Python (using `pandas`):**
```python
import pandas as pd

# Load predictions and performance
predictions = pd.read_parquet("results/predictions.parquet")
performance = pd.read_parquet("results/performance.parquet")

# Example: View the true vs predicted labels for a specific method
print(predictions[predictions['method'] == 'SpaGCN'].head())
```

**In R (using `arrow` and `dplyr`):**
```R
# Install if needed: install.packages("arrow")
library(arrow)
library(dplyr)

# Load predictions and performance
predictions <- read_parquet("results/predictions.parquet")
performance <- read_parquet("results/performance.parquet")

# Example: Quick accuracy check per tissue
accuracy_summary <- predictions %>%
  group_by(cohort, tissue, method) %>%
  summarize(accuracy = sum(true_label == predicted_label) / n(), .groups = "drop")

print(accuracy_summary)
```
