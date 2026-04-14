## Evaluation Results Data Format

This pipeline outputs evaluation results using **Apache Parquet (`.parquet`)**. 

### 1. Predictions Table (`predictions.parquet`)
This table stores the spot-level clustering/classification results for downstream metric calculations (e.g., ARI, spARI). 

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `cohort` | String | The dataset identifier. | `"DLPFC_10x"` |
| `sampleID` | String | The specific tissue or sample. | `"151507"` |
| `spotID` | String | The unique barcode/name of the spot. | `"AAACAACGAATAGTTC-1"` |
| `method` | String | The multi-sample method evaluated. | `"JADE"` |
| `z` | String | Pathologist annotation or ground truth. | `"L1"` |
| `z_pred` | String | The method's output cluster/label. | `"1"` |

### 2. Performance Metrics Table (`performance.parquet`)

| Column Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `cohort` | String | The dataset identifier. | `"DLPFC_10x"` |
| `method` | String | The multi-sample method evaluated. | `"JADE"` |
| `runtime` | Float | Total wall-clock time in seconds. | `345.67` |
| `memory` | Float | Peak RAM usage in Megabytes. | `8042.5` |

---

### Loading the Data

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

---

## Cohort Spatial Domains

For multi-sample methods that require the number of spatial domains to be specified *a priori*, please refer to the following expected domain counts for each cohort:

| Cohort | Expected Spatial Domains |
| :--- | :--- |
| `BC_10x` | 6 |
| `BC_HER2+_ST` | 4 |
| `BC_HP_10x` | 6 |
| `BC_TNBC_ST` | 4 |
| `CRC_CMS_10x` | 6 |
| `DLPFC_10x` | 7 |
| `KC_TLS_10x` | 3 |
| `LC_TLS_10x` | 3 |
| `MOB_ST` | 3 |
| `RCC_TLS_10x` | 3 |
