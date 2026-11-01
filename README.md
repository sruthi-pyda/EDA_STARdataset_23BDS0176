# Exploratory Data Analysis – Star Classification Dataset
### BCSE331P | VIT, School of Computer Science and Engineering
**Student:** Sruthi Pyda &nbsp;|&nbsp; **Reg No:** 23BDS0176 &nbsp;|&nbsp; **Semester:** Fall 2026-27

---

## Dataset

| Property | Value |
|---|---|
| File | `23BDS0176_EDADATASET.csv` |
| Records | 240 rows × 7 columns |
| Domain | Stellar Astrophysics |

**Features:**

| Column | Description |
|---|---|
| Temperature (K) | Surface temperature in Kelvin |
| Luminosity (L/Lo) | Luminosity relative to the Sun |
| Radius (R/Ro) | Radius relative to the Sun |
| Absolute magnitude (Mv) | Intrinsic brightness |
| Star type | 0–5 (Brown Dwarf → Hypergiant) |
| Star color | Observed color (Blue, Red, White, …) |
| Spectral Class | O, B, A, F, G, K, M |

**Star types mapped:**
`0` Brown Dwarf · `1` Red Dwarf · `2` White Dwarf · `3` Main Sequence · `4` Supergiant · `5` Hypergiant

---

## Repository Structure

```
.
├── 23BDS0176_EDADATASET.csv          # Dataset
├── EDA_Project_23BDS0176.ipynb       # Main project notebook (149 cells)
├── demo_23BDS0176.py                 # Standalone demo script — saves 28 plots
├── dashboard_23BDS0176.py            # Interactive Streamlit dashboard
├── vit_logo.png                      # VIT logo (used in dashboard sidebar)
├── Exploratory Data Analysis Lab QP 1.pdf   # Assessment 1 – Merging & Transformations
├── Exploratory Data Analysis Lab QP 2.pdf   # Assessment 2
├── Exploratory Data Analysis Lab QP 3.pdf   # Assessment 3
├── Exploratory Data Analysis Lab QP 4.pdf   # Assessment 4
├── Exploratory Data Analysis Lab QP 5.pdf   # Assessment 5
├── Exploratory Data Analysis Lab QP 6.pdf   # Assessment 6
├── Exploratory Data Analysis Lab QP 7.pdf   # Assessment 7
├── Exploratory Data Analysis Lab QP 8.pdf   # Assessment 8
└── Exploratory Data Analysis Lab QP 9.pdf   # Assessment 9 – PCA
```

---

## Sections Covered

| # | Topic | Key Techniques |
|---|---|---|
| 1 | Dataset Loading & Exploration | head/tail, dtypes, describe, missing value check |
| 2 | Data Cleaning & Preprocessing | Mean/Median/Mode imputation, ffill/bfill, type conversion, derived features |
| 3 | Descriptive Statistics | Min, Max, Range, Variance, Std Dev, IQR, Skewness, Kurtosis |
| 4 | Univariate Analysis | Boxplots, Histograms, Violin, Bar, Pie, Lollipop, Density, Faceted Histograms |
| 5 | Bivariate Analysis | Contingency table, Stacked/Grouped bar, H-R Scatter, Heatmap, Correlogram |
| 6 | Multivariate Analysis | 3/4-variable scatter, Bubble plot, Facet grid, Parallel Coordinates, Pairs plot |
| 7 | Statistical Analysis | Pearson & Spearman correlation, Regression line, Error-bar charts |
| 8 | Time Series Analysis | Decomposition (multiplicative), ACF/PACF, SARIMA(2,1,2)(1,1,1,12) forecast |
| 9 | Clustering Analysis | K-Means (Elbow method, K=2 & K=6), Hierarchical (Ward's dendrogram) |
| 10 | PCA | Scree plot, Biplot with loadings, pca_scores table, PC1 vs PC2 scatter |

---

## How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels streamlit plotly
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook EDA_Project_23BDS0176.ipynb
```

### 3. Run the demo script (saves 28 plots to `demo_output/`)
```bash
python demo_23BDS0176.py
```

### 4. Launch the interactive Streamlit dashboard
```bash
streamlit run dashboard_23BDS0176.py
```
Then open **http://localhost:8501** in your browser.

---

## Key Findings

- **PC1 captures 60.3%** of variance; PC1 + PC2 together explain **83.8%** — the dataset is well-summarised in 2D.
- **K-Means with K=6** (elbow recommendation) aligns closely with the 6 true star types.
- The **Hertzsprung–Russell diagram** clearly separates Brown/Red Dwarfs, Main Sequence, and Giants.
- Temperature has a **positive Pearson correlation (r = 0.44)** with Log Luminosity, statistically significant (p < 1e-12).
- Luminosity and Radius are **right-skewed** (skew > 1.9) — log transformation was applied for analysis.

---

*Generated as part of BCSE331P – Exploratory Data Analysis, VIT University*
