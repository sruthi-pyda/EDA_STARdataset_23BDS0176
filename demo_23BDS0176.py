"""
BCSE331P – Exploratory Data Analysis
VIT, School of Computer Science and Engineering
Student : 23BDS0176
Dataset : Star Classification (240 records × 7 features)

Run:  python demo_23BDS0176.py
All plots are saved to  demo_output/<section>/
"""

import warnings; warnings.filterwarnings('ignore')
import os, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive – saves to files
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── output directory setup ────────────────────────────────────────────────────
ROOT = "demo_output"
SECTIONS = [
    "01_loading", "02_cleaning", "03_descriptive",
    "04_univariate", "05_bivariate", "06_multivariate",
    "07_statistical", "08_timeseries", "09_clustering", "10_pca"
]
for s in SECTIONS:
    os.makedirs(os.path.join(ROOT, s), exist_ok=True)

def save(fig, section, name):
    path = os.path.join(ROOT, section, name)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    [saved] {path}")

def header(n, title):
    bar = "=" * 60
    print(f"\n{bar}\n  SECTION {n}: {title}\n{bar}")

# ── palette / style ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10")
STAR_MAP   = {0:'Brown Dwarf',1:'Red Dwarf',2:'White Dwarf',
              3:'Main Sequence',4:'Supergiant',5:'Hypergiant'}
STAR_COLORS = {
    'Brown Dwarf':'saddlebrown', 'Red Dwarf':'tomato',
    'White Dwarf':'lightblue',   'Main Sequence':'gold',
    'Supergiant':'orchid',        'Hypergiant':'limegreen'
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET LOADING & PRELIMINARY EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
header(1, "Dataset Loading & Preliminary Exploration")

df = pd.read_csv("23BDS0176_EDADATASET.csv")
df.replace('?', np.nan, inplace=True)
df['Luminosity(L/Lo)'] = pd.to_numeric(df['Luminosity(L/Lo)'], errors='coerce')

print(f"  Shape            : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Columns          : {df.columns.tolist()}")
print(f"\n  First 5 records:\n{df.head(5).to_string()}")
print(f"\n  Last 5 records:\n{df.tail(5).to_string()}")
print(f"\n  Data types:\n{df.dtypes.to_string()}")
print(f"\n  Missing values:\n{df.isnull().sum().to_string()}")
print(f"\n  Descriptive stats:\n{df.describe().round(3).to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
header(2, "Data Cleaning & Preprocessing")

df_clean = df.copy()

# Missing value imputation
df_clean['Luminosity(L/Lo)'].fillna(df_clean['Luminosity(L/Lo)'].mean(), inplace=True)
df_clean['Radius(R/Ro)'].fillna(df_clean['Radius(R/Ro)'].median(), inplace=True)
df_clean['Star color'].fillna(df_clean['Star color'].mode()[0], inplace=True)
df_clean.ffill(inplace=True)
df_clean.bfill(inplace=True)
print(f"  Missing values after imputation:\n{df_clean.isnull().sum().to_string()}")

# Duplicates
dups = df_clean.duplicated().sum()
df_clean.drop_duplicates(inplace=True)
print(f"  Duplicate rows removed: {dups}")

# Type conversion & normalisation
df_clean['Star color'] = df_clean['Star color'].astype(str).str.strip().str.title()
df_clean['Spectral Class'] = df_clean['Spectral Class'].astype(str).str.strip().str.upper()

# Rename & derived features
df_clean.rename(columns={'Absolute magnitude(Mv)': 'Abs_Magnitude'}, inplace=True)
df_clean['Log_Luminosity'] = np.log10(df_clean['Luminosity(L/Lo)'].astype(float) + 1e-9)
df_clean['Log_Radius']     = np.log10(df_clean['Radius(R/Ro)'].astype(float)     + 1e-9)
df_clean['Star_Label']     = df_clean['Star type'].astype(int).map(STAR_MAP)

print(f"  Cleaned dataset shape: {df_clean.shape}")
print(f"  New columns: Log_Luminosity, Log_Radius, Star_Label")

# Color simplification helper (used in bivariate section)
def simplify_color(c):
    c = str(c).lower()
    for k in ['blue','red','white','yellow','orange']:
        if k in c: return k.title()
    return 'Other'
df_clean['Color_Group'] = df_clean['Star color'].apply(simplify_color)

# ══════════════════════════════════════════════════════════════════════════════
# 3. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
header(3, "Descriptive Statistics")

num_cols = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Abs_Magnitude']

print(f"\n  {'Column':<25} {'Min':>10} {'Max':>12} {'Range':>12} {'Var':>14} {'Std':>10} {'IQR':>10}")
print("  " + "-"*95)
for col in num_cols:
    s = df_clean[col].astype(float)
    print(f"  {col:<25} {s.min():>10.3g} {s.max():>12.3g} "
          f"{s.max()-s.min():>12.3g} {s.var():>14.3g} {s.std():>10.3g} "
          f"{s.quantile(.75)-s.quantile(.25):>10.3g}")

print(f"\n  {'Column':<25} {'Mean':>10} {'Median':>10} {'Skew':>8} {'Kurt':>8}")
print("  " + "-"*65)
for col in num_cols:
    s = df_clean[col].astype(float)
    print(f"  {col:<25} {s.mean():>10.3g} {s.median():>10.3g} "
          f"{skew(s.dropna()):>8.3f} {kurtosis(s.dropna()):>8.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
header(4, "Univariate Analysis")
sec = "04_univariate"

# 4a – Boxplots (outlier detection)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    df_clean[col].astype(float).plot.box(ax=ax, color='steelblue',
                                          patch_artist=True,
                                          boxprops=dict(facecolor='lightblue'))
    ax.set_title(col.split('(')[0].strip(), fontsize=10)
    ax.grid(alpha=0.3)
fig.suptitle('Boxplots – Outlier Detection', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "4a_boxplots.png")

# 4b – IQR outlier count
print("\n  IQR Outlier counts:")
for col in ['Temperature (K)', 'Abs_Magnitude']:
    s = df_clean[col].astype(float)
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    out = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
    print(f"    {col}: {out} outliers")

# 4c – Histograms with density
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ['Temperature (K)', 'Abs_Magnitude']):
    s = df_clean[col].astype(float)
    ax.hist(s, bins=25, density=True, color='steelblue', alpha=0.6, edgecolor='white')
    xr = np.linspace(s.min(), s.max(), 300)
    ax.plot(xr, stats.norm.pdf(xr, s.mean(), s.std()), 'r-', lw=2, label='Normal curve')
    ax.set_title(f'Histogram – {col.split("(")[0].strip()}')
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle('Histograms with Density Curves', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "4b_histograms.png")

# 4d – Violin plot
fig, ax = plt.subplots(figsize=(12, 5))
order = list(STAR_MAP.values())
sns.violinplot(data=df_clean, x='Star_Label', y='Temperature (K)',
               order=order, palette=STAR_COLORS, ax=ax)
ax.set_title('Violin Plot – Temperature by Star Type', fontsize=13, fontweight='bold')
ax.set_xlabel('Star Type'); ax.set_ylabel('Temperature (K)')
plt.xticks(rotation=15); plt.tight_layout()
save(fig, sec, "4c_violin_temp.png")

# 4e – Bar chart + Pie chart (Spectral Class / Star Type)
sc = df_clean['Spectral Class'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.plot.bar(ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Spectral Class – Frequency'); axes[0].set_xlabel('Class')
axes[0].tick_params(axis='x', rotation=0)
st_counts = df_clean['Star_Label'].value_counts()
axes[1].pie(st_counts, labels=st_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette('tab10', len(st_counts)))
axes[1].set_title('Star Type Distribution (Pie)')
fig.suptitle('Frequency Charts', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "4d_bar_pie.png")

# 4f – Density plot by Star Type
fig, ax = plt.subplots(figsize=(11, 5))
for lbl, grp in df_clean.groupby('Star_Label'):
    grp['Temperature (K)'].astype(float).plot.kde(ax=ax, label=lbl,
                                                    color=STAR_COLORS.get(lbl))
ax.set_title('Density Plot – Temperature by Star Type', fontsize=13, fontweight='bold')
ax.set_xlabel('Temperature (K)'); ax.legend(fontsize=8)
ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "4e_density_temp.png")

# 4g – Faceted histograms
g = sns.FacetGrid(df_clean, col='Star_Label', col_wrap=3, height=3, sharey=False)
g.map(plt.hist, 'Temperature (K)', bins=15, color='steelblue', edgecolor='white')
g.set_titles(col_template="{col_name}"); g.set_axis_labels("Temperature (K)", "Count")
g.figure.suptitle('Faceted Histograms – Temperature Across Star Types',
                   fontsize=12, fontweight='bold', y=1.02)
save(g.figure, sec, "4f_facet_hist.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
header(5, "Bivariate Analysis")
sec = "05_bivariate"

# 5a – Contingency table
ct = pd.crosstab(df_clean['Star_Label'], df_clean['Color_Group'])
print(f"\n  Contingency Table (Star Type × Color Group):\n{ct.to_string()}")

# 5b – Stacked & grouped bar
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ct.plot(kind='bar', stacked=True,  ax=axes[0], colormap='tab10', edgecolor='white')
ct.plot(kind='bar', stacked=False, ax=axes[1], colormap='tab10', edgecolor='white')
axes[0].set_title('Stacked Bar – Star Type vs Color')
axes[1].set_title('Grouped Bar – Star Type vs Color')
for ax in axes: ax.tick_params(axis='x', rotation=30); ax.legend(fontsize=7)
fig.suptitle('Star Type × Star Color', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "5a_stacked_grouped_bar.png")

# 5c – Scatter Temperature vs Log Luminosity
fig, ax = plt.subplots(figsize=(10, 6))
for lbl, grp in df_clean.groupby('Star_Label'):
    ax.scatter(grp['Temperature (K)'], grp['Log_Luminosity'],
               label=lbl, alpha=0.7, s=40, color=STAR_COLORS.get(lbl))
ax.set_title('Scatter – Temperature vs Log Luminosity\n(Hertzsprung–Russell Diagram)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Log Luminosity')
ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, sec, "5b_scatter_HR.png")

# 5d – Correlation heatmap
num_h = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)',
         'Abs_Magnitude', 'Log_Luminosity', 'Log_Radius']
corr = df_clean[num_h].astype(float).corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=.5, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "5c_heatmap.png")

# 5e – Box + Violin bivariate
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.boxplot(data=df_clean, x='Star_Label', y='Temperature (K)',
            order=order, palette=STAR_COLORS, ax=axes[0])
sns.violinplot(data=df_clean, x='Star_Label', y='Abs_Magnitude',
               order=order, palette=STAR_COLORS, ax=axes[1])
axes[0].set_title('Box – Temperature by Star Type')
axes[1].set_title('Violin – Abs Magnitude by Star Type')
for ax in axes: ax.tick_params(axis='x', rotation=20)
fig.suptitle('Temperature & Magnitude by Star Type', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "5d_box_violin.png")

# 5f – Correlogram
fig = sns.pairplot(df_clean[['Temperature (K)', 'Log_Luminosity',
                              'Log_Radius', 'Abs_Magnitude']],
                   diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20})
fig.figure.suptitle('Correlogram – Scatter Plot Matrix', y=1.02,
                    fontsize=13, fontweight='bold')
save(fig.figure, sec, "5e_correlogram.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. MULTIVARIATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
header(6, "Multivariate Analysis")
sec = "06_multivariate"

# 6a – 3-variable scatter (X, Y, Color)
fig, ax = plt.subplots(figsize=(10, 6))
sc6 = ax.scatter(df_clean['Temperature (K)'],
                 df_clean['Log_Luminosity'],
                 c=df_clean['Star type'].astype(int),
                 cmap='tab10', alpha=0.7, s=50)
plt.colorbar(sc6, ax=ax, label='Star Type (0–5)')
ax.set_title('3-Variable Scatter: Temp × Log-Lum (color = Star Type)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Log Luminosity')
ax.invert_xaxis(); ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, sec, "6a_3var_scatter.png")

# 6b – Bubble plot (X, Y, size=Log Radius)
bsize = (df_clean['Log_Radius'] - df_clean['Log_Radius'].min() + 0.5) * 60
fig, ax = plt.subplots(figsize=(10, 6))
sc6b = ax.scatter(df_clean['Temperature (K)'],
                  df_clean['Abs_Magnitude'],
                  s=bsize,
                  c=df_clean['Star type'].astype(int),
                  cmap='tab10', alpha=0.6, edgecolors='white', linewidths=0.4)
plt.colorbar(sc6b, ax=ax, label='Star Type')
ax.set_title('Bubble Plot: Temp × Abs-Mag (size = Log Radius)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Abs Magnitude')
ax.invert_xaxis(); ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, sec, "6b_bubble.png")

# 6c – Facet wrap scatter by Star Type
g6 = sns.FacetGrid(df_clean, col='Star_Label', col_wrap=3, height=3)
g6.map(plt.scatter, 'Temperature (K)', 'Log_Luminosity', alpha=0.6, s=20, color='steelblue')
g6.set_titles("{col_name}"); g6.set_axis_labels("Temp (K)", "Log Lum")
g6.figure.suptitle('Facet Wrap – Scatter by Star Type', fontsize=12,
                    fontweight='bold', y=1.02)
save(g6.figure, sec, "6c_facet_scatter.png")

# 6d – Parallel coordinates
from pandas.plotting import parallel_coordinates
cols_pc = ['Temperature (K)', 'Log_Luminosity', 'Log_Radius', 'Abs_Magnitude', 'Star_Label']
pc_df = df_clean[cols_pc].copy()
fig, ax = plt.subplots(figsize=(12, 6))
parallel_coordinates(pc_df, 'Star_Label',
                     color=[STAR_COLORS[l] for l in pc_df['Star_Label'].unique()],
                     alpha=0.3, ax=ax)
ax.set_title('Parallel Coordinates Plot', fontsize=13, fontweight='bold')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, sec, "6d_parallel_coords.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
header(7, "Statistical Analysis")
sec = "07_statistical"

x = df_clean['Temperature (K)'].astype(float)
y = df_clean['Log_Luminosity'].astype(float)
r_p, p_p = pearsonr(x, y)
r_s, p_s = spearmanr(x, y)
print(f"\n  Pearson  r = {r_p:.4f}  (p = {p_p:.3e})")
print(f"  Spearman r = {r_s:.4f}  (p = {p_s:.3e})")

# Scatter with regression line
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.5, s=30, color='steelblue')
m, b = np.polyfit(x, y, 1)
xline = np.linspace(x.min(), x.max(), 200)
ax.plot(xline, m*xline + b, 'r-', lw=2,
        label=f'y = {m:.2e}x + {b:.2f}')
ax.set_title(f'Temperature vs Log Luminosity\n'
             f'Pearson r = {r_p:.3f} | Spearman r = {r_s:.3f}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Log Luminosity')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
save(fig, sec, "7a_regression_scatter.png")

# Grouped statistics
grp = df_clean.groupby('Star_Label')['Temperature (K)'].agg(['mean','median','std']).round(2)
print(f"\n  Grouped Temperature Statistics:\n{grp.to_string()}")

# Faceted box + grouped bar with error bars
df_sub = df_clean[df_clean['Color_Group'].isin(['Red','Blue','White'])]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.boxplot(data=df_sub, x='Star_Label', y='Temperature (K)',
            hue='Color_Group', palette={'Red':'tomato','Blue':'steelblue','White':'grey'},
            ax=axes[0])
axes[0].set_title('Box: Temp by Star Type × Color Group')
axes[0].tick_params(axis='x', rotation=20); axes[0].legend(fontsize=8)

err_df = (df_sub.groupby(['Star_Label','Color_Group'])['Temperature (K)']
          .agg(['mean','std']).reset_index())
for color_grp, grp_data in err_df.groupby('Color_Group'):
    c_map = {'Red':'tomato','Blue':'steelblue','White':'grey'}
    axes[1].bar(grp_data['Star_Label'], grp_data['mean'],
                yerr=grp_data['std'], label=color_grp,
                color=c_map.get(color_grp, 'grey'),
                alpha=0.7, capsize=4, edgecolor='white')
axes[1].set_title('Bar with Error Bars: Avg Temp by Star & Color')
axes[1].tick_params(axis='x', rotation=20); axes[1].legend(fontsize=8)
axes[1].set_ylabel('Mean Temperature (K)')
fig.suptitle('Bivariate Statistical Plots', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "7b_facet_box_bar.png")

# Full correlation matrix
corr2 = df_clean[num_h].astype(float).corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr2, dtype=bool))
sns.heatmap(corr2, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=.5, ax=ax)
ax.set_title('Full Correlation Matrix (lower triangle)', fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "7c_full_corr_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TIME SERIES ANALYSIS  (AirPassengers – classic benchmark)
# ══════════════════════════════════════════════════════════════════════════════
header(8, "Time Series Analysis")
sec = "08_timeseries"

air_raw = [
    112,118,132,129,121,135,148,148,136,119,104,118,
    115,126,141,135,125,149,170,170,158,133,114,140,
    145,150,178,163,172,178,199,199,184,162,146,166,
    171,180,193,181,183,218,230,242,209,191,172,194,
    196,196,236,235,229,243,264,272,237,211,180,201,
    204,188,235,227,234,264,302,293,259,229,203,229,
    242,233,267,269,270,315,364,347,312,274,237,278,
    284,277,317,313,318,374,413,405,355,306,271,306,
    315,301,356,348,355,422,465,467,404,347,305,336,
    340,318,362,348,363,435,491,505,404,359,310,337,
    360,342,406,396,420,472,548,559,463,407,362,405,
    417,391,419,461,472,535,622,606,508,461,390,432,
]
ts = pd.Series(air_raw,
               index=pd.date_range('1949-01', periods=len(air_raw), freq='ME'),
               name='Passengers')

# 8a – Monthly trend
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(ts.index, ts.values, color='steelblue', lw=1.5)
ax.set_title('AirPassengers: Monthly Airline Passengers (1949–1960)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Passengers (thousands)')
ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "8a_ts_trend.png")

# 8b – Decomposition
dec = seasonal_decompose(ts, model='multiplicative', period=12)
fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for ax, data, title, c in zip(axes,
        [dec.observed, dec.trend, dec.seasonal, dec.resid],
        ['Observed', 'Trend', 'Seasonal', 'Residual'],
        ['steelblue','darkgreen','coral','grey']):
    data.plot(ax=ax, color=c); ax.set_title(title); ax.grid(alpha=0.3)
fig.suptitle('Time Series Decomposition (Multiplicative Model)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save(fig, sec, "8b_decomposition.png")

# 8c – ACF & PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts,  lags=40, ax=axes[0], color='steelblue')
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(ts, lags=20, ax=axes[1], color='coral', method='ywm')
axes[1].set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
save(fig, sec, "8c_acf_pacf.png")

# 8d – SARIMA fit + forecast
print("\n  Fitting SARIMA(2,1,2)(1,1,1,12) on log-transformed series …")
log_ts = np.log(ts)
fit = SARIMAX(log_ts, order=(2,1,2), seasonal_order=(1,1,1,12),
              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
print(fit.summary().tables[0].as_text())

fcast  = fit.get_forecast(steps=12)
f_mean = np.exp(fcast.predicted_mean)
f_ci   = np.exp(fcast.conf_int())
f_idx  = pd.date_range('1961-01', periods=12, freq='ME')

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(ts.index, ts.values, label='Actual', color='steelblue', lw=1.5)
ax.plot(f_idx, f_mean.values, '--', color='coral', lw=2, label='Forecast (1961)')
ax.fill_between(f_idx, f_ci.iloc[:,0].values, f_ci.iloc[:,1].values,
                color='coral', alpha=0.2, label='95 % CI')
ax.set_title('SARIMA Forecast – Next 12 Months', fontsize=13, fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Passengers (thousands)')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "8d_sarima_forecast.png")

# ══════════════════════════════════════════════════════════════════════════════
# 9. CLUSTERING ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
header(9, "Clustering Analysis")
sec = "09_clustering"

feat_cols = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Abs_Magnitude']
df_cl  = df_clean[feat_cols].astype(float).dropna()
scaler = StandardScaler()
X_sc   = scaler.fit_transform(df_cl)

# K-Means k=2
km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_sc)
print(f"\n  K=2 | Sizes: {np.bincount(km2.labels_)} | WCSS: {km2.inertia_:.2f}")

# Elbow plot
wcss = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc).inertia_
        for k in range(1, 11)]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1,11), wcss, 'o-', color='steelblue', lw=2, ms=8)
ax.set_title('Elbow Plot: WCSS vs Number of Clusters (K)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('K'); ax.set_ylabel('WCSS (Inertia)')
ax.set_xticks(range(1,11)); ax.grid(alpha=0.4)
plt.tight_layout()
save(fig, sec, "9a_elbow.png")

print("\n  WCSS by K:")
for k, w in enumerate(wcss, 1):
    print(f"    K={k}: {w:.2f}")

# Optimal k=6 vs true labels
km6  = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_sc)
pca2 = PCA(n_components=2); X2d = pca2.fit_transform(X_sc)
true_lbl = df_clean.loc[df_cl.index, 'Star type'].astype(int).values

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, lbl, title in zip(axes,
        [km6.labels_, true_lbl],
        ['K-Means Clusters (K=6)', 'True Star Types']):
    sc9 = ax.scatter(X2d[:,0], X2d[:,1], c=lbl, cmap='tab10', alpha=0.7, s=40)
    plt.colorbar(sc9, ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
fig.suptitle('K-Means vs True Star Types (PCA projection)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save(fig, sec, "9b_kmeans_vs_true.png")

# Hierarchical – dendrogram (sample of 80)
rng   = np.random.RandomState(42)
idx80 = rng.choice(len(X_sc), 80, replace=False)
Z     = linkage(X_sc[idx80], method='ward')

fig, ax = plt.subplots(figsize=(16, 6))
dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7, color_threshold=10)
ax.set_title("Hierarchical Clustering Dendrogram (Ward's Method, n=80 sample)",
             fontsize=13, fontweight='bold')
ax.set_xlabel('Sample Index'); ax.set_ylabel('Euclidean Distance')
plt.tight_layout()
save(fig, sec, "9c_dendrogram.png")

labels3 = fcluster(Z, t=3, criterion='maxclust')
print("\n  Hierarchical Clustering – 3 Clusters:")
for c, n in enumerate(np.bincount(labels3)[1:], 1):
    print(f"    Cluster {c}: {n} samples")

# ══════════════════════════════════════════════════════════════════════════════
# 10. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ══════════════════════════════════════════════════════════════════════════════
header(10, "Principal Component Analysis (PCA)")
sec = "10_pca"

pca_cols   = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Abs_Magnitude']
df_pca_raw = df_clean[pca_cols].astype(float).dropna()
print(f"\n  PCA input: {df_pca_raw.shape[0]} rows × {df_pca_raw.shape[1]} columns")

scaler_p   = StandardScaler()
data_scaled = scaler_p.fit_transform(df_pca_raw)

pca_full    = PCA()
pca_full.fit(data_scaled)
scores_full = pca_full.transform(data_scaled)

exp_var = pca_full.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

print(f"\n  {'PC':<6}{'Eigenvalue':>12}{'% Var':>10}{'Cum %':>10}")
print("  " + "-"*40)
for i, (ev, v, c) in enumerate(zip(pca_full.explained_variance_, exp_var, cum_var)):
    print(f"  PC{i+1:<4}{ev:>12.4f}{v:>9.2f}%{c:>9.2f}%")

# 10a – Scree plot
n_pc = len(exp_var)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(1, n_pc+1), exp_var, color='steelblue', edgecolor='white', alpha=0.8)
ax.plot(range(1, n_pc+1), exp_var,  'o-', color='darkblue', lw=2, label='% Variance')
ax.plot(range(1, n_pc+1), cum_var, 's--', color='coral',    lw=2, label='Cumulative %')
for bar, v in zip(bars, exp_var):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, 60)
ax.set_title('Scree Plot: % Variance Explained per Principal Component',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Principal Component'); ax.set_ylabel('Explained Variance (%)')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "10a_scree.png")

# 10b – Biplot
loadings    = pca_full.components_
star_colors = df_clean.loc[df_pca_raw.index, 'Star type'].astype(int).values

fig, ax = plt.subplots(figsize=(10, 8))
sc10 = ax.scatter(scores_full[:,0], scores_full[:,1],
                  c=star_colors, cmap='tab10', alpha=0.6, s=40)
plt.colorbar(sc10, ax=ax, label='Star Type (0–5)')
scale = 3.5
for i, feat in enumerate(pca_cols):
    ax.annotate('', xy=(loadings[0,i]*scale, loadings[1,i]*scale),
                xytext=(0,0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    label = feat.split('(')[0].strip()
    ax.text(loadings[0,i]*scale*1.18, loadings[1,i]*scale*1.18,
            label, color='red', fontsize=9, fontweight='bold',
            ha='center', va='center')
ax.axhline(0, color='grey', lw=0.8, ls='--')
ax.axvline(0, color='grey', lw=0.8, ls='--')
ax.set_title('PCA Biplot – Scores & Loadings', fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({exp_var[0]:.1f}% variance)')
ax.set_ylabel(f'PC2 ({exp_var[1]:.1f}% variance)')
ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "10b_biplot.png")

# 10c – PCA Scores table
pca_scores = pd.DataFrame(scores_full,
                           columns=[f'PC{i+1}' for i in range(n_pc)],
                           index=df_pca_raw.index)
print(f"\n  PCA Scores (pca_scores) – First 6 rows:\n{pca_scores.head(6).round(4).to_string()}")

# 10d – PC1 vs PC2 scatter (registration number in title)
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(pca_scores['PC1'], pca_scores['PC2'],
           color='blue', marker='o', alpha=0.6, s=50,
           edgecolors='white', linewidths=0.5)
ax.set_title('PC1 vs PC2 Scatter Plot – Registration No: 23BDS0176',
             fontsize=13, fontweight='bold')
ax.set_xlabel(f'PC1 ({exp_var[0]:.1f}% variance)')
ax.set_ylabel(f'PC2 ({exp_var[1]:.1f}% variance)')
ax.axhline(0, color='grey', lw=.8, ls='--')
ax.axvline(0, color='grey', lw=.8, ls='--')
ax.grid(alpha=0.3); plt.tight_layout()
save(fig, sec, "10d_pc1_pc2.png")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  DEMONSTRATION COMPLETE")
print("=" * 60)
print(f"  Student   : 23BDS0176")
print(f"  Course    : BCSE331P – Exploratory Data Analysis")
print(f"  Institute : VIT, School of Computer Science & Engineering")
print(f"  Dataset   : Star Classification  ({df_clean.shape[0]} rows × {df_clean.shape[1]} cols)")
print()
total = 0
for s in SECTIONS:
    folder = os.path.join(ROOT, s)
    files  = [f for f in os.listdir(folder) if f.endswith('.png')]
    total += len(files)
    print(f"  {s}/  →  {len(files)} plot(s)")
print(f"\n  Total plots saved : {total}")
print(f"  Output directory  : {os.path.abspath(ROOT)}/")
print("=" * 60)
