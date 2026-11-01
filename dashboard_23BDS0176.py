"""
BCSE331P – Exploratory Data Analysis  |  Interactive Faculty Dashboard
Student : 23BDS0176
Run     : streamlit run dashboard_23BDS0176.py
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDA Dashboard – 23BDS0176",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

STAR_MAP = {
    0: "Brown Dwarf", 1: "Red Dwarf", 2: "White Dwarf",
    3: "Main Sequence", 4: "Supergiant", 5: "Hypergiant",
}
STAR_COLORS = {
    "Brown Dwarf": "#8B4513", "Red Dwarf": "#FF4500",
    "White Dwarf": "#87CEEB", "Main Sequence": "#FFD700",
    "Supergiant": "#DA70D6", "Hypergiant": "#32CD32",
}
COLOR_SEQ = list(STAR_COLORS.values())

# ── data loader (cached) ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("23BDS0176_EDADATASET.csv")
    df.replace("?", np.nan, inplace=True)
    df["Luminosity(L/Lo)"] = pd.to_numeric(df["Luminosity(L/Lo)"], errors="coerce")

    dc = df.copy()
    dc["Luminosity(L/Lo)"].fillna(dc["Luminosity(L/Lo)"].mean(), inplace=True)
    dc["Radius(R/Ro)"].fillna(dc["Radius(R/Ro)"].median(), inplace=True)
    dc["Star color"].fillna(dc["Star color"].mode()[0], inplace=True)
    dc.ffill(inplace=True); dc.bfill(inplace=True)
    dc.drop_duplicates(inplace=True)
    dc["Star color"] = dc["Star color"].astype(str).str.strip().str.title()
    dc["Spectral Class"] = dc["Spectral Class"].astype(str).str.strip().str.upper()
    dc.rename(columns={"Absolute magnitude(Mv)": "Abs_Magnitude"}, inplace=True)
    dc["Log_Luminosity"] = np.log10(dc["Luminosity(L/Lo)"].astype(float) + 1e-9)
    dc["Log_Radius"]     = np.log10(dc["Radius(R/Ro)"].astype(float)     + 1e-9)
    dc["Star_Label"]     = dc["Star type"].astype(int).map(STAR_MAP)

    def simplify_color(c):
        c = str(c).lower()
        for k in ["blue","red","white","yellow","orange"]:
            if k in c: return k.title()
        return "Other"
    dc["Color_Group"] = dc["Star color"].apply(simplify_color)
    return df, dc

df_raw, df = load_data()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("vit_logo.png", width=180) if __import__("os").path.exists("vit_logo.png") else None
    st.markdown("## EDA Dashboard")
    st.markdown("**Student :** 23BDS0176  \n**Course :** BCSE331P  \n**Institute :** VIT")
    st.divider()
    section = st.radio(
        "Navigate to Section",
        [
            "1. Dataset Overview",
            "2. Data Cleaning",
            "3. Descriptive Statistics",
            "4. Univariate Analysis",
            "5. Bivariate Analysis",
            "6. Multivariate Analysis",
            "7. Statistical Analysis",
            "8. Time Series Analysis",
            "9. Clustering Analysis",
            "10. Principal Component Analysis",
        ],
    )
    st.divider()
    st.caption("Star Classification Dataset  \n240 rows × 7 features")

NUM_COLS = ["Temperature (K)", "Luminosity(L/Lo)", "Radius(R/Ro)", "Abs_Magnitude"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if section == "1. Dataset Overview":
    st.title("1. Dataset Loading & Preliminary Exploration")
    st.markdown("**Dataset:** Star Classification – physical properties of 240 stars across 6 types.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df_raw.shape[0])
    c2.metric("Columns", df_raw.shape[1])
    c3.metric("Missing Values", int(df_raw.isnull().sum().sum()))
    c4.metric("Star Types", df_raw["Star type"].nunique())

    st.subheader("Raw Dataset")
    st.dataframe(df_raw, use_container_width=True, height=300)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Types")
        st.dataframe(df_raw.dtypes.rename("dtype").reset_index().rename(columns={"index": "Column"}),
                     use_container_width=True, hide_index=True)
    with col2:
        st.subheader("Missing Values")
        mv = df_raw.isnull().sum().reset_index()
        mv.columns = ["Column", "Missing"]
        fig = px.bar(mv, x="Column", y="Missing", color="Missing",
                     color_continuous_scale="Reds", title="Missing Values per Column")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df_raw.describe().round(3), use_container_width=True)

    st.subheader("Star Type Distribution")
    vc = df_raw["Star type"].map(STAR_MAP).value_counts().reset_index()
    vc.columns = ["Star Type", "Count"]
    fig = px.pie(vc, names="Star Type", values="Count",
                 color="Star Type", color_discrete_map=STAR_COLORS,
                 hole=0.4, title="Star Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Data Cleaning
# ══════════════════════════════════════════════════════════════════════════════
elif section == "2. Data Cleaning":
    st.title("2. Data Cleaning & Preprocessing")

    st.subheader("Missing Value Imputation Strategy")
    impute_df = pd.DataFrame({
        "Column": ["Luminosity(L/Lo)", "Radius(R/Ro)", "Star color", "Spectral Class"],
        "Strategy": ["Mean imputation", "Median imputation", "Mode imputation", "Forward/Backward fill"],
        "Missing Before": [1, 0, 3, 3],
        "Missing After": [0, 0, 0, 0],
    })
    st.dataframe(impute_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Duplicates")
        st.info(f"Duplicate rows found: **0**  \nNo records removed.")
    with col2:
        st.subheader("Type Conversions")
        st.info("Star color → Title Case  \nSpectral Class → UPPER  \nLuminosity → numeric (coerce)")

    st.subheader("Derived Features Added")
    derived = pd.DataFrame({
        "Feature": ["Log_Luminosity", "Log_Radius", "Star_Label", "Color_Group"],
        "Formula / Source": [
            "log10(Luminosity + 1e-9)", "log10(Radius + 1e-9)",
            "Star type mapped to name", "Star color simplified to 5 groups"
        ],
        "Purpose": [
            "Compress wide luminosity range", "Compress wide radius range",
            "Human-readable labels for plots", "Reduce color categories"
        ],
    })
    st.dataframe(derived, use_container_width=True, hide_index=True)

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df, use_container_width=True, height=280)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Descriptive Statistics
# ══════════════════════════════════════════════════════════════════════════════
elif section == "3. Descriptive Statistics":
    st.title("3. Descriptive Statistics")

    col = st.selectbox("Select column", NUM_COLS)
    s = df[col].astype(float)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Min",    f"{s.min():.3g}")
    c2.metric("Max",    f"{s.max():.3g}")
    c3.metric("Mean",   f"{s.mean():.3g}")
    c4.metric("Median", f"{s.median():.3g}")
    c5.metric("Std Dev",f"{s.std():.3g}")
    c6.metric("IQR",    f"{s.quantile(.75)-s.quantile(.25):.3g}")

    c7,c8,c9 = st.columns(3)
    c7.metric("Variance", f"{s.var():.3g}")
    c8.metric("Skewness", f"{skew(s.dropna()):.4f}")
    c9.metric("Kurtosis", f"{kurtosis(s.dropna()):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=col, nbins=30, marginal="box",
                           color_discrete_sequence=["steelblue"],
                           title=f"Distribution – {col}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(df, x="Star_Label", y=col, color="Star_Label",
                     color_discrete_map=STAR_COLORS,
                     category_orders={"Star_Label": list(STAR_MAP.values())},
                     title=f"{col} by Star Type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Summary Table (all numeric columns)")
    rows = []
    for c in NUM_COLS:
        v = df[c].astype(float)
        rows.append({
            "Column": c, "Min": round(v.min(),4), "Max": round(v.max(),4),
            "Range": round(v.max()-v.min(),4), "Mean": round(v.mean(),4),
            "Median": round(v.median(),4), "Std": round(v.std(),4),
            "Variance": round(v.var(),4), "IQR": round(v.quantile(.75)-v.quantile(.25),4),
            "Skewness": round(skew(v.dropna()),4), "Kurtosis": round(kurtosis(v.dropna()),4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Univariate Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "4. Univariate Analysis":
    st.title("4. Univariate Analysis")

    plot_type = st.selectbox("Choose plot", [
        "Boxplots (all numeric)", "Histogram + Density",
        "Violin – Temperature by Star Type", "Bar Chart – Spectral Class",
        "Pie Chart – Star Type", "Density Curves by Star Type",
        "Faceted Histograms",
    ])

    if plot_type == "Boxplots (all numeric)":
        fig = make_subplots(rows=1, cols=4, subplot_titles=NUM_COLS)
        for i, col in enumerate(NUM_COLS, 1):
            fig.add_trace(go.Box(y=df[col].astype(float), name=col,
                                 marker_color="steelblue", showlegend=False), row=1, col=i)
        fig.update_layout(title="Boxplots – Outlier Detection", height=450)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("IQR Outlier Counts")
        out_rows = []
        for col in NUM_COLS:
            v = df[col].astype(float)
            q1,q3 = v.quantile(.25), v.quantile(.75)
            iqr = q3-q1
            n = int(((v < q1-1.5*iqr) | (v > q3+1.5*iqr)).sum())
            out_rows.append({"Column": col, "Outliers (IQR method)": n})
        st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)

    elif plot_type == "Histogram + Density":
        col = st.selectbox("Column", NUM_COLS)
        bins = st.slider("Bins", 10, 60, 25)
        fig = px.histogram(df, x=col, nbins=bins, marginal="violin",
                           color_discrete_sequence=["steelblue"],
                           title=f"Histogram – {col}", histnorm="probability density")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Violin – Temperature by Star Type":
        fig = px.violin(df, x="Star_Label", y="Temperature (K)",
                        color="Star_Label", color_discrete_map=STAR_COLORS,
                        box=True, points="outliers",
                        category_orders={"Star_Label": list(STAR_MAP.values())},
                        title="Violin Plot – Temperature by Star Type")
        fig.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bar Chart – Spectral Class":
        sc = df["Spectral Class"].value_counts().reset_index()
        sc.columns = ["Class", "Count"]
        tab1, tab2, tab3 = st.tabs(["Vertical", "Horizontal", "Percentage"])
        with tab1:
            fig = px.bar(sc, x="Class", y="Count", color="Count",
                         color_continuous_scale="Blues", title="Spectral Class Frequency")
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.bar(sc, y="Class", x="Count", orientation="h",
                         color="Count", color_continuous_scale="Blues",
                         title="Spectral Class – Horizontal")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            sc["Percentage"] = (sc["Count"] / sc["Count"].sum() * 100).round(1)
            fig = px.bar(sc, x="Class", y="Percentage",
                         color="Percentage", color_continuous_scale="Blues",
                         text="Percentage", title="Spectral Class – Percentage")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pie Chart – Star Type":
        st_counts = df["Star_Label"].value_counts().reset_index()
        st_counts.columns = ["Star Type","Count"]
        fig = px.pie(st_counts, names="Star Type", values="Count",
                     color="Star Type", color_discrete_map=STAR_COLORS,
                     hole=0.35, title="Star Type Distribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Density Curves by Star Type":
        import plotly.figure_factory as ff
        groups = [df[df["Star_Label"]==lbl]["Temperature (K)"].astype(float).dropna().tolist()
                  for lbl in STAR_MAP.values()]
        labels = list(STAR_MAP.values())
        colors = list(STAR_COLORS.values())
        fig = ff.create_distplot(groups, labels, colors=colors, show_hist=False, show_rug=False)
        fig.update_layout(title="Density Plot – Temperature by Star Type",
                          xaxis_title="Temperature (K)", height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Faceted Histograms":
        fig = px.histogram(df, x="Temperature (K)", color="Star_Label",
                           facet_col="Star_Label", facet_col_wrap=3,
                           color_discrete_map=STAR_COLORS, nbins=15,
                           title="Faceted Histograms – Temperature by Star Type",
                           height=550)
        fig.update_layout(showlegend=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – Bivariate Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "5. Bivariate Analysis":
    st.title("5. Bivariate Analysis")

    plot_type = st.selectbox("Choose plot", [
        "Contingency Table – Star Type × Color",
        "Stacked / Grouped Bar",
        "Scatter – Hertzsprung-Russell Diagram",
        "Correlation Heatmap",
        "Box & Violin – Temperature / Magnitude by Star Type",
        "Correlogram (Scatter Matrix)",
    ])

    if plot_type == "Contingency Table – Star Type × Color":
        ct = pd.crosstab(df["Star_Label"], df["Color_Group"])
        st.subheader("Count Table")
        st.dataframe(ct, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Row % (within Star Type)")
            st.dataframe((ct.div(ct.sum(axis=1),axis=0)*100).round(1), use_container_width=True)
        with col2:
            st.subheader("Column % (within Color)")
            st.dataframe((ct.div(ct.sum(axis=0),axis=1)*100).round(1), use_container_width=True)

    elif plot_type == "Stacked / Grouped Bar":
        ct = pd.crosstab(df["Star_Label"], df["Color_Group"])
        ct_long = ct.reset_index().melt(id_vars="Star_Label",
                                        var_name="Color_Group", value_name="Count")
        tab1, tab2 = st.tabs(["Stacked", "Grouped"])
        with tab1:
            fig = px.bar(ct_long, x="Star_Label", y="Count", color="Color_Group",
                         barmode="stack", title="Stacked Bar – Star Type × Color Group",
                         category_orders={"Star_Label": list(STAR_MAP.values())})
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.bar(ct_long, x="Star_Label", y="Count", color="Color_Group",
                         barmode="group", title="Grouped Bar – Star Type × Color Group",
                         category_orders={"Star_Label": list(STAR_MAP.values())})
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Scatter – Hertzsprung-Russell Diagram":
        st.markdown("The **H-R Diagram** plots Temperature vs Luminosity — the fundamental stellar classification chart.")
        fig = px.scatter(df, x="Temperature (K)", y="Log_Luminosity",
                         color="Star_Label", color_discrete_map=STAR_COLORS,
                         hover_data=["Spectral Class","Abs_Magnitude","Radius(R/Ro)"],
                         title="Hertzsprung–Russell Diagram  |  Temperature vs Log Luminosity",
                         labels={"Log_Luminosity": "Log Luminosity (L/Lo)"},
                         height=550)
        fig.update_xaxes(autorange="reversed")
        fig.update_traces(marker=dict(size=7, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlation Heatmap":
        num_h = ["Temperature (K)","Luminosity(L/Lo)","Radius(R/Ro)",
                 "Abs_Magnitude","Log_Luminosity","Log_Radius"]
        corr = df[num_h].astype(float).corr().round(3)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto",
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(corr, use_container_width=True)

    elif plot_type == "Box & Violin – Temperature / Magnitude by Star Type":
        metric = st.radio("Metric", ["Temperature (K)", "Abs_Magnitude"], horizontal=True)
        tab1, tab2 = st.tabs(["Box Plot", "Violin Plot"])
        with tab1:
            fig = px.box(df, x="Star_Label", y=metric, color="Star_Label",
                         color_discrete_map=STAR_COLORS, points="outliers",
                         category_orders={"Star_Label": list(STAR_MAP.values())},
                         title=f"Box Plot – {metric} by Star Type")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            fig = px.violin(df, x="Star_Label", y=metric, color="Star_Label",
                            color_discrete_map=STAR_COLORS, box=True,
                            category_orders={"Star_Label": list(STAR_MAP.values())},
                            title=f"Violin Plot – {metric} by Star Type")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Correlogram (Scatter Matrix)":
        cols4 = ["Temperature (K)","Log_Luminosity","Log_Radius","Abs_Magnitude"]
        fig = px.scatter_matrix(df, dimensions=cols4, color="Star_Label",
                                color_discrete_map=STAR_COLORS,
                                title="Correlogram – Scatter Plot Matrix",
                                height=650)
        fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Multivariate Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "6. Multivariate Analysis":
    st.title("6. Multivariate Analysis")

    plot_type = st.selectbox("Choose plot", [
        "3-Variable Scatter (X, Y, Color)",
        "Bubble Plot (X, Y, Color, Size)",
        "4-Variable Scatter (X, Y, Color, Shape)",
        "Facet Wrap – Scatter by Star Type",
        "Parallel Coordinates",
        "Pairs Plot",
    ])

    if plot_type == "3-Variable Scatter (X, Y, Color)":
        x_ax = st.selectbox("X axis", NUM_COLS, index=0)
        y_ax = st.selectbox("Y axis", ["Log_Luminosity","Log_Radius","Abs_Magnitude"], index=0)
        fig = px.scatter(df, x=x_ax, y=y_ax, color="Star_Label",
                         color_discrete_map=STAR_COLORS,
                         hover_data=["Spectral Class"],
                         title=f"3-Variable Scatter: {x_ax} × {y_ax} (color = Star Type)",
                         height=520)
        fig.update_traces(marker=dict(size=7, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Bubble Plot (X, Y, Color, Size)":
        fig = px.scatter(df, x="Temperature (K)", y="Abs_Magnitude",
                         color="Star_Label", size="Log_Radius",
                         color_discrete_map=STAR_COLORS,
                         hover_data=["Spectral Class","Luminosity(L/Lo)"],
                         title="Bubble Plot: Temp × Abs Magnitude  (size = Log Radius)",
                         height=550)
        fig.update_xaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "4-Variable Scatter (X, Y, Color, Shape)":
        symbol_map = {
            "Brown Dwarf":"circle","Red Dwarf":"square",
            "White Dwarf":"diamond","Main Sequence":"triangle-up",
            "Supergiant":"cross","Hypergiant":"star",
        }
        fig = px.scatter(df, x="Temperature (K)", y="Log_Luminosity",
                         color="Star_Label", symbol="Star_Label",
                         color_discrete_map=STAR_COLORS,
                         symbol_map=symbol_map,
                         title="4-Variable Scatter: Temp × Log-Lum  (color + shape = Star Type)",
                         height=540)
        fig.update_xaxes(autorange="reversed")
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Facet Wrap – Scatter by Star Type":
        fig = px.scatter(df, x="Temperature (K)", y="Log_Luminosity",
                         facet_col="Star_Label", facet_col_wrap=3,
                         color="Star_Label", color_discrete_map=STAR_COLORS,
                         title="Facet Wrap – Scatter by Star Type",
                         height=550)
        fig.update_layout(showlegend=False)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Parallel Coordinates":
        df_pc = df[["Temperature (K)","Log_Luminosity","Log_Radius","Abs_Magnitude","Star type"]].copy()
        fig = px.parallel_coordinates(df_pc,
                                      color="Star type",
                                      color_continuous_scale=px.colors.qualitative.D3,
                                      dimensions=["Temperature (K)","Log_Luminosity","Log_Radius","Abs_Magnitude"],
                                      title="Parallel Coordinates Plot")
        st.plotly_chart(fig, use_container_width=True)

    elif plot_type == "Pairs Plot":
        cols4 = ["Temperature (K)","Log_Luminosity","Log_Radius","Abs_Magnitude"]
        fig = px.scatter_matrix(df, dimensions=cols4, color="Star_Label",
                                color_discrete_map=STAR_COLORS,
                                title="Pairs Plot (Scatter Matrix)",
                                height=680)
        fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.6))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "7. Statistical Analysis":
    st.title("7. Statistical Analysis")

    tab1, tab2, tab3 = st.tabs(["Correlation Analysis", "Grouped Statistics", "Advanced Plots"])

    with tab1:
        x_col = st.selectbox("X variable", NUM_COLS, index=0, key="stat_x")
        y_col = st.selectbox("Y variable", ["Log_Luminosity","Log_Radius","Abs_Magnitude"], index=0, key="stat_y")
        x = df[x_col].astype(float)
        y = df[y_col].astype(float)
        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Pearson r",  f"{r_p:.4f}")
        c2.metric("Pearson p",  f"{p_p:.2e}")
        c3.metric("Spearman r", f"{r_s:.4f}")
        c4.metric("Spearman p", f"{p_s:.2e}")

        m, b = np.polyfit(x, y, 1)
        fig = px.scatter(df, x=x_col, y=y_col, color="Star_Label",
                         color_discrete_map=STAR_COLORS, opacity=0.7,
                         title=f"{x_col} vs {y_col}  |  Pearson r={r_p:.3f}")
        xr = np.linspace(x.min(), x.max(), 200)
        fig.add_trace(go.Scatter(x=xr, y=m*xr+b, mode="lines",
                                 line=dict(color="red", width=2, dash="dash"),
                                 name=f"Regression: y={m:.2e}x+{b:.2f}"))
        fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        grp_col = st.selectbox("Metric", NUM_COLS, key="grp_col")
        grp = (df.groupby("Star_Label")[grp_col]
               .agg(["mean","median","std","min","max"])
               .round(3)
               .reset_index()
               .rename(columns={"Star_Label":"Star Type"}))
        st.dataframe(grp, use_container_width=True, hide_index=True)
        fig = px.bar(grp, x="Star Type", y="mean", error_y="std",
                     color="Star Type", color_discrete_map=STAR_COLORS,
                     title=f"Mean {grp_col} by Star Type (error bars = std)",
                     category_orders={"Star Type": list(STAR_MAP.values())})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        df_sub = df[df["Color_Group"].isin(["Red","Blue","White"])]
        fig = px.box(df_sub, x="Star_Label", y="Temperature (K)",
                     color="Color_Group",
                     color_discrete_map={"Red":"tomato","Blue":"steelblue","White":"lightgrey"},
                     category_orders={"Star_Label": list(STAR_MAP.values())},
                     title="Faceted Box Plot: Temperature by Star Type × Color Group")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        num_h = ["Temperature (K)","Luminosity(L/Lo)","Radius(R/Ro)",
                 "Abs_Magnitude","Log_Luminosity","Log_Radius"]
        corr2 = df[num_h].astype(float).corr().round(3)
        fig2 = px.imshow(corr2, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Full Correlation Matrix", aspect="auto")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – Time Series Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "8. Time Series Analysis":
    st.title("8. Time Series Analysis")
    st.markdown("Using the **AirPassengers** benchmark dataset (1949–1960, 144 monthly observations).")

    @st.cache_data
    def get_ts():
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
        return pd.Series(air_raw,
                         index=pd.date_range("1949-01", periods=len(air_raw), freq="ME"),
                         name="Passengers")

    ts = get_ts()
    ts_df = ts.reset_index(); ts_df.columns = ["Date","Passengers"]

    tab1, tab2, tab3, tab4 = st.tabs(["Trend", "Decomposition", "ACF / PACF", "SARIMA Forecast"])

    with tab1:
        fig = px.line(ts_df, x="Date", y="Passengers",
                      title="AirPassengers: Monthly Airline Passengers (1949–1960)",
                      markers=False, color_discrete_sequence=["steelblue"])
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dec = seasonal_decompose(ts, model="multiplicative", period=12)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=["Observed","Trend","Seasonal","Residual"])
        pairs = [(dec.observed,"steelblue"),(dec.trend,"darkgreen"),
                 (dec.seasonal,"coral"),(dec.resid,"grey")]
        for i,(d,c) in enumerate(pairs,1):
            fig.add_trace(go.Scatter(x=d.index, y=d.values,
                                     line=dict(color=c), showlegend=False), row=i, col=1)
        fig.update_layout(height=700,
                          title="Time Series Decomposition (Multiplicative Model)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig_m, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(ts,  lags=40, ax=axes[0], color="steelblue")
        axes[0].set_title("ACF")
        plot_pacf(ts, lags=20, ax=axes[1], color="coral", method="ywm")
        axes[1].set_title("PACF")
        plt.tight_layout()
        st.pyplot(fig_m)
        plt.close(fig_m)

    with tab4:
        @st.cache_data
        def fit_sarima(ts_vals):
            ts_s = pd.Series(ts_vals,
                             index=pd.date_range("1949-01", periods=len(ts_vals), freq="ME"))
            log_ts = np.log(ts_s)
            fit = SARIMAX(log_ts, order=(2,1,2), seasonal_order=(1,1,1,12),
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fcast = fit.get_forecast(steps=12)
            return np.exp(fcast.predicted_mean), np.exp(fcast.conf_int())

        steps = st.slider("Forecast horizon (months)", 6, 24, 12)
        with st.spinner("Fitting SARIMA model…"):
            f_mean, f_ci = fit_sarima(ts.values.tolist())
        f_idx = pd.date_range("1961-01", periods=12, freq="ME")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Actual",
                                 line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=f_idx, y=f_mean.values, name="Forecast",
                                 line=dict(color="coral", dash="dash")))
        fig.add_trace(go.Scatter(
            x=list(f_idx)+list(f_idx[::-1]),
            y=list(f_ci.iloc[:,1].values)+list(f_ci.iloc[:,0].values[::-1]),
            fill="toself", fillcolor="rgba(255,100,70,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
        fig.update_layout(title="SARIMA(2,1,2)(1,1,1,12) Forecast – Next 12 Months",
                          xaxis_title="Year", yaxis_title="Passengers (thousands)",
                          height=480)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 – Clustering Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif section == "9. Clustering Analysis":
    st.title("9. Clustering Analysis")

    feat_cols = ["Temperature (K)","Luminosity(L/Lo)","Radius(R/Ro)","Abs_Magnitude"]
    df_cl = df[feat_cols].astype(float).dropna()
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(df_cl)

    tab1, tab2, tab3 = st.tabs(["K-Means & Elbow", "K-Means vs True Labels", "Hierarchical"])

    with tab1:
        max_k = st.slider("Max K for elbow", 5, 15, 10)
        with st.spinner("Computing WCSS…"):
            wcss = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_sc).inertia_
                    for k in range(1, max_k+1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=wcss,
                                 mode="lines+markers", marker=dict(size=8, color="steelblue"),
                                 line=dict(color="steelblue", width=2)))
        fig.update_layout(title="Elbow Plot: WCSS vs Number of Clusters",
                          xaxis_title="K", yaxis_title="WCSS (Inertia)",
                          xaxis=dict(tickmode="linear"), height=420)
        st.plotly_chart(fig, use_container_width=True)
        wcss_df = pd.DataFrame({"K": range(1, max_k+1), "WCSS": [round(w,2) for w in wcss]})
        st.dataframe(wcss_df.T, use_container_width=True)

    with tab2:
        k_sel = st.slider("Select K", 2, 10, 6)
        km   = KMeans(n_clusters=k_sel, random_state=42, n_init=10).fit(X_sc)
        pca2 = PCA(n_components=2); X2d = pca2.fit_transform(X_sc)
        true_lbl = df.loc[df_cl.index, "Star type"].astype(int).values

        plot_df = pd.DataFrame(X2d, columns=["PC1","PC2"])
        plot_df["K-Means Cluster"] = km.labels_.astype(str)
        plot_df["True Star Type"]  = [STAR_MAP[t] for t in true_lbl]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="K-Means Cluster",
                             title=f"K-Means (K={k_sel})", height=430, opacity=0.75)
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="True Star Type",
                             color_discrete_map=STAR_COLORS,
                             title="True Star Types", height=430, opacity=0.75)
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Cluster Sizes (K={k_sel})")
        sizes = pd.Series(np.bincount(km.labels_),
                          index=[f"Cluster {i}" for i in range(k_sel)])
        st.dataframe(sizes.rename("Count").reset_index().rename(columns={"index":"Cluster"}),
                     use_container_width=True, hide_index=True)

    with tab3:
        n_samp = st.slider("Sample size for dendrogram", 40, 120, 80)
        rng   = np.random.RandomState(42)
        idx   = rng.choice(len(X_sc), n_samp, replace=False)
        Z     = linkage(X_sc[idx], method="ward")
        n_cut = st.slider("Number of clusters (cut)", 2, 6, 3)
        labels_cut = fcluster(Z, t=n_cut, criterion="maxclust")

        fig_d, ax = plt.subplots(figsize=(14, 5))
        dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=7, color_threshold=10)
        ax.set_title(f"Hierarchical Clustering Dendrogram  (Ward, n={n_samp})",
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Sample Index"); ax.set_ylabel("Euclidean Distance")
        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

        sizes3 = pd.Series(np.bincount(labels_cut)[1:],
                           index=[f"Cluster {i}" for i in range(1, n_cut+1)])
        st.subheader(f"Hierarchical Cluster Sizes (cut at {n_cut})")
        st.dataframe(sizes3.rename("Count").reset_index().rename(columns={"index":"Cluster"}),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 – PCA
# ══════════════════════════════════════════════════════════════════════════════
elif section == "10. Principal Component Analysis":
    st.title("10. Principal Component Analysis (PCA)")
    st.markdown("**Reg No: 23BDS0176**")

    pca_cols   = ["Temperature (K)","Luminosity(L/Lo)","Radius(R/Ro)","Abs_Magnitude"]
    df_pca_raw = df[pca_cols].astype(float).dropna()
    scaler_p   = StandardScaler()
    data_scaled = scaler_p.fit_transform(df_pca_raw)

    pca_full    = PCA()
    pca_full.fit(data_scaled)
    scores_full = pca_full.transform(data_scaled)
    exp_var = pca_full.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    n_pc    = len(exp_var)

    # PCA summary table
    pca_tbl = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(n_pc)],
        "Eigenvalue": pca_full.explained_variance_.round(4),
        "% Variance": exp_var.round(2),
        "Cumulative %": cum_var.round(2),
    })
    st.subheader("PCA Summary")
    st.dataframe(pca_tbl, use_container_width=True, hide_index=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("PC1 explains", f"{exp_var[0]:.1f}%")
    c2.metric("PC1+PC2 explain", f"{cum_var[1]:.1f}%")
    c3.metric("PCs for 90% variance", int(np.searchsorted(cum_var, 90)+1))

    tab1, tab2, tab3, tab4 = st.tabs(["Scree Plot", "Biplot", "PCA Scores", "PC1 vs PC2"])

    with tab1:
        fig = go.Figure()
        fig.add_bar(x=[f"PC{i+1}" for i in range(n_pc)], y=exp_var,
                    name="% Variance", marker_color="steelblue", opacity=0.8)
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_pc)], y=exp_var,
                                 mode="lines+markers", name="% Variance",
                                 line=dict(color="darkblue"), marker=dict(size=8),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_pc)], y=cum_var,
                                 mode="lines+markers+text", name="Cumulative %",
                                 line=dict(color="coral", dash="dash"),
                                 marker=dict(size=8), text=[f"{v:.1f}%" for v in cum_var],
                                 textposition="top center"))
        for i, v in enumerate(exp_var):
            fig.add_annotation(x=f"PC{i+1}", y=v+1, text=f"{v:.1f}%",
                               showarrow=False, font=dict(size=10))
        fig.update_layout(title="Scree Plot: % Variance Explained per Principal Component",
                          yaxis=dict(range=[0,60]), height=480,
                          xaxis_title="Principal Component",
                          yaxis_title="Explained Variance (%)")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        loadings    = pca_full.components_
        star_types  = df.loc[df_pca_raw.index, "Star type"].astype(int).values
        star_labels = [STAR_MAP[t] for t in star_types]

        biplot_df = pd.DataFrame({
            "PC1": scores_full[:,0], "PC2": scores_full[:,1],
            "Star Type": star_labels,
        })
        fig = px.scatter(biplot_df, x="PC1", y="PC2", color="Star Type",
                         color_discrete_map=STAR_COLORS, opacity=0.6,
                         title="PCA Biplot – Scores & Loadings", height=580)
        fig.update_traces(marker=dict(size=5))
        scale = 3.5
        for i, feat in enumerate(pca_cols):
            lx, ly = loadings[0,i]*scale, loadings[1,i]*scale
            fig.add_annotation(x=lx*1.2, y=ly*1.2,
                               text=feat.split("(")[0].strip(),
                               showarrow=True, arrowhead=2,
                               ax=0, ay=0, font=dict(color="red", size=11),
                               arrowcolor="red", arrowwidth=2)
        fig.add_hline(y=0, line_width=0.8, line_dash="dash", line_color="grey")
        fig.add_vline(x=0, line_width=0.8, line_dash="dash", line_color="grey")
        fig.update_xaxes(title=f"PC1 ({exp_var[0]:.1f}% variance)")
        fig.update_yaxes(title=f"PC2 ({exp_var[1]:.1f}% variance)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        pca_scores = pd.DataFrame(scores_full,
                                   columns=[f"PC{i+1}" for i in range(n_pc)],
                                   index=df_pca_raw.index)
        st.subheader("pca_scores – First 6 rows")
        st.dataframe(pca_scores.head(6).round(4), use_container_width=True)
        st.subheader("Full PCA Scores Table")
        st.dataframe(pca_scores.round(4), use_container_width=True, height=350)

    with tab4:
        pca_scores2 = pd.DataFrame(scores_full[:, :2], columns=["PC1","PC2"],
                                    index=df_pca_raw.index)
        fig = px.scatter(pca_scores2, x="PC1", y="PC2",
                         color_discrete_sequence=["blue"],
                         title="PC1 vs PC2 Scatter Plot – Registration No: 23BDS0176",
                         height=540)
        fig.update_traces(marker=dict(size=7, opacity=0.6,
                                      line=dict(color="white", width=0.5)))
        fig.add_hline(y=0, line_width=0.8, line_dash="dash", line_color="grey")
        fig.add_vline(x=0, line_width=0.8, line_dash="dash", line_color="grey")
        fig.update_xaxes(title=f"PC1 ({exp_var[0]:.1f}% variance)")
        fig.update_yaxes(title=f"PC2 ({exp_var[1]:.1f}% variance)")
        st.plotly_chart(fig, use_container_width=True)
