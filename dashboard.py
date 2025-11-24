import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t, f
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Multiple Linear Regression Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Multiple Linear Regression Dashboard</p>', unsafe_allow_html=True)

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Dataset CSV", type=['csv'])
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")

# Main content
if uploaded_file is None:
    st.info("👈 Please upload a CSV file to begin analysis")
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

# Feature Engineering
st.sidebar.header("🔧 Feature Engineering")

# Find industry columns
industry_cols = [col for col in df.columns if 'industri' in col.lower() and 
                any(q in col.lower() for q in ['q1', 'q2', 'q3', 'q4'])]

if len(industry_cols) > 0:
    df['jumlah_industri_total'] = df[industry_cols].sum(axis=1)
    st.sidebar.success(f"Created: jumlah_industri_total ({len(industry_cols)} columns)")

# Target selection
target_options = [col for col in df.columns if 'miskin' in col.lower() or 'kemiskinan' in col.lower()]
if not target_options:
    target_options = df.select_dtypes(include=[np.number]).columns.tolist()

target = st.sidebar.selectbox("Select Target Variable", target_options, 
                            index=0 if target_options else None)

# Feature selection
default_features = [
    'Indeks Pembangunan Manusia (IPM)',
    'Tingkat Pengangguran Terbuka (TPT) - Agustus',
    'Pengeluaran Per Kapita Riil Disesuaikan (Ribu Rupiah)',
    'Gini Ratio',
    'Jumlah Penerima Bansos',
    'Harapan Lama Sekolah (Tahun)',
    'STATUS BALITA PENDEK TB-U',
    'PRODUKSI PADI (TON)',
    'jumlah_industri_total',
    'Laju Pertumbuhan Penduduk per Tahun (Persen)',
    'Kepadatan Penduduk per km persegi (km²)'
]

available_features = [f for f in default_features if f in df.columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

selected_features = st.sidebar.multiselect(
    "Select Features (Predictors)",
    numeric_cols,
    default=available_features[:5] if available_features else numeric_cols[:5]
)

if not selected_features:
    st.warning("⚠️ Please select at least one feature")
    st.stop()

# Data cleaning
df_clean = df[[target] + selected_features].dropna()

st.sidebar.metric("Original Rows", len(df))
st.sidebar.metric("Clean Rows", len(df_clean))
st.sidebar.metric("Features Selected", len(selected_features))

# Prepare matrices
@st.cache_data
def prepare_regression_data(df_clean, target, features):
    Y = df_clean[target].values.reshape(-1, 1)
    X_raw = df_clean[features].values
    intercept = np.ones((X_raw.shape[0], 1))
    X = np.hstack((intercept, X_raw))
    
    N = X.shape[0]
    k = X.shape[1] - 1
    p = X.shape[1]
    
    return Y, X, X_raw, N, k, p

Y, X, X_raw, N, k, p = prepare_regression_data(df_clean, target, selected_features)

# Calculate regression
@st.cache_data
def calculate_regression(X, Y, N, k, p):
    # Coefficients
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    XTY = X.T @ Y
    B_hat = XTX_inv @ XTY
    
    # Predictions and residuals
    Y_hat = X @ B_hat
    e = Y - Y_hat
    
    # Sum of squares
    Y_mean = np.mean(Y)
    SST = np.sum((Y - Y_mean)**2)
    SSR = np.sum((Y_hat - Y_mean)**2)
    SSE = np.sum(e**2)
    
    # Degrees of freedom
    df_regression = k
    df_residual = N - k - 1
    df_total = N - 1
    
    # Mean squares
    MSR = SSR / df_regression
    MSE = SSE / df_residual
    
    # R-squared
    R2 = SSR / SST
    Adj_R2 = 1 - ((1 - R2) * (N - 1) / (N - k - 1))
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(e))
    
    # F-test
    F_stat = MSR / MSE
    F_critical = f.ppf(0.95, df_regression, df_residual)
    F_pvalue = 1 - f.cdf(F_stat, df_regression, df_residual)
    
    # Standard errors and t-tests
    s2 = MSE
    Var_B = s2 * XTX_inv
    SE_B = np.sqrt(np.diag(Var_B)).reshape(-1, 1)
    t_stat = B_hat / SE_B
    t_critical = t.ppf(0.975, df_residual)
    p_values = 2 * (1 - t.cdf(np.abs(t_stat), df=df_residual))
    
    # Confidence intervals
    CI_lower = B_hat - t_critical * SE_B
    CI_upper = B_hat + t_critical * SE_B
    
    return {
        'B_hat': B_hat, 'Y_hat': Y_hat, 'e': e,
        'SST': SST, 'SSR': SSR, 'SSE': SSE,
        'MSR': MSR, 'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE,
        'R2': R2, 'Adj_R2': Adj_R2,
        'F_stat': F_stat, 'F_critical': F_critical, 'F_pvalue': F_pvalue,
        'SE_B': SE_B, 't_stat': t_stat, 'p_values': p_values,
        'CI_lower': CI_lower, 'CI_upper': CI_upper,
        'df_regression': df_regression, 'df_residual': df_residual, 'df_total': df_total
    }

results = calculate_regression(X, Y, N, k, p)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview", 
    "📊 Model Summary", 
    "🔍 Coefficients", 
    "📉 Diagnostics",
    "🎯 Predictions",
    "📐 Data Explorer"
])

# TAB 1: Overview
with tab1:
    st.header("Model Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{results['R2']:.4f}", 
                help="Proportion of variance explained")
    with col2:
        st.metric("Adjusted R²", f"{results['Adj_R2']:.4f}",
                help="R² adjusted for number of predictors")
    with col3:
        st.metric("RMSE", f"{results['RMSE']:.4f}",
                help="Root Mean Square Error")
    with col4:
        st.metric("MAE", f"{results['MAE']:.4f}",
                help="Mean Absolute Error")
    
    st.divider()
    
    # Model information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        info_df = pd.DataFrame({
            'Metric': ['Observations', 'Features', 'Parameters', 'DF Residual'],
            'Value': [N, k, p, results['df_residual']]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("F-Test Results")
        f_test_df = pd.DataFrame({
            'Metric': ['F-Statistic', 'F-Critical (α=0.05)', 'P-value', 'Significant?'],
            'Value': [
                f"{results['F_stat']:.4f}",
                f"{results['F_critical']:.4f}",
                f"{results['F_pvalue']:.2e}",
                "✅ Yes" if results['F_pvalue'] < 0.05 else "❌ No"
            ]
        })
        st.dataframe(f_test_df, hide_index=True, use_container_width=True)
    
    # Regression equation
    st.subheader("Regression Equation")
    equation = f"ŷ = {results['B_hat'][0][0]:.4f}"
    for i, feature in enumerate(selected_features, 1):
        coef = results['B_hat'][i][0]
        sign = "+" if coef >= 0 else "-"
        equation += f" {sign} {abs(coef):.4f}×({feature[:30]}...)" if len(feature) > 30 else f" {sign} {abs(coef):.4f}×({feature})"
    
    st.code(equation, language=None)

# TAB 2: Model Summary
with tab2:
    st.header("ANOVA Table")
    
    anova_df = pd.DataFrame({
        'Source': ['Regression', 'Residual', 'Total'],
        'DF': [results['df_regression'], results['df_residual'], results['df_total']],
        'Sum of Squares': [results['SSR'], results['SSE'], results['SST']],
        'Mean Square': [results['MSR'], results['MSE'], '-']
    })
    
    st.dataframe(anova_df, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # Visualization of variance decomposition
    st.subheader("Variance Decomposition")
    
    fig = go.Figure(data=[
        go.Bar(name='SSR (Explained)', x=['Variance'], y=[results['SSR']], 
            marker_color='lightblue'),
        go.Bar(name='SSE (Unexplained)', x=['Variance'], y=[results['SSE']], 
            marker_color='lightcoral')
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Total Sum of Squares Decomposition',
        yaxis_title='Sum of Squares',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Coefficients
with tab3:
    st.header("Regression Coefficients")
    
    coefficient_names = ['Intercept'] + selected_features
    
    coef_df = pd.DataFrame({
        'Variable': coefficient_names,
        'Coefficient': results['B_hat'].flatten(),
        'Std Error': results['SE_B'].flatten(),
        't-statistic': results['t_stat'].flatten(),
        'P-value': results['p_values'].flatten(),
        'CI Lower (95%)': results['CI_lower'].flatten(),
        'CI Upper (95%)': results['CI_upper'].flatten(),
        'Significant': ['✅' if p < 0.05 else '❌' for p in results['p_values'].flatten()]
    })
    
    st.dataframe(
        coef_df.style.format({
            'Coefficient': '{:.6f}',
            'Std Error': '{:.6f}',
            't-statistic': '{:.4f}',
            'P-value': '{:.4e}',
            'CI Lower (95%)': '{:.6f}',
            'CI Upper (95%)': '{:.6f}'
        }),
        hide_index=True,
        use_container_width=True
    )
    
    st.divider()
    
    # Coefficient plot
    st.subheader("Coefficient Visualization")
    
    fig = go.Figure()
    
    # Remove intercept for better visualization
    coef_df_no_intercept = coef_df.iloc[1:]
    
    colors = ['green' if p < 0.05 else 'red' for p in coef_df_no_intercept['P-value']]
    
    fig.add_trace(go.Bar(
        x=coef_df_no_intercept['Coefficient'],
        y=coef_df_no_intercept['Variable'],
        orientation='h',
        marker_color=colors,
        text=coef_df_no_intercept['Coefficient'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Feature Coefficients (Green = Significant, Red = Not Significant)',
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=max(400, len(selected_features) * 40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Diagnostics
with tab4:
    st.header("Model Diagnostics")
    
    residuals_flat = results['e'].flatten()
    Y_hat_flat = results['Y_hat'].flatten()
    
    # Create subplot
    col1, col2 = st.columns(2)
    
    with col1:
        # Residuals vs Fitted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=Y_hat_flat,
            y=residuals_flat,
            mode='markers',
            marker=dict(color='steelblue', size=6, opacity=0.6),
            name='Residuals'
        ))
        fig1.add_hline(y=0, line_dash="dash", line_color="red")
        fig1.update_layout(
            title='Residuals vs Fitted Values',
            xaxis_title='Fitted Values',
            yaxis_title='Residuals',
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Q-Q Plot
        theoretical_quantiles = stats.probplot(residuals_flat, dist="norm")[0][0]
        sample_quantiles = stats.probplot(residuals_flat, dist="norm")[0][1]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(color='blue', size=6, opacity=0.6),
            name='Sample Quantiles'
        ))
        
        # Add reference line
        fig2.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=theoretical_quantiles,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Reference Line'
        ))
        
        fig2.update_layout(
            title='Normal Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Histogram of residuals
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=residuals_flat,
            nbinsx=30,
            marker_color='coral',
            opacity=0.7,
            name='Residuals'
        ))
        fig3.update_layout(
            title='Distribution of Residuals',
            xaxis_title='Residuals',
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Residual statistics
        st.subheader("Residual Statistics")
        residual_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{np.mean(residuals_flat):.6f}",
                f"{np.std(residuals_flat, ddof=1):.6f}",
                f"{np.min(residuals_flat):.6f}",
                f"{np.max(residuals_flat):.6f}",
                f"{stats.skew(residuals_flat):.6f}",
                f"{stats.kurtosis(residuals_flat):.6f}"
            ]
        })
        st.dataframe(residual_stats, hide_index=True, use_container_width=True)

# TAB 5: Predictions
with tab5:
    st.header("Actual vs Predicted Values")
    
    # Scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=Y.flatten(),
        y=Y_hat_flat,
        mode='markers',
        marker=dict(color='darkblue', size=8, opacity=0.6),
        name='Data Points'
    ))
    
    # Perfect prediction line
    min_val = min(Y.min(), Y_hat_flat.min())
    max_val = max(Y.max(), Y_hat_flat.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Values',
        xaxis_title=f'Actual {target}',
        yaxis_title=f'Predicted {target}',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Prediction table
    st.subheader("Prediction Details (First 20 Observations)")
    
    pred_df = pd.DataFrame({
        'Observation': range(1, min(21, len(Y) + 1)),
        'Actual': Y.flatten()[:20],
        'Predicted': Y_hat_flat[:20],
        'Residual': residuals_flat[:20],
        'Abs Error': np.abs(residuals_flat[:20]),
        'Pct Error (%)': (np.abs(residuals_flat[:20]) / Y.flatten()[:20] * 100)
    })
    
    st.dataframe(
        pred_df.style.format({
            'Actual': '{:.4f}',
            'Predicted': '{:.4f}',
            'Residual': '{:.4f}',
            'Abs Error': '{:.4f}',
            'Pct Error (%)': '{:.2f}'
        }),
        hide_index=True,
        use_container_width=True
    )

# TAB 6: Data Explorer
with tab6:
    st.header("Data Exploration")
    
    # Target distribution
    st.subheader(f"Distribution of Target: {target}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df_clean[target],
            nbinsx=30,
            marker_color='skyblue',
            opacity=0.7
        ))
        fig.update_layout(
            title='Target Distribution',
            xaxis_title=target,
            yaxis_title='Frequency',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=df_clean[target],
            marker_color='lightblue',
            name=target
        ))
        fig.update_layout(
            title='Target Boxplot',
            yaxis_title=target,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Correlation heatmap
    st.subheader("Correlation Matrix")
    
    corr_matrix = df_clean[[target] + selected_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Feature vs Target scatter
    st.subheader("Feature vs Target Relationships")
    
    selected_feature = st.selectbox("Select a feature to visualize", selected_features)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_clean[selected_feature],
        y=df_clean[target],
        mode='markers',
        marker=dict(color='steelblue', size=8, opacity=0.6),
        name='Data Points'
    ))
    
    # Add trend line
    z = np.polyfit(df_clean[selected_feature], df_clean[target], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df_clean[selected_feature].min(), df_clean[selected_feature].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=p(x_trend),
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Trend Line'
    ))
    
    corr = df_clean[[selected_feature, target]].corr().iloc[0, 1]
    
    fig.update_layout(
        title=f'{selected_feature} vs {target} (Correlation: {corr:.3f})',
        xaxis_title=selected_feature,
        yaxis_title=target,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: gray; padding: 2rem;'>
        <p>📊 Multiple Linear Regression Dashboard</p>
        <p>Built with Streamlit • Powered by NumPy & SciPy</p>
    </div>
""", unsafe_allow_html=True)