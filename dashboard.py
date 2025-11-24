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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", 
    "📊 Model Summary", 
    "🔍 Coefficients", 
    "📉 Diagnostics",
    "🎯 Predictions",
    "🔮 Testing Prediction",
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

# TAB 6: Testing Prediction
with tab6:
    st.header("🔮 Testing Prediction")
    st.write("Predict new values by entering feature values manually")
    
    # Helper functions
    def predict_new_data(new_data_dict, B_hat, features):
        try:
            X_new = np.array([new_data_dict[name] for name in features]).reshape(1, -1)
            X_new_with_intercept = np.hstack([np.ones((1, 1)), X_new])
            Y_pred = X_new_with_intercept @ B_hat
            return Y_pred[0][0], None
        except Exception as e:
            return None, str(e)
    
    def predict_with_confidence_interval(new_data_dict, B_hat, features, MSE, XTX_inv, t_critical):
        try:
            X_new = np.array([new_data_dict[name] for name in features]).reshape(1, -1)
            X_new_with_intercept = np.hstack([np.ones((1, 1)), X_new])
            Y_pred = (X_new_with_intercept @ B_hat)[0][0]
            
            SE_pred = np.sqrt(MSE * (1 + X_new_with_intercept @ XTX_inv @ X_new_with_intercept.T))
            SE_pred = SE_pred[0][0]
            
            CI_lower = Y_pred - t_critical * SE_pred
            CI_upper = Y_pred + t_critical * SE_pred
            
            return Y_pred, CI_lower, CI_upper, SE_pred, None
        except Exception as e:
            return None, None, None, None, str(e)
    
    # Get XTX_inv from results
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    t_critical = t.ppf(0.975, results['df_residual'])
    
    # Tabs untuk prediction modes
    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "📝 Single Prediction", 
        "📊 Scenario Analysis", 
        "📦 Batch Prediction"
    ])
    
    # SUB-TAB 1: Single Prediction
    with pred_tab1:
        st.subheader("Enter Feature Values")
        
        col1, col2 = st.columns(2)
        new_data = {}
        
        for i, feature in enumerate(selected_features):
            # Get statistics for default values
            mean_val = df_clean[feature].mean()
            min_val = float(df_clean[feature].min())
            max_val = float(df_clean[feature].max())
            
            with col1 if i % 2 == 0 else col2:
                new_data[feature] = st.number_input(
                    f"{feature[:50]}{'...' if len(feature) > 50 else ''}",
                    value=float(mean_val),
                    min_value=min_val,
                    max_value=max_val,
                    format="%.4f",
                    key=f"single_{i}"
                )
        
        if st.button("🎯 Predict", type="primary", key="predict_single"):
            prediction, error = predict_new_data(new_data, results['B_hat'], selected_features)
            
            if error:
                st.error(f"❌ Prediction failed: {error}")
            else:
                pred_with_ci, ci_lower, ci_upper, se, error_ci = predict_with_confidence_interval(
                    new_data, results['B_hat'], selected_features, 
                    results['MSE'], XTX_inv, t_critical
                )
                
                if error_ci:
                    st.error(f"❌ CI calculation failed: {error_ci}")
                else:
                    st.success("✅ Prediction successful!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Value", f"{prediction:.4f}")
                    with col2:
                        st.metric("Standard Error", f"{se:.4f}")
                    with col3:
                        st.metric("95% CI Width", f"{ci_upper - ci_lower:.4f}")
                    
                    st.info(f"📊 **95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]")
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=[prediction],
                        y=['Prediction'],
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[ci_upper - prediction],
                            arrayminus=[prediction - ci_lower],
                            color='lightblue',
                            thickness=10
                        ),
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name='Prediction with 95% CI'
                    ))
                    
                    fig.update_layout(
                        title='Prediction with Confidence Interval',
                        xaxis_title=target,
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # SUB-TAB 2: Scenario Analysis
    with pred_tab2:
        st.subheader("Compare Multiple Scenarios")
        
        # Baseline values
        st.write("**Baseline Scenario (Mean Values)**")
        baseline_data = {feature: df_clean[feature].mean() for feature in selected_features}
        
        # Display baseline
        baseline_cols = st.columns(min(3, len(selected_features)))
        for i, (feature, value) in enumerate(baseline_data.items()):
            with baseline_cols[i % min(3, len(selected_features))]:
                st.metric(feature[:30], f"{value:.2f}")
        
        st.divider()
        
        # Scenario builder
        st.write("**Create Scenarios**")
        num_scenarios = st.number_input("Number of scenarios", min_value=1, max_value=5, value=3)
        
        scenarios = {'Baseline': baseline_data.copy()}
        
        for i in range(num_scenarios):
            with st.expander(f"Scenario {i+1}"):
                scenario_name = st.text_input(f"Scenario {i+1} Name", value=f"Scenario {i+1}", key=f"scenario_name_{i}")
                
                scenario_data = baseline_data.copy()
                
                # Select feature to modify
                feature_to_modify = st.selectbox(
                    "Select feature to modify", 
                    selected_features,
                    key=f"scenario_feature_{i}"
                )
                
                # Modify value
                change_type = st.radio(
                    "Change type",
                    ["Absolute Value", "Percentage Change"],
                    key=f"scenario_change_type_{i}",
                    horizontal=True
                )
                
                if change_type == "Absolute Value":
                    new_value = st.number_input(
                        f"New value for {feature_to_modify}",
                        value=float(baseline_data[feature_to_modify]),
                        key=f"scenario_value_{i}"
                    )
                    scenario_data[feature_to_modify] = new_value
                else:
                    pct_change = st.slider(
                        "Percentage change",
                        min_value=-50,
                        max_value=50,
                        value=0,
                        key=f"scenario_pct_{i}"
                    )
                    scenario_data[feature_to_modify] = baseline_data[feature_to_modify] * (1 + pct_change/100)
                
                scenarios[scenario_name] = scenario_data
        
        if st.button("📊 Compare Scenarios", type="primary", key="compare_scenarios"):
            # Calculate predictions for all scenarios
            scenario_results = []
            baseline_pred, _ = predict_new_data(scenarios['Baseline'], results['B_hat'], selected_features)
            
            for scenario_name, scenario_data in scenarios.items():
                pred, error = predict_new_data(scenario_data, results['B_hat'], selected_features)
                
                if not error:
                    change = pred - baseline_pred if scenario_name != 'Baseline' else 0
                    pct_change = (change / baseline_pred * 100) if baseline_pred != 0 and scenario_name != 'Baseline' else 0
                    
                    scenario_results.append({
                        'Scenario': scenario_name,
                        'Prediction': pred,
                        'Change': change,
                        'Change (%)': pct_change
                    })
            
            # Display results
            results_df = pd.DataFrame(scenario_results)
            
            st.dataframe(
                results_df.style.format({
                    'Prediction': '{:.4f}',
                    'Change': '{:+.4f}',
                    'Change (%)': '{:+.2f}%'
                }).background_gradient(subset=['Prediction'], cmap='RdYlGn_r'),
                hide_index=True,
                use_container_width=True
            )
            
            # Visualization
            fig = go.Figure()
            
            colors = ['blue' if name == 'Baseline' else 'green' if change > 0 else 'red' 
                     for name, change in zip(results_df['Scenario'], results_df['Change'])]
            
            fig.add_trace(go.Bar(
                x=results_df['Prediction'],
                y=results_df['Scenario'],
                orientation='h',
                marker_color=colors,
                text=results_df['Prediction'].round(4),
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Scenario Comparison',
                xaxis_title=f'Predicted {target}',
                yaxis_title='Scenario',
                height=max(300, len(scenarios) * 60)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # SUB-TAB 3: Batch Prediction
    with pred_tab3:
        st.subheader("Batch Prediction from CSV or Manual Entry")
        
        upload_method = st.radio(
            "Select input method",
            ["Upload CSV", "Manual Entry"],
            horizontal=True
        )
        
        if upload_method == "Upload CSV":
            st.write("Upload a CSV file with the same features as the training data")
            
            batch_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'], key="batch_upload")
            
            if batch_file is not None:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.write(f"Loaded {len(batch_df)} rows")
                    
                    # Check if all features are present
                    missing_features = [f for f in selected_features if f not in batch_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Missing features: {', '.join(missing_features)}")
                    else:
                        st.success("✅ All required features found!")
                        
                        if st.button("🎯 Predict Batch", type="primary", key="predict_batch_csv"):
                            # Prepare batch data
                            X_batch_raw = batch_df[selected_features].values
                            X_batch = np.hstack([np.ones((len(X_batch_raw), 1)), X_batch_raw])
                            
                            # Predict
                            Y_batch_pred = X_batch @ results['B_hat']
                            
                            # Add predictions to dataframe
                            batch_df['Predicted_Value'] = Y_batch_pred.flatten()
                            
                            st.success(f"✅ Predicted {len(batch_df)} values!")
                            
                            # Display results
                            st.dataframe(
                                batch_df.style.format({'Predicted_Value': '{:.4f}'}),
                                use_container_width=True
                            )
                            
                            # Download button
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                key='download_batch_csv'
                            )
                            
                            # Visualization
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(batch_df) + 1)),
                                y=batch_df['Predicted_Value'],
                                mode='lines+markers',
                                marker=dict(size=8, color='blue'),
                                line=dict(color='lightblue', width=2),
                                name='Predictions'
                            ))
                            
                            fig.update_layout(
                                title='Batch Prediction Results',
                                xaxis_title='Observation',
                                yaxis_title=f'Predicted {target}',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        else:  # Manual Entry
            st.write("Enter multiple observations manually")
            
            num_observations = st.number_input("Number of observations", min_value=1, max_value=10, value=3)
            
            manual_data = []
            
            for obs_idx in range(num_observations):
                with st.expander(f"Observation {obs_idx + 1}"):
                    obs_data = {}
                    cols = st.columns(2)
                    
                    for i, feature in enumerate(selected_features):
                        mean_val = df_clean[feature].mean()
                        
                        with cols[i % 2]:
                            obs_data[feature] = st.number_input(
                                f"{feature[:40]}",
                                value=float(mean_val),
                                key=f"manual_{obs_idx}_{i}"
                            )
                    
                    manual_data.append(obs_data)
            
            if st.button("🎯 Predict Manual Batch", type="primary", key="predict_batch_manual"):
                # Prepare batch data
                X_batch_raw = np.array([[obs[f] for f in selected_features] for obs in manual_data])
                X_batch = np.hstack([np.ones((len(X_batch_raw), 1)), X_batch_raw])
                
                # Predict
                Y_batch_pred = X_batch @ results['B_hat']
                
                # Create results dataframe
                results_list = []
                for i, (obs, pred) in enumerate(zip(manual_data, Y_batch_pred.flatten())):
                    row = {'Observation': i + 1}
                    row.update(obs)
                    row['Predicted_Value'] = pred
                    results_list.append(row)
                
                manual_results_df = pd.DataFrame(results_list)
                
                st.success(f"✅ Predicted {len(manual_results_df)} values!")
                
                # Display results
                st.dataframe(
                    manual_results_df.style.format({col: '{:.4f}' for col in manual_results_df.columns if col != 'Observation'}),
                    use_container_width=True
                )
                
                # Download button
                csv = manual_results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "manual_predictions.csv",
                    "text/csv",
                    key='download_batch_manual'
                )

# TAB 7: Data Explorer
with tab7:
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