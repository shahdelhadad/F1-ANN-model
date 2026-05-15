import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import tensorflow as tf

# Page Config
st.set_page_config(page_title="F1 Podium Predictor", page_icon="🏎️", layout="wide")

# Custom CSS for F1 Theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e10600;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff1e17;
        color: white;
    }
    .reportview-container {
        background: url("assets/images/background.png");
        background-size: cover;
    }
    .metric-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e10600;
    }
    h1, h2, h3 {
        color: #e10600 !important;
        font-family: 'Arial Black', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    # Load models and encoders
    try:
        models = {
            'Model A (Baseline)': tf.keras.models.load_model('models/Model_A.keras'),
            'Model B (Complex)': tf.keras.models.load_model('models/Model_B.keras'),
            'Model C (Regularized)': tf.keras.models.load_model('models/Model_C.keras')
        }
        le_driver = joblib.load('models/le_driver.pkl')
        le_gp = joblib.load('models/le_grand_prix.pkl')
        le_car = joblib.load('models/le_car.pkl')
        scaler = joblib.load('models/scaler.pkl')
        training_results = joblib.load('models/training_results.pkl')
        return models, le_driver, le_gp, le_car, scaler, training_results
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure training is complete.")
        return None

def main():
    st.title("🏎️ F1 Podium Prediction Dashboard")
    st.markdown("### High-Performance Multi-ANN Race Outcome Simulation")
    
    assets = load_assets()
    if not assets:
        st.info("Models not found. Run training script first.")
        return
        
    models, le_driver, le_gp, le_car, scaler, training_results = assets

    # Sidebar
    st.sidebar.image("assets/images/background.png", use_container_width=True)
    st.sidebar.header("Settings")
    selected_model_name = st.sidebar.selectbox("Select ANN Architecture", list(models.keys()))
    model = models[selected_model_name]

    # Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏁 Race Parameters")
        year = st.slider("Season Year", 1950, 2024, 2024)
        gp = st.selectbox("Grand Prix", le_gp.classes_)
        driver = st.selectbox("Driver", le_driver.classes_)
        car = st.selectbox("Constructor (Car)", le_car.classes_)
        
    with col2:
        st.subheader("⏱️ Qualifying Data")
        grid = st.number_input("Grid Position", 1, 22, 1)
        qualifying = st.number_input("Qualifying Position", 1, 22, 1)
        
        if st.button("🏁 Predict Podium Probability"):
            # Prepare Input
            input_df = pd.DataFrame({
                'Year': [year],
                'gp_encoded': [le_gp.transform([gp])[0]],
                'driver_encoded': [le_driver.transform([driver])[0]],
                'car_encoded': [le_car.transform([car])[0]],
                'grid_position': [grid],
                'qualifying_position': [qualifying]
            })
            
            # Scale
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prob = model.predict(input_scaled, verbose=0)[0][0]
            
            # Display Result
            st.markdown("---")
            st.subheader("🏆 Prediction Result")
            
            prob_pct = float(prob) * 100
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Podium Probability (%)", 'font': {'size': 24, 'color': "#e10600"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#e10600"},
                    'bgcolor': "#1f2937",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#3b0000'},
                        {'range': [30, 70], 'color': '#7a0000'},
                        {'range': [70, 100], 'color': '#e10600'}
                    ],
                }
            ))
            fig.update_layout(paper_bgcolor="#0e1117", font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
            
            if prob > 0.7:
                st.success(f"🔥 **HIGH CHANCE!** {driver} is likely to finish on the podium!")
            elif prob > 0.4:
                st.warning(f"⚖️ **MODERATE CHANCE.** A podium finish is possible for {driver}.")
            else:
                st.error(f"🧊 **LOW CHANCE.** {driver} is unlikely to reach the podium.")

    # Performance Section
    st.markdown("---")
    st.subheader("📊 Model Performance Analysis")
    
    # Metrics Comparison Table
    comp_df = pd.DataFrame({
        name: results['metrics'] for name, results in training_results.items()
    }).T
    comp_df.index.name = "Model"

    st.dataframe(
        comp_df.style.format({
            'accuracy': '{:.1%}', 'precision': '{:.1%}',
            'recall':   '{:.1%}', 'f1':        '{:.1%}', 'auc': '{:.4f}',
        }).highlight_max(axis=0, color='#7a0000'),
        use_container_width=True
    )

    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        # Metrics Bar Chart
        comp_df = pd.DataFrame({
            name: results['metrics'] for name, results in training_results.items()
        }).T
        
        fig_metrics = px.bar(comp_df, barmode='group', title="Model Metrics Comparison",
                             color_discrete_sequence=['#e10600', '#ffffff', '#7a7a7a', '#ff4d4d', '#333333'])
        fig_metrics.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_metrics, use_container_width=True)
        
    with p_col2:
        # Loss Curves
        fig_loss = go.Figure()
        for name, results in training_results.items():
            fig_loss.add_trace(go.Scatter(y=results['loss_curve'], mode='lines', name=f"{name} Loss"))
            
        fig_loss.update_layout(title="Training Loss Curves", paper_bgcolor="#0e1117", 
                               plot_bgcolor="#0e1117", font={'color': "white"})
        st.plotly_chart(fig_loss, use_container_width=True)

if __name__ == "__main__":
    main()
