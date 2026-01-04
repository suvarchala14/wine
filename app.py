import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #722F37;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #8B3A47;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #722F37;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0;
    }
    h2 {
        color: #722F37;
    }
    h3 {
        color: #8B3A47;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        scaler = pickle.load(open('scaler_model.sav', 'rb'))
        model = pickle.load(open('dtc_model.sav', 'rb'))
        return scaler, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first by running the notebook.")
        return None, None

# Load data for visualization
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("winequality-red.csv")
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some visualizations will be unavailable.")
        return None

# Quality interpretation
def interpret_quality(quality):
    if quality <= 4:
        return "Poor Quality 😞", "#e74c3c", "This wine has poor characteristics."
    elif quality == 5:
        return "Below Average 😐", "#e67e22", "This wine is below average quality."
    elif quality == 6:
        return "Average Quality 🙂", "#f39c12", "This wine has average characteristics."
    elif quality == 7:
        return "Good Quality 😊", "#27ae60", "This wine has good characteristics!"
    else:
        return "Excellent Quality 🤩", "#2ecc71", "This is an excellent wine!"

# Main app
def main():
    # Header
    st.markdown("<h1>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict wine quality using advanced machine learning</p>", unsafe_allow_html=True)
    
    # Load models
    scaler, model = load_models()
    df = load_data()
    
    if scaler is None or model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=150)
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select Page:", ["🔮 Predict Quality", "📊 Data Insights", "ℹ️ About"])
    
    if page == "🔮 Predict Quality":
        prediction_page(scaler, model)
    elif page == "📊 Data Insights":
        insights_page(df)
    else:
        about_page()

def prediction_page(scaler, model):
    st.header("🔮 Predict Wine Quality")
    st.write("Enter the wine characteristics below to predict its quality rating (3-8 scale)")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧪 Acidity & Sugar")
        fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1, 
                                  help="Most acids involved with wine (g/dm³)")
        volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7, 0.01,
                                     help="Amount of acetic acid (g/dm³)")
        citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, 0.01,
                               help="Adds freshness and flavor (g/dm³)")
        residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 2.5, 0.1,
                                   help="Amount of sugar remaining (g/dm³)")
    
    with col2:
        st.subheader("🧂 Salts & Sulfur")
        chlorides = st.slider("Chlorides", 0.01, 0.61, 0.08, 0.001,
                             help="Amount of salt (g/dm³)")
        free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 72, 11, 1,
                                        help="Free form of SO2 (mg/dm³)")
        total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6, 289, 34, 1,
                                         help="Total amount of SO2 (mg/dm³)")
        sulphates = st.slider("Sulphates", 0.3, 2.0, 0.66, 0.01,
                             help="Wine additive (g/dm³)")
    
    # Create third row for remaining features
    col3, col4, col5 = st.columns(3)
    
    with col3:
        density = st.slider("Density", 0.990, 1.004, 0.996, 0.0001,
                           help="Density of wine (g/cm³)")
    
    with col4:
        pH = st.slider("pH Level", 2.7, 4.0, 3.3, 0.01,
                      help="Acidity level (0-14 scale)")
    
    with col5:
        alcohol = st.slider("Alcohol Content", 8.0, 15.0, 10.4, 0.1,
                           help="Alcohol percentage (%)")
    
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the predict button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🎯 Predict Wine Quality", use_container_width=True)
    
    if predict_button:
        # Apply log transformation to match training preprocessing
        residual_sugar_log = np.log1p(residual_sugar)
        chlorides_log = np.log1p(chlorides)
        free_sulfur_dioxide_log = np.log1p(free_sulfur_dioxide)
        total_sulfur_dioxide_log = np.log1p(total_sulfur_dioxide)
        sulphates_log = np.log1p(sulphates)
        
        # Create feature dictionary
        feature_values = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar_log,
            'chlorides': chlorides_log,
            'free sulfur dioxide': free_sulfur_dioxide_log,
            'total sulfur dioxide': total_sulfur_dioxide_log,
            'density': density,
            'pH': pH,
            'sulphates': sulphates_log,
            'alcohol': alcohol
        }
        
        # Create DataFrame
        feature_names = list(feature_values.keys())
        user_input = pd.DataFrame([feature_values], columns=feature_names)
        
        # Scale the input
        user_input_scaled = scaler.transform(user_input)
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        
        # Display result with animation
        st.markdown("<br>", unsafe_allow_html=True)
        
        label, color, description = interpret_quality(prediction)
        
        # Create a beautiful result card
        st.markdown(f"""
            <div class='prediction-box'>
                <h2 style='color: white; margin: 0;'>Predicted Wine Quality</h2>
                <h1 style='color: white; font-size: 5rem; margin: 1rem 0;'>{int(prediction)}</h1>
                <h3 style='color: white; margin: 0;'>{label}</h3>
                <p style='color: white; margin-top: 1rem; font-size: 1.1rem;'>{description}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a gauge chart
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Quality Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [3, 8], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [3, 4], 'color': '#ffcccc'},
                    {'range': [4, 5], 'color': '#ffe6cc'},
                    {'range': [5, 6], 'color': '#ffffcc'},
                    {'range': [6, 7], 'color': '#ccffcc'},
                    {'range': [7, 8], 'color': '#ccffcc'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 6.5
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "#722F37", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance visualization (if available)
        if hasattr(model, 'feature_importances_'):
            st.markdown("### 📊 Feature Importance")
            st.write("How much each feature contributed to this prediction:")
            
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance in Model',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            fig_importance.update_layout(
                height=500,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)

def insights_page(df):
    st.header("📊 Data Insights & Visualizations")
    
    if df is None:
        st.error("Dataset not available for visualization.")
        return
    
    # Display dataset statistics
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Samples", f"{len(df):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Features", f"{len(df.columns) - 1}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg Quality", f"{df['quality'].mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Quality Range", f"{df['quality'].min()} - {df['quality'].max()}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quality distribution
    st.subheader("🍷 Wine Quality Distribution")
    
    quality_counts = df['quality'].value_counts().sort_index()
    
    fig_quality = go.Figure(data=[
        go.Bar(
            x=quality_counts.index,
            y=quality_counts.values,
            marker=dict(
                color=quality_counts.values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            text=quality_counts.values,
            textposition='auto',
        )
    ])
    
    fig_quality.update_layout(
        title="Distribution of Wine Quality Ratings",
        xaxis_title="Quality Rating",
        yaxis_title="Number of Wines",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig_quality, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    
    corr_matrix = df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig_corr.update_layout(
        title="Feature Correlation Matrix",
        height=700,
        xaxis={'side': 'bottom'},
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature distributions
    st.subheader("📊 Feature Distributions")
    
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        options=[col for col in df.columns if col != 'quality']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df,
            x=selected_feature,
            color='quality',
            marginal='box',
            title=f'Distribution of {selected_feature}',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        fig_box = px.box(
            df,
            x='quality',
            y=selected_feature,
            color='quality',
            title=f'{selected_feature} by Quality Rating',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical summary
    st.subheader("📋 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

def about_page():
    st.header("ℹ️ About Wine Quality Predictor")
    
    st.markdown("""
    ### 🍷 Welcome to the Wine Quality Prediction App!
    
    This application uses **Machine Learning** to predict the quality of red wine based on its physicochemical properties.
    
    #### 🎯 How it works:
    1. **Data Collection**: The model was trained on a dataset of red wines with 11 features
    2. **Preprocessing**: Features are normalized and transformed for optimal performance
    3. **Model Training**: Decision Tree Classifier achieves high accuracy in quality prediction
    4. **Prediction**: Enter wine characteristics to get instant quality predictions
    
    #### 📊 Features Used:
    - **Fixed Acidity**: Tartaric acid content
    - **Volatile Acidity**: Acetic acid content (vinegar taste)
    - **Citric Acid**: Adds freshness and flavor
    - **Residual Sugar**: Remaining sugar after fermentation
    - **Chlorides**: Salt content
    - **Free Sulfur Dioxide**: Prevents microbial growth
    - **Total Sulfur Dioxide**: Total SO2 content
    - **Density**: Wine density (related to alcohol and sugar)
    - **pH**: Acidity/basicity level
    - **Sulphates**: Wine additive for antimicrobial properties
    - **Alcohol**: Alcohol percentage by volume
    
    #### 🏆 Model Performance:
    - **Algorithm**: Decision Tree Classifier
    - **Preprocessing**: StandardScaler normalization + SMOTE balancing
    - **Quality Scale**: 3-8 (Poor to Excellent)
    
    #### 🔬 Technology Stack:
    - **Python** - Core programming language
    - **Scikit-learn** - Machine learning library
    - **Streamlit** - Web application framework
    - **Plotly** - Interactive visualizations
    - **Pandas & NumPy** - Data manipulation
    
    #### 👨‍💻 Made with ❤️ for Wine Enthusiasts
    
    ---
    
    *This application is for educational and entertainment purposes. Professional wine evaluation requires expert tasting and analysis.*
    """)
    
    # Add some metrics in columns
    st.markdown("### 📈 Quick Stats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**11** Input Features")
    with col2:
        st.success("**6** Quality Levels")
    with col3:
        st.warning("**ML-Powered** Predictions")

if __name__ == "__main__":
    main()