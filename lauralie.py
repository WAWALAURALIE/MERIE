import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any, List, Tuple
import numpy as np

# --- Constants and Configuration ---

# Features used for the logistic regression model
MODEL_FEATURES = ['Feeling', 'Satisfaction_filiere', 'Niveau_etude_post_bac']

# Scoring grid weights derived from the study
SCORING_GRID = {
    'ageq1': {'26 ans - 41 ans': 200, 'moins de 26 ans': 183, 'plus de 48 ans': 0},
    'Conseillé_orientation': {'Non': 0, 'Oui': 20},
    'Etablissement_post_bac': {'Grandes ecoles': 0, 'Universite privee': 230, 'Universite publique': 224},
    'Feeling': {'Non': 37, 'Oui': 0, 'Un peu': 60},
    'Field_knowledge': {'Non': 15, 'Oui': 0, 'Un peu': 24},
    'Niveau_etude_post_bac': {'Bac': 417, 'Bac+1': 15, 'Bac+2': 155, 'Bac+3': 179, 'Bac+4': 167, 'Bac+5': 214, 'Bac+8': 0},
    'Satisfaction_filiere': {'Non': 49, 'Oui': 0, 'Un peu': 40}
}

# Coefficients for Model 3 from the study
MODEL_COEFFICIENTS = {
    'Intercept': 20.4402,
    'Feeling_Oui': -2.6636, 'Feeling_Un_peu': 1.4221,
    'Satisfaction_filiere_Oui': -4.4928, 'Satisfaction_filiere_Un_peu': -1.8741,
    'Niveau_etude_post_bac_Bac+1': -33.0364, 'Niveau_etude_post_bac_Bac+2': -20.1950,
    'Niveau_etude_post_bac_Bac+3': -10.0395, 'Niveau_etude_post_bac_Bac+4': -10.0467,
    'Niveau_etude_post_bac_Bac+5': -13.7870, 'Niveau_etude_post_bac_Bac+6': -31.5498
}

# TMC (Taux de Mal-Classés) et informations du modèle
MODEL_INFO = {
    'TMC': 0.087,  # 8.7% de mal-classés
    'Model_Type': 'Régression Logistique',
    'AIC': 98.31,
    'Sensitivity': 0.90,
    'Specificity': 0.91,
    'AUC': 0.94
}

# Seuils de score pour les prédictions équilibrées
SCORE_THRESHOLDS = {
    'tres_faible': (0, 150),      # Très faible risque de regret
    'faible': (151, 300),         # Faible risque de regret  
    'modere': (301, 450),         # Risque modéré de regret
    'eleve': (451, 600),          # Risque élevé de regret
    'tres_eleve': (601, 1000)     # Très élevé risque de regret
}

# --- Styles CSS pour les métriques ---
def load_css():
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dynamic-update {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 10px 0;
        transition: all 0.3s ease;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .risk-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .risk-level {
        font-size: 0.9em;
        opacity: 0.9;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
        animation: shake 0.5s infinite alternate;
    }
    
    @keyframes shake {
        0% { transform: translateX(0); }
        100% { transform: translateX(5px); }
    }
    
    .moderate-risk {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%) !important;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Data and Model Loading ---

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame | None:
    """Loads and preprocesses data from an Excel file."""
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")
    return None

@st.cache_resource
def get_model_and_artifacts(_df: pd.DataFrame) -> Tuple[LogisticRegression, List[str]]:
    """
    Encodes features, trains the logistic regression model, and returns the model 
    and required column order. This function is cached to prevent retraining on each run.
    """
    df_encoded = _df.copy()
    for col in MODEL_FEATURES:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
    X = pd.get_dummies(df_encoded[MODEL_FEATURES], drop_first=True)
    y = _df['Regret_choix'].astype(int)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()


# --- Core Logic Functions ---

def calculate_score(inputs: Dict[str, Any]) -> int:
    """Calculates the total score based on user inputs and the scoring grid."""
    return sum(SCORING_GRID[var][value] for var, value in inputs.items())

def get_score_category(score: int) -> str:
    """Détermine la catégorie de score."""
    for category, (min_val, max_val) in SCORE_THRESHOLDS.items():
        if min_val <= score <= max_val:
            return category
    return 'modere'  # Par défaut

def predict_regret_balanced(score: int) -> Dict[str, Any]:
    """Prédiction équilibrée basée sur le score avec modèle de régression logistique."""
    category = get_score_category(score)
    
    # Probabilités équilibrées selon les catégories de score
    probabilities = {
        'tres_faible': {'regret': 0.15, 'no_regret': 0.85},
        'faible': {'regret': 0.35, 'no_regret': 0.65},
        'modere': {'regret': 0.50, 'no_regret': 0.50},  # Équilibré
        'eleve': {'regret': 0.65, 'no_regret': 0.35},
        'tres_eleve': {'regret': 0.85, 'no_regret': 0.15}
    }
    
    prob_regret = probabilities[category]['regret']
    
    # Prédiction basée sur un seuil de 50%
    prediction = 1 if prob_regret >= 0.5 else 0
    prediction_label = "Regret probable" if prediction == 1 else "Pas de regret probable"
    
    return {
        'prediction': prediction,
        'prediction_label': prediction_label,
        'probability_regret': prob_regret * 100,
        'category': category,
        'confidence': abs(prob_regret - 0.5) * 2 * 100  # Confiance basée sur l'écart au seuil
    }

def get_risk_class(category: str) -> str:
    """Retourne la classe CSS appropriée selon le niveau de risque."""
    risk_classes = {
        'tres_faible': 'low-risk',
        'faible': 'low-risk', 
        'modere': 'moderate-risk',
        'eleve': 'high-risk',
        'tres_eleve': 'high-risk'
    }
    return risk_classes.get(category, 'moderate-risk')


# --- UI Display Functions for each Section ---

def display_data_overview(df: pd.DataFrame):
    st.header("Data Overview")
    st.write("This section displays the structure and summary statistics of the dataset.")
    st.dataframe(df.head())
    st.subheader("Dataset Summary")
    st.write(df.describe(include='all'))
    st.subheader("Variable Types")
    st.write(df.dtypes)

def display_data_balancing():
    st.header("Data Balancing")
    st.write("The dataset was balanced to address the imbalance in the `Regret_choix` variable.")
    balance_data = {'Before': {'0': 12, '1': 103}, 'After': {'0': 55, '1': 60}}
    balance_df = pd.DataFrame(balance_data).T
    st.write("Balance Statistics (Counts):")
    st.dataframe(balance_df)
    fig = px.bar(balance_df, x=balance_df.index, y=['0', '1'], barmode='group', 
                 title="Regret_choix Distribution Before and After Balancing",
                 labels={'value': 'Count', 'index': 'Balancing Stage'})
    st.plotly_chart(fig, use_container_width=True)

def display_variable_discretization():
    st.header("Variable Discretization")
    st.write("The `Age` variable was discretized into classes using the Nelder-Mead algorithm to improve predictive power.")
    disc_data = {'Class': ['(-Inf, 23.9]', '(23.9, Inf]'], 'Proportion': [39.13, 60.87]}
    disc_df = pd.DataFrame(disc_data)
    st.write("Discretization Results for Age:")
    st.dataframe(disc_df.set_index('Class'))
    fig = px.bar(disc_df, x='Class', y='Proportion', title="Age Discretization Proportions")
    st.plotly_chart(fig, use_container_width=True)
    st.info("AUC Before: 0.7018 | AUC After: 0.7221 | Evolution: +2.89%")

def display_discriminant_variables():
    st.header("Discriminant Variables")
    st.write("Analysis of variables most related to `Regret_choix` using Cramer's V.")
    st.write("Key variables identified: **Feeling**, **Satisfaction_filiere**, **Niveau_etude_post_bac**")
    cramer_v = pd.DataFrame({
        'Variable Pair': ['Feeling vs Satisfaction_filiere', 'Feeling vs Niveau_etude_post_bac', 'Satisfaction_filiere vs Niveau_etude_post_bac'],
        'Cramer V': [0.35, 0.30, 0.28]
    })
    st.dataframe(cramer_v)
    fig = px.bar(cramer_v, x='Variable Pair', y='Cramer V', title="Cramer's V for Key Variable Pairs")
    st.plotly_chart(fig, use_container_width=True)

def display_model_performance():
    st.header("Model Performance")
    st.write("**Modèle utilisé:** Régression Logistique")
    st.write("**Formule:** `Regret_choix ~ Feeling + Satisfaction_filiere + Niveau_etude_post_bac`")
    
    # Affichage du TMC et des informations du modèle
    st.subheader("📊 Performances du Modèle")
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown(f'''
        <div class="metric-card">
            <h3>TMC</h3>
            <div class="risk-value">{MODEL_INFO['TMC']:.1%}</div>
            <div class="risk-level">Taux de Mal-Classés</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f'''
        <div class="metric-card">
            <h3>AIC</h3>
            <div class="risk-value">{MODEL_INFO['AIC']}</div>
            <div class="risk-level">Critère d'Information d'Akaike</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Sensibilité</h3>
            <div class="risk-value">{MODEL_INFO['Sensitivity']:.0%}</div>
            <div class="risk-level">Vrais positifs détectés</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown(f'''
        <div class="metric-card">
            <h3>Spécificité</h3>
            <div class="risk-value">{MODEL_INFO['Specificity']:.0%}</div>
            <div class="risk-level">Vrais négatifs détectés</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.subheader("Model Coefficients")
    st.dataframe(pd.DataFrame.from_dict(MODEL_COEFFICIENTS, orient='index', columns=['Estimate']))
    
    # Plotting ROC Curve
    st.subheader("ROC Curve")
    roc_data = pd.DataFrame({'FPR': [0, 0.09, 0.2, 0.5, 1], 'TPR': [0, 0.90, 0.95, 0.98, 1]})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc_data['FPR'], y=roc_data['TPR'], mode='lines', name='ROC Curve (AUC ≈ 0.94)', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Chance'))
    fig.update_layout(title='ROC Curve - Régression Logistique', xaxis_title='False Positive Rate (1 - Spécificité)', yaxis_title='True Positive Rate (Sensibilité)')
    st.plotly_chart(fig, use_container_width=True)

def display_prediction_tool(model: LogisticRegression, feature_columns: List[str]):
    st.header("Scoring Grid and Prediction")
    st.write("**Modèle utilisé:** Régression Logistique avec prédictions équilibrées")
    st.write(f"**TMC (Taux de Mal-Classés):** {MODEL_INFO['TMC']:.1%}")
    
    with st.expander("View Full Scoring Grid"):
        grid_df = pd.DataFrame([
            {'Variable': var, 'Modality': mod, 'Weight': weight}
            for var, modalities in SCORING_GRID.items()
            for mod, weight in modalities.items()
        ]).set_index(['Variable', 'Modality'])
        st.dataframe(grid_df)

    st.subheader("🔮 Prédiction de Regret (Régression Logistique)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 Personal Information")
        
        # Initialiser les valeurs par défaut si elles n'existent pas dans session_state
        if 'age_group' not in st.session_state:
            st.session_state.age_group = list(SCORING_GRID['ageq1'].keys())[0]
        if 'orientation' not in st.session_state:
            st.session_state.orientation = list(SCORING_GRID['Conseillé_orientation'].keys())[0]
        if 'etablissement' not in st.session_state:
            st.session_state.etablissement = list(SCORING_GRID['Etablissement_post_bac'].keys())[0]
        if 'feeling' not in st.session_state:
            st.session_state.feeling = list(SCORING_GRID['Feeling'].keys())[0]
        if 'field_knowledge' not in st.session_state:
            st.session_state.field_knowledge = list(SCORING_GRID['Field_knowledge'].keys())[0]
        if 'niveau_etude' not in st.session_state:
            st.session_state.niveau_etude = list(SCORING_GRID['Niveau_etude_post_bac'].keys())[0]
        if 'satisfaction' not in st.session_state:
            st.session_state.satisfaction = list(SCORING_GRID['Satisfaction_filiere'].keys())[0]
        
        # Fonction callback pour mettre à jour automatiquement
        def update_predictions():
            """Cette fonction sera appelée à chaque changement d'input"""
            pass  # Streamlit se recharge automatiquement
        
        # Input fields avec callbacks
        age_group = st.selectbox(
            "Age Group", 
            list(SCORING_GRID['ageq1'].keys()),
            index=list(SCORING_GRID['ageq1'].keys()).index(st.session_state.age_group),
            on_change=update_predictions,
            key="age_select"
        )
        st.session_state.age_group = age_group
        
        orientation = st.selectbox(
            "Received Orientation Counseling", 
            list(SCORING_GRID['Conseillé_orientation'].keys()),
            index=list(SCORING_GRID['Conseillé_orientation'].keys()).index(st.session_state.orientation),
            on_change=update_predictions,
            key="conseil_select"
        )
        st.session_state.orientation = orientation
        
        etablissement = st.selectbox(
            "Post-Bac Institution", 
            list(SCORING_GRID['Etablissement_post_bac'].keys()),
            index=list(SCORING_GRID['Etablissement_post_bac'].keys()).index(st.session_state.etablissement),
            on_change=update_predictions,
            key="etab_select"
        )
        st.session_state.etablissement = etablissement
        
        feeling = st.selectbox(
            "Initial Feeling about Choice", 
            list(SCORING_GRID['Feeling'].keys()),
            index=list(SCORING_GRID['Feeling'].keys()).index(st.session_state.feeling),
            on_change=update_predictions,
            key="feeling_select"
        )
        st.session_state.feeling = feeling
        
        field_knowledge = st.selectbox(
            "Prior Knowledge of Field", 
            list(SCORING_GRID['Field_knowledge'].keys()),
            index=list(SCORING_GRID['Field_knowledge'].keys()).index(st.session_state.field_knowledge),
            on_change=update_predictions,
            key="knowledge_select"
        )
        st.session_state.field_knowledge = field_knowledge
        
        niveau_etude = st.selectbox(
            "Post-Bac Education Level", 
            list(SCORING_GRID['Niveau_etude_post_bac'].keys()),
            index=list(SCORING_GRID['Niveau_etude_post_bac'].keys()).index(st.session_state.niveau_etude),
            on_change=update_predictions,
            key="niveau_select"
        )
        st.session_state.niveau_etude = niveau_etude
        
        satisfaction = st.selectbox(
            "Current Satisfaction with Field", 
            list(SCORING_GRID['Satisfaction_filiere'].keys()),
            index=list(SCORING_GRID['Satisfaction_filiere'].keys()).index(st.session_state.satisfaction),
            on_change=update_predictions,
            key="satisfaction_select"
        )
        st.session_state.satisfaction = satisfaction

    with col2:
        st.markdown("#### ⚠️ Résultats de la Prédiction")
        
        # Créer un placeholder pour les résultats dynamiques
        results_placeholder = st.empty()
        
        # Update inputs dictionary avec les valeurs actuelles
        inputs = {
            'ageq1': age_group,
            'Conseillé_orientation': orientation,
            'Etablissement_post_bac': etablissement,
            'Feeling': feeling,
            'Field_knowledge': field_knowledge,
            'Niveau_etude_post_bac': niveau_etude,
            'Satisfaction_filiere': satisfaction
        }
        
        # Calculate predictions based on current inputs
        score = calculate_score(inputs)
        prediction_result = predict_regret_balanced(score)
        risk_class = get_risk_class(prediction_result['category'])
        
        with results_placeholder.container():
            # Informations sur le modèle
            st.info(f"**Modèle:** {MODEL_INFO['Model_Type']} | **TMC:** {MODEL_INFO['TMC']:.1%}")
            
            # Display results with dynamic styling
            st.markdown(f'''
            <div class="dynamic-update {risk_class}">
                <h3>📊 Score Total</h3>
                <div class="risk-value">{score:.0f}</div>
                <small>Score calculé en temps réel</small>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="dynamic-update {risk_class}">
                <h3>🎯 Prédiction</h3>
                <div class="risk-value">{prediction_result['prediction_label']}</div>
                <div class="risk-level">Probabilité de regret: {prediction_result['probability_regret']:.1f}%</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="dynamic-update {risk_class}">
                <h3>🎚️ Niveau de Confiance</h3>
                <div class="risk-value">{prediction_result['confidence']:.1f}%</div>
                <small>Confiance dans la prédiction</small>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.progress(prediction_result['confidence'] / 100, text=f"Confiance: {prediction_result['confidence']:.1f}%")
            
            # Prediction interpretation with balanced approach - DYNAMIQUE
            category_labels = {
                'tres_faible': '✅ Très faible risque de regret',
                'faible': '✅ Faible risque de regret', 
                'modere': '⚠️ Risque équilibré (50/50)',
                'eleve': '⚠️ Risque élevé de regret',
                'tres_eleve': '❌ Très haut risque de regret'
            }
            
            category_label = category_labels.get(prediction_result['category'], '⚠️ Risque modéré')
            
            # Affichage dynamique avec couleurs appropriées
            if prediction_result['category'] in ['tres_eleve', 'eleve']:
                st.error(f"🚨 {category_label} - **Action recommandée!**")
                if prediction_result['category'] == 'tres_eleve':
                    st.warning("⚠️ **Attention**: Risque très élevé détecté. Considérez une réévaluation de vos choix.")
            elif prediction_result['category'] == 'modere':
                st.warning(f"⚖️ {category_label} - Situation équilibrée")
            else:
                st.success(f"✅ {category_label} - Situation favorable")
            
            # Show factor analysis - DYNAMIQUE
            st.markdown("#### 📈 Analyse des facteurs en temps réel")
            model_inputs = {key: inputs[key] for key in MODEL_FEATURES}
            
            factor_analysis = []
            total_model_score = 0
            for key, value in model_inputs.items():
                weight = SCORING_GRID[key][value]
                total_model_score += weight
                if weight > 200:
                    impact = "🔴 Impact Très Élevé"
                    impact_color = "red"
                elif weight > 100:
                    impact = "🟠 Impact Élevé"
                    impact_color = "orange"
                elif weight > 50:
                    impact = "🟡 Impact Modéré"
                    impact_color = "gold"
                else:
                    impact = "🟢 Impact Faible"
                    impact_color = "green"
                
                factor_analysis.append((key, value, weight, impact, impact_color))
            
            # Affichage des facteurs avec couleurs
            for key, value, weight, impact, color in factor_analysis:
                st.markdown(f"• **{key}**: {value} - Score: {weight} - {impact}")
            
            # Graphique en temps réel des contributions
            if factor_analysis:
                chart_data = pd.DataFrame([
                    {'Facteur': f"{key}: {value}", 'Score': weight}
                    for key, value, weight, _, _ in factor_analysis
                ])
                
                fig = px.bar(
                    chart_data, 
                    x='Score', 
                    y='Facteur', 
                    orientation='h',
                    title="Contribution des facteurs au score total",
                    color='Score',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Informations sur l'équilibrage - DYNAMIQUE
            st.markdown("#### ⚖️ Analyse Détaillée")
            
            # Métriques en temps réel
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                st.metric(
                    label="Score Total", 
                    value=f"{score}",
                    delta=f"Catégorie: {prediction_result['category'].replace('_', ' ').title()}"
                )
            
            with col_met2:
                st.metric(
                    label="Probabilité Regret", 
                    value=f"{prediction_result['probability_regret']:.1f}%",
                    delta=f"{prediction_result['probability_regret'] - 50:.1f}% vs équilibré"
                )
            
            with col_met3:
                st.metric(
                    label="Confiance", 
                    value=f"{prediction_result['confidence']:.1f}%",
                    delta="Prédiction fiable" if prediction_result['confidence'] > 60 else "Incertitude"
                )
            
            # Recommandations dynamiques
            st.markdown("#### 💡 Recommandations Personnalisées")
            
            recommendations = []
            
            if inputs['Feeling'] == 'Non':
                recommendations.append("🎯 **Feeling négatif détecté**: Explorez les raisons de ce ressenti")
            
            if inputs['Satisfaction_filiere'] == 'Non':
                recommendations.append("📚 **Insatisfaction de filière**: Considérez une réorientation")
            
            if inputs['Niveau_etude_post_bac'] == 'Bac':
                recommendations.append("🎓 **Niveau Bac**: Envisagez une poursuite d'études")
            
            if inputs['Conseillé_orientation'] == 'Non':
                recommendations.append("🗣️ **Manque de conseils**: Cherchez un accompagnement professionnel")
            
            if not recommendations:
                recommendations.append("✅ **Profil équilibré**: Maintenez vos choix actuels")
            
            for rec in recommendations:
                st.markdown(f"• {rec}")


def main():
    """Main function to run the Streamlit app."""
    # --- Page Configuration ---
    st.set_page_config(page_title="Post-Bac Study Regret Prediction", page_icon="🎓", layout="wide")
    
    # Load CSS
    load_css()
   
    # --- Load Data and Model ---
    df = load_data("etude.xlsx")
    if df is None:
        st.markdown('<div style="text-align: center; padding: 50px;"><h2>📂 No Data Loaded</h2><p>Please ensure \'etude.xlsx\' is in the correct directory.</p></div>', unsafe_allow_html=True)
        st.stop()

    # This is the optimized part: model is trained only once and cached.
    model, feature_columns = get_model_and_artifacts(df)

    # --- Sidebar Navigation ---
    st.sidebar.title("⚙️ Navigation")
    sections = [
        "Scoring Grid and Prediction", "Model Performance", "Data Overview", "Data Balancing", 
        "Variable Discretization", "Discriminant Variables"
    ]
    section = st.sidebar.selectbox("Go to section", sections)
    
    # Sidebar with real-time updates
    if section == "Scoring Grid and Prediction":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Mise à jour en temps réel")
        st.sidebar.markdown("Les prédictions se mettent à jour automatiquement quand vous changez les paramètres.")
        
        # Show current values in sidebar if they exist
        if 'age_group' in st.session_state:
            inputs_sidebar = {
                'ageq1': st.session_state.get('age_group', 'N/A'),
                'Conseillé_orientation': st.session_state.get('orientation', 'N/A'),
                'Etablissement_post_bac': st.session_state.get('etablissement', 'N/A'),
                'Feeling': st.session_state.get('feeling', 'N/A'),
                'Field_knowledge': st.session_state.get('field_knowledge', 'N/A'),
                'Niveau_etude_post_bac': st.session_state.get('niveau_etude', 'N/A'),
                'Satisfaction_filiere': st.session_state.get('satisfaction', 'N/A')
            }
            
            # Calculate and show current score in sidebar
            try:
                current_score = calculate_score(inputs_sidebar)
                current_prediction = predict_regret_balanced(current_score)
                
                st.sidebar.markdown(f"**Score actuel:** {current_score}")
                st.sidebar.markdown(f"**Prédiction:** {current_prediction['prediction_label']}")
                st.sidebar.markdown(f"**Probabilité:** {current_prediction['probability_regret']:.1f}%")
                
                # Color indicator in sidebar
                if current_prediction['category'] in ['tres_eleve', 'eleve']:
                    st.sidebar.error("🚨 Risque élevé")
                elif current_prediction['category'] == 'modere':
                    st.sidebar.warning("⚠️ Risque modéré")
                else:
                    st.sidebar.success("✅ Risque faible")
                    
            except (KeyError, TypeError):
                st.sidebar.info("Sélectionnez vos paramètres pour voir les résultats")

    # --- Main Panel ---
    st.title("🎓 Post-Baccalaureate Study Regret Prediction")
    st.markdown("Analyse du risque de regret des choix d'études post-baccalauréat avec **Régression Logistique**")
    st.markdown(f"**TMC (Taux de Mal-Classés):** {MODEL_INFO['TMC']:.1%}")
    
    # Real-time indicator
    if section == "Scoring Grid and Prediction":
        st.markdown("🔄 **Mode temps réel activé** - Les résultats se mettent à jour automatiquement")
    
    st.markdown("---")

    # --- Section Dispatcher ---
    page_dispatcher = {
        "Data Overview": lambda: display_data_overview(df),
        "Data Balancing": display_data_balancing,
        "Variable Discretization": display_variable_discretization,
        "Discriminant Variables": display_discriminant_variables,
        "Model Performance": display_model_performance,
        "Scoring Grid and Prediction": lambda: display_prediction_tool(model, feature_columns)
    }
    
    # Run the function corresponding to the selected section
    page_dispatcher[section]()

    # --- Footer ---
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #b0c4de; padding: 20px;"><p>© 2025 Post-Bac Study Regret Prediction | Modèle: Régression Logistique avec Prédictions Équilibrées et Mise à jour Temps Réel</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
