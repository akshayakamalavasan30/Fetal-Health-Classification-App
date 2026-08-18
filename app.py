import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

#Page Config
st.set_page_config(page_title="Fetal Health Classifier", layout="wide")

#Styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #1f3c88;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 20px;
}

/* Highlight box */
.highlight {
    border: 2px solid #1f77b4;
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

#Title
st.markdown('<div class="main-title">🩺 Fetal Health Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload → Select Model → Analyze</div>', unsafe_allow_html=True)

#Load
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("models/Logistic Regression.pkl"),
        "Decision Tree": joblib.load("models/Decision Tree.pkl"),
        "KNN": joblib.load("models/KNN.pkl"),
        "Naive Bayes": joblib.load("models/Naive Bayes.pkl"),
        "Random Forest": joblib.load("models/Random Forest.pkl")
    }

def load_scaler():
    return joblib.load("models/scaler.pkl")

models = load_models()
scaler = load_scaler()

#Uploading the dataset
st.subheader("📂 Upload Dataset")
uploaded_file = st.file_uploader("Uploaded CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload dataset to continue")
    st.stop()

data = pd.read_csv(uploaded_file)

#Preview
st.subheader("📊 Dataset Preview")
st.dataframe(data.head(), use_container_width=True)

st.markdown("---")  
#Prepare
if "fetal_health" in data.columns:
    X_input = data.drop("fetal_health", axis=1)
    y_true = data["fetal_health"]
else:
    X_input = data
    y_true = None

#Scale
try:
    X_scaled = scaler.transform(X_input)
except:
    st.error("Column mismatch")
    st.stop()

#Model Comparison
st.markdown("### 📊 Model Comparison (All Models)")

comparison_results = []

if "fetal_health" in data.columns:
    for name, mdl in models.items():
        preds = mdl.predict(X_scaled)

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')

        comparison_results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1
        })

    comp_df = pd.DataFrame(comparison_results)

    st.dataframe(comp_df, use_container_width=True)

    st.markdown("### 📊 Model Comparison")

    fig, ax = plt.subplots(figsize=(4, 2))

    plot_df = comp_df.set_index("Model")[["Accuracy", "F1 Score"]]

    plot_df.plot(
        kind="bar",
        ax=ax,
        width=0.25
    )

    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Value", fontsize=5)
    ax.set_xlabel("")

    ax.set_title("Model Performance Comparison", fontsize=5)
    ax.tick_params(axis='x', labelsize=4)
    ax.tick_params(axis='y', labelsize=4)
    
    plt.xticks(rotation=20, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=5, loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

#Model Selection

st.subheader("⚙️ Select Model")

model_name = st.selectbox(
    "",
    ["-- Select a Model --"] + list(models.keys())
)

st.markdown('</div>', unsafe_allow_html=True)

if model_name == "-- Select a Model --":
    st.warning("Select a model to see results")
    st.stop()

model = models[model_name]

#Prepare
if "fetal_health" in data.columns:
    X_input = data.drop("fetal_health", axis=1)
    y_true = data["fetal_health"]
else:
    X_input = data
    y_true = None

#Scale
try:
    X_scaled = scaler.transform(X_input)
except:
    st.error("Column mismatch")
    st.stop()

#Predictions
predictions = model.predict(X_scaled)

label_map = {
    1: "Normal",
    2: "Suspect",
    3: "Pathological"
}

pred_labels = [label_map[int(p)] for p in predictions]

#Output
st.subheader(f"🔍 Predictions ({model_name})")

result_df = data.copy()
result_df["Predicted Health"] = pred_labels
st.dataframe(result_df, use_container_width=True)

#Performance
if y_true is not None:
    st.markdown("---")
    st.subheader("📈 Model Performance")

    label_names = ["Normal", "Suspect", "Pathological"]
    from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

    acc = accuracy_score(y_true, predictions)
    mcc = matthews_corrcoef(y_true, predictions)

    try:
        auc = roc_auc_score(y_true, model.predict_proba(X_scaled), multi_class='ovr')
    except:
        auc = 0

    #Display
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.3f}")
    m2.metric("AUC", f"{auc:.3f}")
    m3.metric("MCC", f"{mcc:.3f}")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Classification Report")
        report = classification_report(
            y_true,
            predictions,
            target_names=label_names,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    with col2:
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
        st.dataframe(cm_df, use_container_width=True)

#Footer
st.markdown("---")
st.markdown("<center> </center>", unsafe_allow_html=True)