import streamlit as st
import pandas as pd
from train_models import train_and_evaluate_models
from predict import predict_new_input

st.title("🔍 Smart AutoML App - Regression & Classification")

uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Data Preview:", df.head())

    task_type = st.radio("📌 Select Task Type", ["regression", "classification"])

    all_columns = df.columns.tolist()
    features = st.multiselect("✅ Select Features (Independent Variables)", options=all_columns)
    target = st.selectbox("🎯 Select Target (Dependent Variable)", options=all_columns)

    if features and target:
        if st.button("🚀 Train Models"):
            try:
                best_model, best_model_name, train_score, cv_score = train_and_evaluate_models(df, features, target, task_type)
                st.session_state.best_model = best_model
                st.session_state.features = features
                st.session_state.task_type = task_type

                st.success("✅ Model training complete!")
                st.write(f"🏆 Best Model: **{best_model_name}**")
                st.write(f"📈 Training Score: **{train_score:.4f}**")
                st.write(f"📉 Validation CV Score: **{cv_score:.4f}**")
            except Exception as e:
                st.error(f"❌ Error during training: {e}")

    if "best_model" in st.session_state:
        st.subheader("🧠 Make a Prediction")
        user_input = {}
        for feat in st.session_state.features:
            val = st.text_input(f"Enter value for '{feat}'")
            user_input[feat] = val

        if st.button("🔮 Predict"):
            prediction = predict_new_input(st.session_state.best_model, user_input)
            st.success(f"🎯 Prediction Result: {prediction}")
