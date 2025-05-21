import pandas as pd

def predict_new_input(model, input_data):
    try:
        # Convert dict to dataframe
        input_df = pd.DataFrame([input_data])

        # Convert columns to numeric if possible
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='ignore')

        # Predict
        pred = model.predict(input_df)

        return pred[0]
    except Exception as e:
        return f"❌ Prediction error: {e}"
