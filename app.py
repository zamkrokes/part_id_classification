import pandas as pd
import streamlit as st
import numpy as np
import time
import pickle

# Function to load vectorizer, label encoders, and model
def load_vectorizer_label_encoders_model():
    loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
    loaded_label_encoder_org = pickle.load(open('label_encoder_org.pickle', 'rb'))
    loaded_label_encoder_part_id = pickle.load(open('label_encoder_part_id.pickle', 'rb'))
    loaded_model_RF = pickle.load(open('classification.model', 'rb'))
    return loaded_vectorizer, loaded_label_encoder_org, loaded_label_encoder_part_id, loaded_model_RF

# Function to make predictions
def prediction(df_input, output_csv, tfidf, label_encoder_org, label_encoder_part_id, model):
    start_time = time.time()
    X_text_input = tfidf.transform(df_input['description'])
    X_org_input = label_encoder_org.transform(df_input['organization']).reshape(-1, 1)
    X_input = np.hstack((X_text_input.toarray(), X_org_input))
    
    y_pred_input = model.predict(X_input)
    df_input['prediction'] = label_encoder_part_id.inverse_transform(y_pred_input)
    df_input = df_input[['prediction']]
    df_input.to_csv(output_csv, index=False)
    st.success("Prediction completed successfully!")
    end_time = time.time()
    return end_time - start_time

def main():
    st.title("Part ID Prediction App")
    
    st.write("Upload a CSV file with 'description' and 'organization' columns to make predictions.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        
        st.write("File uploaded successfully!")
        df = pd.read_csv(uploaded_file)
        
        st.write("Preview of the uploaded data:")
        st.write(df.head())
        
        # Load the joblib files
        try:
            vectorizer, label_encoder_org, label_encoder_part_id, model = load_vectorizer_label_encoders_model()
        except FileNotFoundError as e:
            st.error(f"Error loading joblib files: {e}")
            return
        
        if st.button("Make Predictions"):
            run_time = prediction(df, "results.csv", vectorizer, label_encoder_org, label_encoder_part_id, model)
            st.write("Predictions saved to results.csv")
            st.write(f"Prediction completed in {run_time} seconds")

if __name__ == "__main__":
    main()
