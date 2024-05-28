# Part Classification

This project includes scripts for training a machine learning model and a Streamlit application to predict part IDs based on descriptions and organizations.

## Instructions

### Model Training
1. Ensure `dataset.csv` is in the same directory as `train_model.py`.
2. Run `train_model.py` to train the model and save the necessary components.
    ```sh
    python train_model.py
    ```
3. The following files will be generated:
    - `vectorizer.pickle`
    - `label_encoder_org.pickle`
    - `label_encoder_part_id.pickle`
    - `classification.model`

### Prediction using Streamlit
1. Ensure the files generated from the training step are in the same directory as `app.py`.
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```
4. Upload a CSV file with `description` and `organization` columns.
5. Click "Make Predictions" to generate the `results.csv` file with the predictions.

### Example CSV
An example input CSV should have the following columns:
- `description`
- `organization`

### Deliverables
- `train_model.py`: Script to train the model.
- `app.py`: Streamlit app for predictions.
- `requirements.txt`: List of required packages.
- `README.md`: Instructions on how to run the scripts.