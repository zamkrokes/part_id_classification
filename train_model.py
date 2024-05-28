import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Preprocess text data using TF-IDF
tfidf = TfidfVectorizer()
X_text = tfidf.fit_transform(df['description'])

# Encode the organization column
label_encoder_org = LabelEncoder()
X_org = label_encoder_org.fit_transform(df['organization']).reshape(-1, 1)

# Encode the part_id column
label_encoder_part_id = LabelEncoder()
y = label_encoder_part_id.fit_transform(df['part_id'])

# Combine features
X = np.hstack((X_text.toarray(), X_org))

# Train-test split with return_index to get indices
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, df.index, test_size=0.2, random_state=42)

# Determine the minimum number of samples in the minority class
class_counts = np.bincount(y_train)
min_class_samples = min(class_counts)

# Set k_neighbors to be less than or equal to the minimum number of samples in the minority class
# Ensure k_neighbors is at least 1
k_neighbors = max(1, min(5, min_class_samples - 1))

# Handle imbalance using SMOTE with adjusted k_neighbors
if min_class_samples > 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
else:
    X_train_resampled, y_train_resampled = X_train, y_train

# Model training
model_RF = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
model_RF.fit(X_train_resampled, y_train_resampled)

# Save the vectorizer
vec_file = 'vectorizer.pickle'
pickle.dump(tfidf, open(vec_file, 'wb'))

# Save the label encoders
label_encoder_org_file = 'label_encoder_org.pickle'
pickle.dump(label_encoder_org, open(label_encoder_org_file, 'wb'))

label_encoder_part_id_file = 'label_encoder_part_id.pickle'
pickle.dump(label_encoder_part_id, open(label_encoder_part_id_file, 'wb'))

# Save the model
model_file = 'classification.model'
pickle.dump(model_RF, open(model_file, 'wb'))

print("Model and other components saved successfully.")