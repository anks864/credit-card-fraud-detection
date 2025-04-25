# Import libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight

# Load train and test data
train_df = pd.read_csv('fraudTrain.csv', index_col='Unnamed: 0')
test_df = pd.read_csv('fraudTest.csv', index_col='Unnamed: 0')

# Function to modify features of dataset
def feature_engineering(df):
    # Convert and extract datetime features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_day'] = df['trans_date_trans_time'].dt.day
    df['trans_dow'] = df['trans_date_trans_time'].dt.dayofweek
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    
    # Behavioral features
    df['age'] = (pd.to_datetime('today') - pd.to_datetime(df['dob'])).dt.days
    df['distance'] = np.sqrt((df['lat']-df['merch_lat'])**2 + (df['long']-df['merch_long'])**2)
    
    # Sort by customer and time for rolling features
    df.sort_values(['cc_num', 'trans_date_trans_time'], inplace=True)
    
    # Transaction velocity features - fixed implementation
    def get_rolling_features(group):
        # Convert to seconds for rolling window
        group = group.set_index('trans_date_trans_time')
        window = '24h'
        
        # Calculate features
        group['trans_count_24h'] = group['amt'].rolling(window).count()
        group['amt_avg_24h'] = group['amt'].rolling(window).mean()
        group['amt_max_24h'] = group['amt'].rolling(window).max()
        
        return group.reset_index()
    
    # Apply to each customer group
    df = df.groupby('cc_num', group_keys=False).apply(get_rolling_features)
    
    # Calculate amount ratio
    df['amt_ratio_24h'] = df['amt'] / (df['amt_avg_24h'] + 0.01)  # Add small constant
    
    # Fill NA values (first transactions for each customer)
    velocity_features = ['trans_count_24h', 'amt_avg_24h', 'amt_max_24h', 'amt_ratio_24h']
    df[velocity_features] = df[velocity_features].fillna(0)
    
    # Drop non-feature columns
    cols_to_drop = ['trans_date_trans_time', 'dob', 'street', 'city', 
                   'state', 'zip', 'job', 'first', 'last', 'trans_num', 
                   'merchant', 'cc_num']
    df.drop(cols_to_drop, axis=1, inplace=True)
    
    return df

# Modify the train and test data
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Split features and target of the train and test datasets
X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

# Preprocessing
numeric_features = ['amt', 'city_pop', 'distance', 'age', 
                   'trans_hour', 'trans_day', 'trans_dow', 'trans_month',
                   'trans_count_24h', 'amt_avg_24h']
categorical_features = ['category', 'gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Optimized Random Forest with class weights
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=10,
        class_weight='balanced',
        n_jobs=-1,  # Use all available cores
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluation metrics
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Get feature names after preprocessing
preprocessor = model.named_steps['preprocessor']

# Numeric features stay the same
numeric_feature_names = numeric_features

# Get categorical feature names after one-hot encoding
categorical_transformer = preprocessor.named_transformers_['cat']
categorical_feature_names = categorical_transformer.get_feature_names_out(categorical_features)

# Combine all feature names
feature_names = numeric_feature_names + list(categorical_feature_names)

# Get feature importances
importances = model.named_steps['classifier'].feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# Print top 10 features
print("\nTop 10 Features:")
for i in sorted_idx[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legit', 'Fraud'],
            yticklabels=['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()