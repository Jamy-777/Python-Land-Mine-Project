import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

from ucimlrepo import fetch_ucirepo 
  
# Fetching the dataset: 
land_mines = fetch_ucirepo(id=763) 
  
# Data (as pandas dataframes) 
X = land_mines.data.features 
y = land_mines.data.targets 
  
# Metadata 
print(land_mines.metadata, "\n") 
  
# Variable Information 
print(land_mines.variables, "\n")

#                                     --Data Cleaning and transformation--                                     

# Combining the Features and Targets into a single DataFrame:
landmine_df = pd.concat([X, y], axis=1)

# Print original class distribution
print("Original Class Distribution:")
print(landmine_df['M'].value_counts().sort_index())
print()

# Remove noisy outliers in V and H (|z|>3), but stratify by class to maintain distribution
landmine_df_clean = landmine_df.copy()
for class_val in landmine_df['M'].unique():
    class_df = landmine_df[landmine_df['M'] == class_val]
    # Only filter outliers if there are enough samples
    if len(class_df) > 10:  
        for col in ['V','H']:
            class_df = class_df[np.abs(zscore(class_df[col])) < 3]
        landmine_df_clean = pd.concat([landmine_df_clean[landmine_df_clean['M'] != class_val], class_df])

landmine_df = landmine_df_clean.reset_index(drop=True)
print(f"After outlier removal: {landmine_df.shape[0]} rows\n")

# Print cleaned class distribution
print("Cleaned Class Distribution:")
print(landmine_df['M'].value_counts().sort_index())
print()

# Creating a binary target for detection (1 = Mine, 0 = No mine):
target_column = 'M'landmine_df['is_mine'] = (landmine_df[target_column] != 1).astype(int)

#                                  --Performing Exploratory Data Analysis--                                   

# Correlating the data with the target and determining the most correlated feature:
correlation_with_target = landmine_df.corr()[target_column].abs().sort_values(ascending=False)
print("Absolute correlations with target (M):")
print(correlation_with_target, "\n")

# Summary statistics of the scaled data:
print("Summary statistics:")
print(landmine_df.describe(), "\n")

# Class distribution:
plt.figure(figsize=(10, 6))
landmine_df[target_column].value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution:")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Adding a more detailed plot for mine types
plt.figure(figsize=(12, 6))
mine_counts = landmine_df[target_column].value_counts().sort_index()
mine_labels = {
    1: "No Mine", 
    2: "AT Mine", 
    3: "AP Mine",
    4: "AT & AP Mine"
}
mine_counts.index = [mine_labels.get(i, f"Class {i}") for i in mine_counts.index]
mine_counts.plot(kind='bar')
plt.title("Detailed Mine Type Distribution")
plt.xlabel("Mine Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Histogram for features by class:
for feature in ['V','H']:
    plt.figure(figsize=(10, 6))
    for class_val in sorted(landmine_df[target_column].unique()):
        class_name = mine_labels.get(class_val, f"Class {class_val}")
        subset = landmine_df[landmine_df[target_column] == class_val]
        plt.hist(subset[feature], bins=20, alpha=0.5, label=class_name)
    
    plt.title(f'Distribution of {feature} by Mine Type')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Scatter plot to visualize V vs H by class
plt.figure(figsize=(10, 8))
for class_val in sorted(landmine_df[target_column].unique()):
    class_name = mine_labels.get(class_val, f"Class {class_val}")
    subset = landmine_df[landmine_df[target_column] == class_val]
    plt.scatter(subset['V'], subset['H'], alpha=0.6, label=class_name)

plt.title('V vs H by Mine Type')
plt.xlabel('V')
plt.ylabel('H')
plt.legend()
plt.tight_layout()
plt.show()

#                                  --Detection pipeline--                                   

# Training and testing the split for detection:
X_det = landmine_df[['V','H','S']]
y_det = landmine_df['is_mine']
X_train_det, X_test_det, y_train_det, y_test_det = train_test_split(
    X_det, y_det, test_size=0.2, random_state=42, stratify=y_det
)

# Preprocessor for pipeline (scales V & H, encodes S)
detector_preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), ['V', 'H']),
    ('encode', OneHotEncoder(drop='first', sparse_output=False), ['S'])
])

models_det = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced')
}

model_scores_det = {}
for name, model in models_det.items():
    print(f"\n--- Detection: {name} ---")
    pipeline = Pipeline([
        ('preprocess', detector_preprocessor),
        ('classifier', model)
    ])
    cv_scores = cross_val_score(pipeline, X_train_det, y_train_det, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(X_train_det, y_train_det)
    y_pred_det = pipeline.predict(X_test_det)
    acc_det = accuracy_score(y_test_det, y_pred_det)
    model_scores_det[name] = (acc_det, pipeline)
    print("Test Accuracy:", acc_det)
    print("Classification Report:\n", classification_report(y_test_det, y_pred_det))
    print("Confusion Matrix:\n", confusion_matrix(y_test_det, y_pred_det))

best_det_name = max(model_scores_det, key=lambda k: model_scores_det[k][0])
best_det_pipeline = model_scores_det[best_det_name][1]
print(f"\nBest Detection Model: {best_det_name} with accuracy = {model_scores_det[best_det_name][0]:.3f}\n")
joblib.dump(best_det_pipeline, f"land_mines_{best_det_name.lower()}_detection.joblib")

#                                  --Creating a classification model--                                   

# Use the entire dataset but make classification multi-class (mine types)
X_clf = landmine_df[['V', 'H', 'S']]
y_clf = landmine_df[target_column]  

# Training and testing the split for classification:
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

classifier_preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), ['V', 'H']),
    ('encode', OneHotEncoder(drop='first', sparse_output=False), ['S'])
])

models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced')
}

model_scores = {}
for name, model in models.items():
    print(f"\n--- Classification: {name} ---")
    pipeline = Pipeline([
        ('preprocess', classifier_preprocessor),
        ('classifier', model)
    ])
    cv_scores = cross_val_score(pipeline, X_train_clf, y_train_clf, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(X_train_clf, y_train_clf)
    y_pred = pipeline.predict(X_test_clf)
    test_acc = accuracy_score(y_test_clf, y_pred)
    model_scores[name] = (test_acc, pipeline)
    print("Test Accuracy:", test_acc)
    print("Classification Report:\n", classification_report(y_test_clf, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred))

best_model_name = max(model_scores, key=lambda k: model_scores[k][0])
best_model_pipeline = model_scores[best_model_name][1]
print(f"\nBest Classification Model: {best_model_name} with accuracy = {model_scores[best_model_name][0]:.3f}\n")

filename = f"land_mines_{best_model_name.lower()}_classification.joblib"
joblib.dump(best_model_pipeline, filename)
print(f"Exported best model to {filename}\n")

def predict_landmine(V, H, S, detection_model, classification_model):
    """
    Predicts landmine presence and type using both models.
    
    Args:
        V, H, S: Feature values
        detection_model: Model to detect if there's a mine
        classification_model: Model to classify mine type
        
    Returns:
        Dictionary with prediction results
    """
    input_data = pd.DataFrame({'V': [V], 'H': [H], 'S': [S]})
    
    is_mine = detection_model.predict(input_data)[0]
    is_mine_prob = detection_model.predict_proba(input_data)[0][1]  
    
    mine_type = None
    mine_type_probs = None
    if is_mine == 1:
        mine_type = classification_model.predict(input_data)[0]
        mine_type_probs = classification_model.predict_proba(input_data)[0]
    
    result = {
        'is_mine': bool(is_mine),
        'is_mine_prob': float(is_mine_prob),
        'mine_type': int(mine_type) if mine_type is not None else None,
        'mine_type_probs': mine_type_probs.tolist() if mine_type_probs is not None else None
    }
    
    return result

print("\n--- Example Prediction ---")
# Load the best models
detection_model = joblib.load(f"land_mines_{best_det_name.lower()}_detection.joblib")
classification_model = joblib.load(filename)

# Get a sample from the test set
sample_idx = 0
sample_V = X_test_clf.iloc[sample_idx]['V']
sample_H = X_test_clf.iloc[sample_idx]['H']
sample_S = X_test_clf.iloc[sample_idx]['S']
actual_class = y_test_clf.iloc[sample_idx]

print(f"Sample features: V={sample_V}, H={sample_H}, S={sample_S}")
print(f"Actual class: {actual_class} ({mine_labels.get(actual_class, 'Unknown')})")

# Make prediction
prediction = predict_landmine(sample_V, sample_H, sample_S, detection_model, classification_model)
print("\nPrediction result:")
print(f"Is mine: {prediction['is_mine']} (probability: {prediction['is_mine_prob']:.3f})")
if prediction['is_mine']:
    print(f"Mine type: {prediction['mine_type']} ({mine_labels.get(prediction['mine_type'], 'Unknown')})")
    print(f"Mine type probabilities: {prediction['mine_type_probs']}")
