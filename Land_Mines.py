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

# Remove noisy outliers in V and H (|z|>3)
for col in ['V','H']:
    landmine_df = landmine_df[np.abs(zscore(landmine_df[col])) < 3]
landmine_df.reset_index(drop=True, inplace=True)
print(f"After outlier removal: {landmine_df.shape[0]} rows\n")

# Creating a binary target for detection instead of classification (1 = Mine, 0 = No mine):
target_column = 'M'
landmine_df['is_mine'] = (landmine_df[target_column] != 1).astype(int)

#                                  --Performing Exploratory Data Analysis--                                   

# Correlating the data with the target and determining the most correlated feature:
correlation_with_target = landmine_df.corr()[target_column].abs().sort_values(ascending=False)
print("Absolute correlations with target (M):")
print(correlation_with_target, "\n")

# Summary statistics of the scaled data:
print("Summary statistics:")
print(landmine_df.describe(), "\n")

# Class distribution:
plt.figure()
landmine_df[target_column].value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution:")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Histogram:
for feature in ['V','H']:
    plt.figure()
    landmine_df[feature].hist(bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

#                                  --Detection pipeline--                                   

# Training and testing the split for detection:
X_det = landmine_df[['V','H','S']]
y_det = landmine_df['is_mine']
x_train_det, X_test_det, y_train_det, y_test_det = train_test_split(
    X_det, y_det, test_size=0.2, random_state=42, stratify=y_det
)

# Create a list of all possible values for 'S'
s_values = [1.0, 2.0, 3.0, 4.0]  # Add all possible values you expect

# Preprocessor for pipeline (scales V & H, encodes S) with explicit categories
detector_preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), ['V', 'H']),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', 
                           categories=[s_values]), ['S'])
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
    cv_scores = cross_val_score(pipeline, x_train_det, y_train_det, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(x_train_det, y_train_det)
    y_pred_det = pipeline.predict(X_test_det)
    acc_det = accuracy_score(y_test_det, y_pred_det)
    model_scores_det[name] = (acc_det, pipeline)
    print("Test Accuracy :", acc_det)
    print("Classification Report:\n", classification_report(y_test_det, y_pred_det))
    print("Confusion Matrix:\n", confusion_matrix(y_test_det, y_pred_det))

best_det_name = max(model_scores_det, key=lambda k: model_scores_det[k][0])
best_det_pipeline = model_scores_det[best_det_name][1]
print(f"\nBest Detection Model: {best_det_name} with accuracy = {model_scores_det[best_det_name][0]:.3f}\n")
joblib.dump(best_det_pipeline, f"land_mines_{best_det_name.lower()}_detection.joblib")

#                                  --Creating a classification model--                                   

# Checking that rows where target is 1 are not removed:
df_mines = landmine_df[landmine_df['is_mine'] == 1].copy()
X_clf = df_mines[['V','H','S']]
y_clf = df_mines[target_column]

# Training and testing the split for classification:
x_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# Preprocessor for classification with explicit categories
classifier_preprocessor = ColumnTransformer([
    ('scale', StandardScaler(), ['V', 'H']),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', 
                           categories=[s_values]), ['S'])
])

# Comparing and choosing the best algorithms:
models = {
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced')
}

# Checking which model performs best:
model_scores = {}
for name, model in models.items():
    print(f"\n--- Classification: {name} ---")
    pipeline = Pipeline([
        ('preprocess', classifier_preprocessor),
        ('classifier', model)
    ])
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    model_scores[name] = (test_acc, pipeline)
    print("Test Accuracy :", test_acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

best_model_name = max(model_scores, key=lambda k: model_scores[k][0])
best_model_pipeline = model_scores[best_model_name][1]
print(f"\nBest Classification Model: {best_model_name} with accuracy = {model_scores[best_model_name][0]:.3f}\n")

# Saving the best classification pipeline:
filename = f"land_mines_{best_model_name.lower()}_classification.joblib"
joblib.dump(best_model_pipeline, filename)
print(f"Exported best model to {filename}\n")
