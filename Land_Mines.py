import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
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

# Inspecting the data:
print("Dataframe info:", landmine_df.info(), "\n")

# Separating the target and feature for standardization:
target_column = 'M'
feature_columns = [col for col in landmine_df.columns if col != target_column]

# Standardizing the feature columns: 
standardizer = StandardScaler()
landmine_df_scaled = landmine_df.copy()
landmine_df_scaled[feature_columns] = standardizer.fit_transform(landmine_df[feature_columns])

#                                  --Performing Exploratory Data Analysis--                                  

# Correlating the data with the target and determining the most correlated feature:
correlation_with_target = landmine_df_scaled.corr()[target_column].abs().sort_values(ascending=False)
print("Absolute correlations with target (M):")
print(correlation_with_target, "\n")

# Summary statistics of the scaled data:
print("Summary statistics:")
print(landmine_df_scaled.describe(), "\n")

# Class distribution:
plt.figure()
landmine_df[target_column].value_counts().sort_index().plot(kind='bar')
plt.title("Class Distribution:")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Histogram:
for feature in feature_columns:
    plt.figure()
    landmine_df[feature].hist(bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

#                                  --Creating a classification model--                                  

x_train, X_test, y_train, y_test = train_test_split( landmine_df_scaled[feature_columns], 
    landmine_df_scaled[target_column],test_size=0.2, random_state=42, stratify=landmine_df_scaled[target_column]
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# Checking which model performs best:

model_scores = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    model.fit(x_train, y_train)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    model_scores[name] = test_acc
    
    print("Test Accuracy :", test_acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name} with accuracy = {model_scores[best_model_name]:.3f}", "\n")

# Saving the best model:

filename = f"land_mines_{best_model_name.lower()}.joblib"
joblib.dump(best_model, filename)
print(f"Exported best model to {filename}", "\n")

# Saving the standardizer for future deployment:

standardizer_filename = "land_mines_scaler.joblib"
joblib.dump(standardizer, standardizer_filename)
print(f"Saved standardizer to: {standardizer_filename}")