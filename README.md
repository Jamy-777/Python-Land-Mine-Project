This repository is for a Python project of mine which imports a dataset from UCI ML Repo (https://archive.ics.uci.edu/dataset/763/land+mines-1).

Structure of the project:
├── Land_Mines.py                               This is the training part of the project where I have made the model.
├── Deployed_Model.py                           This is the Streamlit web app that I have implemented for the model I made to give an interactive feel.

├── requirements.txt                            These are the various Python libraries you need in order to run this file.
├── land_mines_randomforest_detection.joblib    This is the detection model written to the disk using JobLib library and the web app refers to this.

├── land_mines_randomeforest_classification.joblib    This is the classification step of the modeling and only runs when the detection step is executed and the result is 1 or true or the mine is there waiting for you.

├── Paper.pdf                              This is the paper I have referred to for the dataset and the working of the model.
├── Mine_Dataset_CSV.csv                   This is the dataset file having three features or variables on which we have to work on to find the target variable.
├── Mine_Dataset.xls                       This is the dataset with the required brief information about the features and the target.
├── README.md                              This is the documentation of the project, I hope it's good enough.

OVERVIEW: 

My project predicts if a mine is present in the soil or not, if yes then of what type.
I have used Streamlit for creating a web app and JobLib for exporting the best model based on the accuracy of the models I have used.

The dataset has three variable or features on which I have to work on and they contribute to the presence of the target which is mine here. The correlation between the features result in a result which is surprising for passive detectors, achieving 98.2% accuracy.
I have used the binary approach for this model meaning I have used two pipelines for making it because a single pipelines will have the
inherent limitation of miss-classifying the mine undermining the main goal which is detecting the mine in the first place as deciding what
type of mine is there is far less important than deciding if a mine is there or not.

WORKING OF THE MODEL:

1. The model first cleans the data by removing the fringe elements by performing Z-score on V and H variables.
2. Two stage modeling:
-- Detection modeling (Binary classification): To mine or not to mine.
-- Classification modeling: Deciding the type of the mine.
3. Comparing which models work the best?
-- Regression
-- RandomeForest or something else...

DEPLOYMENT OF THE MODEL:

I have deployed the app using Streamlit which gives a nice minimalist GUI where the user can enter the data to determine whether a mine is
there or not.

If you want to run the Streamlit app locally then open terminal in the directory of this app and type: streamlit run Land_Mines.py, this 
will open the app locally in your browser.

SOURCES AND HELPING DOCUMENTATION:

Z-Score Outlier Tutorial: https://towardsdatascience.com/identifying-outliers-using-standard-deviation-ebf3b9caa0cf
ColumnTransformer Guide: https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-numeric-and-categorical-data
Train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html  
Cross_val_score: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html  
Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
Confusion Matrix Explained: https://scikit-learn.org/stable/visualizations.html#confusion-matrix
