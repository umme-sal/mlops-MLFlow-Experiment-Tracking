import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='umme-sal', repo_name='mlops-MLFlow-Experiment-Tracking', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/umme-sal/mlops-MLFlow-Experiment-Tracking.mlflow")

# Load wine dataset
wine = load_wine()

X = wine.data
y = wine.target


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)


# Define parameters for RF model
max_depth = 7
n_estimators = 15

# mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("wine_rf_experiment")

with mlflow.start_run():

    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=42
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    #create a confusion matrix
    cm=confusion_matrix(y_test, y_pred)
    #plot the confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(rf, "model")
    print(accuracy)