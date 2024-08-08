import mlflow
from mlflow.models import infer_signature
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification 
#from the lib sklearn to create simulated data for clastification problems 
from sklearn.model_selection import train_test_split
#the lib to split the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier
#import lib sklearn 
import pandas as pd
#import the pandas lib which provides the tool which operate and analyze the tabular data
if __name__ == "__main__":
    run_id = "8cf35d0c32854dcdb6a310e7c2ced3b0"
    #set the id of the run which uploaded 
    
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
    #create simulated dataset for clasification problems with 1000 samples, 10 features, 5 informative and 5 redundant  
    
    X = pd.DataFrame(X, columns=["feature_{}" .format(i) for i in range(10)])
    #transform X into the dataframe of pandas with columns are named from 'feature_0' to 'feature_9'
    
    y = pd.DataFrame(y, columns=["target"])
    #transform y into the dataframe of pandas with column named of 'target'
    
    _, X_test, _, y_test = train_test_split(X,y, test_size=0.2, random_state= 43)
    #split the dataset into training and testing sets with 20% of the data for testing set('X_test', 'y_test')
    #load model
    #model_url = f'runs:/{run_id}/random_access_classifier
    #the path/url of the run 
    model_url = f"/Users/lamvuhoang/Desktop/track9/testing_mlflow1_artifact/{run_id}/artifacts/random_forest_classifier"
    rfc = mlflow.sklearn.load_model(model_uri=model_url)
    #upload the trained model from Mlflow 
    y_pred = rfc.predict(X_test)
    #predict the lable of the 'X_test' by the uploaded model 'rfc'
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])
    #transform the result of the prediction into the DataFrame of Pandas with the column named prediction
    print(y_pred.head())


