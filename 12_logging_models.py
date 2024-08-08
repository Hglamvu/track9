import mlflow
from mlflow_utils import get_mlflow_experiment

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__=="__main__":

    experiment = get_mlflow_experiment(experiment_name="testing_mlflow1")
    print("Name: {}".format(experiment.name))

    with mlflow.start_run(run_name="logging_images", experiment_id=experiment.experiment_id) as run:
    #start the run named 'logging_images' with the id of the currently experiment id 
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
        #create simulated dataset for clasification problems with 1000 samples, 10 features, 5 informative and 5 redundant  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
        #split the dataset into training and testing sets with 20% of the data for testing set('X_test', 'y_test')
        
        # #log model parameters using autolog
        # mlflow.autolog()
        mlflow.sklearn.autolog()
        #activate the mlflow's automatic logging mode of sklearn lib. all the parameters, models and metrics will be logged for this modlel
        rfc = RandomForestClassifier(n_estimators=100, random_state=42)
        #initialize a object RandomForestClassifier with 100 estimators and 42 random state to ensure the reproducibility
        rfc.fit(X_train, y_train)
        #train the model in the trained dataset 'x_train', 'y_train'
        y_pred= rfc.predict(X_test)
        #predict the lable of the 'X_test' by the uploaded model 'rfc'

        #log model
        mlflow.sklearn.log_model(sk_model=rfc, artifact_path= "random_forest_classifier")
        #store the trained model 'rfc' in a artifact format with the path of random_forest_classifier  
        
        #print info of the run
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))



