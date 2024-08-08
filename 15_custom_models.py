import mlflow 
from mlflow_utils import create_mlflow_experiment
    #import the function to create experiment 
class CustomModel(mlflow.pyfunc.PythonModel):
    #defind a optimize class CustomModel to defind a Model which can combine with mlflow 
    def __init__(self):
        pass 
    #a function to initialize class CustomModel
    def fit(self):
        print("Fitting model...")
    #a function to simulate the process of training model
    def predict(self, context, model_input:[str]):
        return self.get_prediction(model_input)
    #a function which receive a list of string model_input 
    def get_prediction(self, model_input:[str]):
        #do sth with the model input
        return " ".join([w.upper() for w in model_input])
    #a function to handle the model_input and return a string  

if __name__=="__main__":
    #create an experiment with the name, artifact location and the tags
    experiment_id = create_mlflow_experiment(
        experiment_name= "Custom Models",
        artifact_location= "custom_model_artifacts",
        tags={"purpose":"learning"}
    )
    #start a new run with a name of custom_model_run, automatic log the info like params, model, result
    with mlflow.start_run(run_name="custom_model_run") as run:
        custom_model = CustomModel()
        #create a object of class CustomModel
        custom_model.fit()
        #know as the process of training model 
        mlflow.pyfunc.log_model(
            artifact_path="custom_model",
            python_model=custom_model)
        #store the trained model in artifact format with the path of customer_model
        mlflow.log_param("param1", "value1")
        #log the para: param1 with the value of value1
        #load model
        custom_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/custom_model")
        #upload the saved model  with the following path with the currently id of the run
        prediction = custom_model.predict(["hello", "world"])
        #call the predict class to predict the dictionary of the list "hello, world"  
        print(prediction)
        