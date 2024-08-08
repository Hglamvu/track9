import mlflow 
from mlflow_utils import create_mlflow_experiment
#import the class create_mlflow_experiment from the optimize module which can create a experiment
experiment_id = create_mlflow_experiment(
    #create an experiment with the name, artifact location and the tags
    expertiment_name="Nested Runs",
    artifact_location="nested_run_artifacts",
    tags={"purpose":"learning"}
)
#denote a run as a parent of the nested
with mlflow.start_run(run_name="parent") as parent:
    print("RUN ID parent:", parent.info.run_id)
    #print the id of the parent run 
    mlflow.log_param("parent_param","parent_value")
    #log the paras (param/value) of the run "parent"
    
    with mlflow.start_run(run_name="child1", nested=True) as child1:
    #start a run as a nested run of the parent ones
        print("RUN ID child1:", child1.info.run_id)
        #print the id of the run 
        mlflow.log_param("child1_param","child1_value")
        #log the paras (param/value) of the run 
        
        with mlflow.start_run(run_name="child11", nested=True) as child11:
        #start a run as a nested run of the run "child1" 
            print("RUN ID child11: ", child11.info.run_id)
            #print the id of the run 
            mlflow.log_param("child11_param","child11_value")
            #log the paras (param/value) of the run 

        with mlflow.start_run(run_name="child12", nested=True) as child12:
        #start a run as a nested run of the run "child1"
            print("RUN ID child12: ", child12.info.run_id)
            #print the id of the run 
            mlflow.log_param("child12_param", "child12_value")
            #log the paras (param/value) of the run

    with mlflow.start_run(run_name="child2", nested=True) as child2:
    #start a run as a nested run of the run "parent"
        print("RUN ID child2: ",child2.info.run_id)
        #print the id of the run 
        mlflow.log_param("child2_param", "child2_value")
         #log the paras (param/value) of the run