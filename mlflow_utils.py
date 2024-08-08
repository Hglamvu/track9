import mlflow
from typing import Any

import mlflow.entities

def create_mlflow_experiment(experiment_name: str, artifact_location:str, tags:dict[str,Any]) -> str:
    #create a new mlflow experiment with the given name and artifact location
    
    try:
        expertiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location,tags=tags
        )
    except:
        print(f"Expertiment {experiment_name} already exists!")
        expertiment_id= mlflow.get_experiment_by_name(experiment_name).experiment_id

    return expertiment_id

def get_mlflow_experiment(experiment_id:str=None, experiment_name:str=None) -> mlflow.entities.Experiment:
    #retrieving the experiment with the parameter of id and name   

    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided!")
    return experiment

    


    