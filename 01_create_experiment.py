import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":
    # # #create a new experiment
    # # mlflow.create_experiment(
    # #     name = "testing_mlflow1",
    # #     artifact_location = "testing_mlflow1_artifacts",
    # #     tags = {"env": "dev", "version": "1.0.0"},
    # # )
    # expertiment_id= create_mlflow_experiment(expertiment_name="testing_mlflow2",artifact_location = "testing_mlflow2_artifacts",tags = {"env": "dev", "version": "1.0.0"},)
    # print(f"Expertiment ID: {expertiment_id}")
    
    experiment_id = mlflow.create_experiment(
        name="testing_mlflow1",
        artifact_location="testing_mlflow1_artifacts",
        tags={"env": "dev", "version": "1.0.0"},
    )

    print(experiment_id)