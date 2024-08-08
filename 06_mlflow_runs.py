import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":
    experiment = create_mlflow_experiment(
        expertiment_name="testing_mlflow1",
        artifact_location="testing_mlflow1_artifact",
        tags={"env":"dev","version":"1.0.0"},
    )
    mlflow.set_experiment(experiment_name="testing_mlflow1")
    with mlflow.start_run(run_name="testing") as run:
        mlflow.log_param("learning_rate",0.01)
        #print run info
        print("run_id: {}" .format(run.info.run_id))
        print("experiment_id: {}" .format(run.info.experiment_id))
        print("status: {}" .format(run.info.status))
        print("start_time: {}" .format(run.info.start_time))
        print("end_time: {}" .format(run.info.end_time))
        print("lifecycle_stage:{}" .format(run.info.lifecycle_stage))
       
