import os
import mlflow
from hydra.core.hydra_config import HydraConfig


def setup_mlflow(cfg, script_name="run"):
    # Get the working directory from Hydra
    run_dir = HydraConfig.get().run.dir
    print(f"[MLflow] Hydra working directory: {run_dir}")

    # Set the tracking URI
    tracking_uri = os.path.abspath(cfg.paths.log_dir)
    mlflow.set_tracking_uri(f"file://{tracking_uri}")
    print(f"[MLflow] Tracking URI set to: {tracking_uri}")

    # Set the experiment name
    mlflow.set_experiment(cfg.experiment.name)

    # Start a run
    run = mlflow.start_run(run_name=script_name)

    # Log basic parameters
    mlflow.log_params({
        "total_frames": cfg.training.total_frames,
        "lr": cfg.training.lr,
        "gamma": cfg.training.gamma,
        "lambda": cfg.training.lmbda,
        "entropy_eps": cfg.algo.ppo.entropy_eps,
        "activation_fn": cfg.model.hidden_activation,
    })

    return run  # Optional, if you want to stop it later manually
