from torch.nn.utils import clip_grad_norm_
from torchrl.envs.utils import ExplorationType, set_exploration_type
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow


def run_training_loop(ppo,  env, cfg, logs, pbar, policy_module, device):
    num_epochs = cfg.training.num_epochs # optimization steps per batch of data collected
    frames_per_batch = cfg.training.frames_per_batch 
    sub_batch_size = cfg.training.sub_batch_size
    max_grad_norm = cfg.training.max_grad_norm

    for i, tensordict_data in enumerate(ppo.collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            ppo.advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            ppo.replay_buffer.extend(data_view.cpu())

            for _ in range(frames_per_batch // sub_batch_size):
                subdata = ppo.replay_buffer.sample(sub_batch_size)
                loss_vals = ppo.loss_module(subdata.to(device))
                loss_objective = loss_vals["loss_objective"]
                loss_critic = loss_vals["loss_critic"]
                loss_entropy = loss_vals["loss_entropy"]

                total_loss = loss_objective + loss_critic + loss_entropy

                # Optimization: backward, grad clipping and optimization step
                total_loss.backward()

                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                grad_norm = clip_grad_norm_(ppo.loss_module.parameters(), max_grad_norm)
                ppo.optim.step()
                ppo.optim.zero_grad()


                # Log loss components and gradient norm to MLflow
                mlflow.log_metric("loss/total", total_loss.item(), step=i)
                mlflow.log_metric("loss/objective", loss_objective.item(), step=i)
                mlflow.log_metric("loss/critic", loss_critic.item(), step=i)
                mlflow.log_metric("loss/entropy", loss_entropy.item(), step=i)
                mlflow.log_metric("loss/grad_norm", grad_norm, step=i)

        
        # Logging 
        reward_mean = tensordict_data["next", "reward"].mean().item()
        step_count = tensordict_data["step_count"].max().item()
        lr = ppo.optim.param_groups[0]["lr"]

        logs["reward"].append(reward_mean)
        logs["step_count"].append(step_count)
        logs["lr"].append(lr)

        pbar.update(tensordict_data.numel())

        cum_reward_str = f"average reward={reward_mean: 4.4f} (init={logs['reward'][0]: 4.4f})"
        stepcount_str = f"step count (max): {step_count}"
        lr_str = f"lr policy: {lr: 4.4f}"

        # MLflow metric logging
        mlflow.log_metric("train/reward_mean", reward_mean, step=i)
        mlflow.log_metric("train/step_count", step_count, step=i)
        mlflow.log_metric("train/lr", lr, step=i)

        # Optional return for printing/logging externally if needed
        yield i, cum_reward_str, stepcount_str, lr_str
