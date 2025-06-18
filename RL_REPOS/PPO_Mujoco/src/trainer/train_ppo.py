from torch.nn.utils import clip_grad_norm_
from torchrl.envs.utils import ExplorationType, set_exploration_type
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


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
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(ppo.loss_module.parameters(), max_grad_norm)
                ppo.optim.step()
                ppo.optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(ppo.optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

        # Optional return for printing/logging externally if needed
        yield i, cum_reward_str, stepcount_str, lr_str
