from torchrl.envs.utils import set_exploration_type, ExplorationType
import torch

def evaluate_policy(env, policy_module, cfg, logs):
    # We evaluate the policy once every 10 batches of data.
    # Evaluation is rather simple: execute the policy without exploration
    # (take the expected value of the action distribution) for a given
    # number of steps (1000, which is our ``env`` horizon).
    # The ``rollout`` method of the ``env`` can take a policy as argument:
    # it will then execute this policy at each step.
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy
        eval_rollout = env.rollout(cfg.eval.rollout_steps, policy_module)
        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
        logs["eval reward (sum)"].append(
            eval_rollout["next", "reward"].sum().item()
        )
        logs["eval step_count"].append(eval_rollout["step_count"].max().item())
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )
        del eval_rollout
        return eval_str
