from torchrl.envs.utils import set_exploration_type, ExplorationType
import torch
import mlflow

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
        reward_mean = eval_rollout["next", "reward"].mean().item()
        reward_sum = eval_rollout["next", "reward"].sum().item()
        step_count = eval_rollout["step_count"].max().item()

        logs["eval reward"].append(reward_mean)
        logs["eval reward (sum)"].append(reward_sum)
        logs["eval step_count"].append(step_count)

        mlflow.log_metric("eval/reward_mean", reward_mean)
        mlflow.log_metric("eval/reward_sum", reward_sum)
        mlflow.log_metric("eval/step_count", step_count)

        #eval_str = (
        #    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
        #    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
        #    f"eval step-count: {logs['eval step_count'][-1]}"
        #)

        eval_str = (
            f"eval cumulative reward: {reward_sum: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {step_count}"
        )

        del eval_rollout
        return eval_str
