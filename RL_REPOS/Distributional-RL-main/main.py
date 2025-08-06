# Reference: https://github.com/ku2482/fqf-iqn-qrdqn.pytorch

from common import set_random_seeds, make_atari
from common import Logger, Evaluator
from agents import get_agent
import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig):
    env = make_atari(config.env.env_name, config.seed)
    config.max_total_steps = config.max_episodes * env.spec.max_episode_steps
    set_random_seeds(config.seed)

    if os.path.exists("api_key.wandb"):
        with open("api_key.wandb", 'r') as f:
            os.environ["WANDB_API_KEY"] = f.read()
            if not config.online_wandb:
                os.environ["WANDB_MODE"] = "offline"

    test_env = make_atari(config.env.env_name, config.seed)

    n_actions = int(test_env.action_space.n)
    config.env.n_actions = n_actions
    del test_env
    
    print(f"Environment: {config.env.env_name}\n"
          f"Number of actions: {config.env.n_actions}")

    
    agent = get_agent(config)
 
    logger = Logger(agent, config)


    if not config.do_test:
        total_steps = 0
        for episode in range(1, 1 + config.max_episodes):
            logger.on()
            episode_reward = 0
            episode_loss = 0
            state, _ = env.reset()
            for step in range(1, 1 + env.spec.max_episode_steps):
                total_steps += 4  # 4: MaxAndSkip env!
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  
                agent.store(state, reward, done, action, next_state)
                agent.increment_step()
                episode_reward += reward


                if total_steps % config.train_interval == 0:
                    metrics = agent.train()
                    episode_loss += metrics["loss/total"]
                if done:
                    break
                state = next_state

            logger.off()
            log_data = {
                        'episode/episode': episode,
                        'reward/episode_reward': episode_reward,
                        'episode/loss': episode_loss / step * config.train_interval,
                        'step': total_steps,
                        'episode/e_len': step,
                        }
            combined_log_data = log_data | metrics 
            logger.log(**combined_log_data)

            if len(agent.memory) > config.init_mem_size_to_train and episode % config.eval_interval == 0:
                evaluator = Evaluator(agent, config)
                evaluator.evaluate(total_steps)



    else:
        checkpoint = logger.load_weights()
        agent.online_model.load_state_dict(checkpoint["online_model_state_dict"])
        agent.exp_eps = 0
        evaluator = Evaluator(agent, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
