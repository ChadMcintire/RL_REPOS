import os
import imageio
import numpy as np
import torch
from torchrl.envs.utils import set_exploration_type, ExplorationType
from hydra.utils import get_original_cwd


def record_current_model(env, cfg, current_step, policy_module):
    print(f"[Render] Visualizing agent policy at step {current_step}")
    from hydra.utils import to_absolute_path
    
    video_dir = to_absolute_path(cfg.paths.video_dir)
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"ppo_render_step{current_step}.mp4")

    with imageio.get_writer(video_path, fps=int(cfg.render.fps)) as video_writer:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for ep in range(cfg.render.eps_range):
                td = env.reset()
                for _ in range(cfg.render.eps_length):
                    td.update(policy_module(td))
                    td = env.step(td)

                    frame = env.base_env.render()

                    if frame is not None and hasattr(frame, "shape"):
                        video_writer.append_data(np.asarray(frame))
                    else:
                        print("[Warning] Invalid frame; skipping...")

    print(f"[Render] Saved {video_path}")
