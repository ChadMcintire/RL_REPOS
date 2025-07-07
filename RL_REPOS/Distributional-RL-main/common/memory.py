from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

from tensordict import TensorDict
import torch

class TorchReplayBuffer:
    def __init__(self, config, device="cpu"):
        print(config)
        self.buffer_size = config.buffer.buffer_size
        self.device = device

     
        if config.buffer.name=="prio":
            alpha = config.buffer.alpha
            beta = config.buffer.beta
            self.sampler = PrioritizedSampler(
                max_capacity=self.buffer_size, alpha=alpha, beta=beta
            )
        else:
            self.sampler = SamplerWithoutReplacement()

        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.buffer_size, device=device),
            sampler=self.sampler,
        )

    def add(self, state, reward, done, action, next_state):
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, device=self.device).unsqueeze(0)
        action = torch.tensor(action, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.bool, device=self.device)

        td = TensorDict({
            "state": state,
            "reward": reward,
            "done": done,
            "action": action,
            "next_state": next_state,
        }, batch_size=[1])
        self.buffer.extend(td)

    def sample(self, batch_size):
        # 1) sample returns (TensorDict, info_dict)
        batch_td, info = self.buffer.sample(batch_size, return_info=True)

        # 2) move only the TensorDict to the correct device
        batch_td = batch_td.to(self.device)

        # 3) If the sampler got exhausted, re-sample
        if batch_td.batch_size[0] != batch_size:
            print("starting buffer over")
            batch_td, info = self.buffer.sample(batch_size, return_info=True)
            batch_td = batch_td.to(self.device)

        # 4) return the TensorDict and the info dict separately
        return batch_td, info


    def update_priorities(self, td_errors, indices):
        if isinstance(self.sampler, PrioritizedSampler):
            priorities = td_errors.abs().detach().cpu() + 1e-6
            self.sampler.update_priority(indices, priorities)

    def __len__(self):
        return len(self.buffer)
