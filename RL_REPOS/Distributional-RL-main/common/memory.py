from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tensordict import TensorDict
import torch

class TorchReplayBuffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer_size = buffer_size
        self.device = device

        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size, device=device),
            sampler=SamplerWithoutReplacement(),
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
        batch = self.buffer.sample(batch_size).to(self.device)

        # Sampler exhausted — reset and try again
        if batch.batch_size[0] != batch_size:
            print("starting buffer over")
            batch = self.buffer.sample(batch_size).to(self.device)

        return batch

    def __len__(self):
        return len(self.buffer)
