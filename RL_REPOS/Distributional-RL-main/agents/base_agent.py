from common.memory import TorchReplayBuffer
import torch
import random
import numpy as np
from torch import from_numpy
import copy
from torch import Tensor


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.exp_eps = 1
        self.memory = TorchReplayBuffer(config)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # build online and target models (in subclasses)
        self.online_model = None
        self.target_model = None
        self.optimizer = None

    def choose_action(self, state: np.ndarray) -> int:
        """ε-greedy action."""
        if random.random() < self.exp_eps:
            return random.randint(0, self.config.env.n_actions - 1)
        x = torch.as_tensor(state, device=self.device, dtype=torch.uint8).unsqueeze(0)
        with torch.no_grad():
            q = self.online_model.get_qvalues(x).cpu()
        return int(q.argmax(-1))
      
    def set_models(self, online_model: torch.nn.Module) -> None:
        """Clone online_model to create the frozen target_model."""
        """
        Call this after constructing your online_model:
          1) Clone it for the target
          2) Freeze the target's grads
        """
        self.online_model = online_model.to(self.device)
        self.target_model = copy.deepcopy(self.online_model).to(self.device)
        for p in self.target_model.parameters():
            p.requires_grad = False


    def store(self,
              state: np.ndarray,
              reward: float,
              done: bool,
              action: int,
              next_state: np.ndarray) -> None:
        """Add a transition to replay, enforcing correct dtypes."""
        assert state.dtype == np.uint8
        assert next_state.dtype == np.uint8
        assert isinstance(done, bool)
        if not isinstance(action, np.uint8):
            action = np.uint8(action)
        ############################
        #  Although we can decrease number of reward's bits but since it turns out to be a numpy array, its
        # overall size increases.
        ############################
        # if not isinstance(reward, np.int8):
        #     reward = np.int8(reward)
        self.memory.add(
            state.astype(np.uint8),
            float(reward),
            bool(done),
            np.uint8(action),
            next_state.astype(np.uint8),
        )

    def unpack_batch(self, batch) -> tuple[Tensor, ...]:
        """Move tensors to device and reshape."""
        states = batch["state"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].unsqueeze(-1).float().to(self.device)  # shape: [batch, 1]
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].unsqueeze(-1).to(self.device)  # convert bool to float if needed

        return states, actions, rewards, next_states, dones

    def update_target(self, tau: float) -> None:
        """Polyak update (tau=1 → hard, tau<1 → soft)."""
        for p_t, p in zip(self.target_model.parameters(),
                          self.online_model.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)

    
    def hard_target_update(self) -> None:
        self.update_target(tau=1.0)

    def soft_target_update(self) -> None:
        self.update_target(tau=self.config.soft_tau)

    # set model to eval, remove exploration
    def prepare_to_play(self) -> None:
        if not hasattr(self, "_exp_eps_backup"):
            self._exp_eps_backup = self.exp_eps
        self.online_model.eval()
        self.exp_eps=0

    # set model to train again, reinstate exploration
    def restore_after_play(self)  -> None:
        if hasattr(self, "_exp_eps_backup"):
            self.exp_eps = self._exp_eps_backup
        del self._exp_eps_backup
        self.online_model.train()



    def train(self):
        raise NotImplementedError

