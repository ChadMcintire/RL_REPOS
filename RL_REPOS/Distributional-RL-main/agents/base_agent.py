from common.memory import TorchReplayBuffer
import torch
import random
import numpy as np
from torch import from_numpy


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size
        self.exp_eps = 1
        self.memory = TorchReplayBuffer(config.mem_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.online_model = None
        self.target_model = None
        self.optimizer = None

    def choose_action(self, state):
        if random.random() < self.exp_eps:
            return random.randint(0, self.config.env.n_actions - 1)
        else:
            state = np.expand_dims(state, axis=0)
            state = from_numpy(state).byte().to(self.device)
            with torch.no_grad():
                q_values = self.online_model.get_qvalues(state).cpu()
            return torch.argmax(q_values, -1).item()

    def store(self, state, reward, done, action, next_state):
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
        self.memory.add(state, reward, done, action, next_state)

    def unpack_batch(self, batch):
        # batch is a TensorDict returned from self.buffer.sample(batch_size)
        device = self.device

        states = batch["state"].to(device)
        actions = batch["action"].to(device)
        rewards = batch["reward"].unsqueeze(-1).to(device)  # shape: [batch, 1]
        next_states = batch["next_state"].to(device)
        dones = batch["done"].unsqueeze(-1).to(device)  # convert bool to float if needed

        return states, actions, rewards, next_states, dones

    def hard_target_update(self):
        self.target_model.load_state_dict(self.online_model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

    def prepare_to_play(self):
        self.online_model.eval()

    def train(self):
        raise NotImplementedError

