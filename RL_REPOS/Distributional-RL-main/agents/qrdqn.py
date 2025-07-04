import torch
from models import QRDQNModel
from common import huber_loss
from .base_agent import BaseAgent


class QRDQN(BaseAgent):
    def __init__(self, config):
        super(QRDQN, self).__init__(config)
        self.online_model = QRDQNModel(config.state_shape,
                                       config.env.n_actions,
                                       config.agent.N,
                                       ).to(self.device)
        self.target_model = QRDQNModel(config.state_shape,
                                       config.env.n_actions,
                                       config.agent.N,
                                       ).to(self.device)
        self.hard_target_update()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(),
                                          self.config.agent.lr,
                                          eps=self.config.adam_eps
                                          )
        taus = torch.arange(0,
                            config.agent.N + 1,
                            device=self.device,
                            dtype=torch.float32
                            ) / config.agent.N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, config.agent.N)

    def train(self):
        if len(self.memory) < self.config.init_mem_size_to_train:
            return 0
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)  # noqa

        with torch.no_grad():
            next_thetas = self.target_model(next_states)
            next_qvalues = self.target_model.get_qvalues(next_states)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, self.config.agent.N, 1)
            next_thetas = next_thetas.gather(dim=-1, index=next_actions).squeeze(-1)
            target_theta = rewards + self.config.gamma * (~dones) * next_thetas

        current_theta = self.online_model(states)
        actions = actions[..., None, None].expand(self.batch_size, self.config.agent.N, 1).long()
        current_theta = current_theta.gather(dim=-1, index=actions)

        u = target_theta.view(target_theta.size(0), 1, target_theta.size(-1)) - current_theta
        hloss = huber_loss(u, self.config.agent.kappa)
        rho = torch.abs(self.tau_hats[..., None] - (u.detach() < 0).float()) * hloss / self.config.agent.kappa
        loss = rho.sum(1).mean(1).mean()  # sum over N -> mean over N -> mean over batch

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

