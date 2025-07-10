import torch
from models import FQFModel, FractionProposalModel
from common import huber_loss
from .base_agent import BaseAgent
from torchrl.data.replay_buffers.samplers import PrioritizedSampler


class FQF(BaseAgent):
    def __init__(self, config):
        super(FQF, self).__init__(config)

        # online and target networks
        online_model = FQFModel(
                config
        ).to(self.device)

        self.set_models(online_model)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(),
            config.agent.lr,
            eps=config.adam_eps
        )
        self.fp_optimizer = torch.optim.RMSprop(
            self.online_model.fp_layer.parameters(),
            config.agent.fp_lr
        )

    def train(self):
        # wait until buffer is warmed up
        if len(self.memory) < self.config.init_mem_size_to_train:
            return {"loss/total": 0}

        # sample a batch + replay info
        batch, info = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self.unpack_batch(batch)  # noqa
        # ----------------------------------------------------------------------------
        # rescale rewards if needed
        rewards = rewards * self.config.reward_scale
        # ----------------------------------------------------------------------------

        # compute taus for quantile networks
        taus, tau_hats, ent, raw_probs, clamped_probs = self.online_model.get_taus(states)

        # build distributional Bellman target
        with torch.no_grad():
            next_z = self.target_model((next_states, tau_hats))
            next_qvalues = self.target_model.get_qvalues(next_states)
            next_actions = torch.argmax(next_qvalues, dim=-1)
            next_actions = next_actions[..., None, None].expand(self.batch_size, self.config.agent.N, 1)
            next_z = next_z.gather(dim=-1, index=next_actions).squeeze(-1)
            target_z = rewards + self.config.gamma * (~dones) * next_z

            # clamp targets to a fixed support
            V_MAX = self.config.target_support  # e.g. 200
            target_z = torch.clamp(target_z, -V_MAX, V_MAX)

        # current quantile predictions
        z_pred = self.online_model((states, tau_hats.detach()))
        a_idx = actions[..., None, None].expand(self.batch_size, self.config.agent.N, 1).long()
        z_pred = z_pred.gather(dim=-1, index=a_idx)

        # TD Errors
        # inputs are (batch_size, singleton “quantile” dimension, infered dimension)
        delta  = target_z.view(self.batch_size, 1, -1) - z_pred

        # quantile Huber loss
        hloss = huber_loss(delta, self.config.agent.kappa)
        rho = (
            torch.abs(tau_hats[..., None].detach() - (delta.detach() < 0).float())
            * hloss
            / self.config.agent.kappa
        )

        # collapse to one loss per sample
        sample_loss = rho.sum(dim=1).mean(dim=1)# sum over N -> mean over N -> mean over batch

        # branch on replay type
        if isinstance(self.memory.sampler, PrioritizedSampler):
            # importance-sampling weights & indices
            is_weights = torch.as_tensor(info["_weight"], device=self.device)
            # clamp them so no single sample blows up
            is_weights = torch.clamp(is_weights, max=self.config.is_weight_max)
            #is_weights = torch.tensor(info["_weight"], device=self.device)  # [batch]
            indices    = info["index"]                                      # np.ndarray

            # apply weights
            loss = (sample_loss * is_weights).mean()

            # update priorities
            td_errors = delta.abs().mean(dim=list(range(1, delta.ndim)))
            self.memory.update_priorities(td_errors.detach(), indices)
        else:
            # uniform sampling
            loss = sample_loss.mean()

        # optimizer step + gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
                self.online_model.parameters(),
                max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()

        # fraction-proposal update (same as before)
        with torch.no_grad():
            z        = self.online_model((states, taus[:, 1:-1]))
            z_hat    = self.online_model((states, tau_hats[:, 1:]))
            z_hat_1  = self.online_model((states, tau_hats[:, :-1]))
            a_fp     = actions[..., None, None].expand(
                self.batch_size, self.config.agent.N - 1, 1
            ).long()
            z        = z.gather(dim=-1, index=a_fp)
            z_hat    = z_hat.gather(dim=-1, index=a_fp)
            z_hat_1  = z_hat_1.gather(dim=-1, index=a_fp)
            fp_grads = 2 * z - z_hat - z_hat_1

        fp_loss = (taus[:, 1:-1] * fp_grads.squeeze(-1)).sum(-1).mean(0)
        lmda = self.config.agent.lmda 
        fp_loss - lmda * ent

        self.fp_optimizer.zero_grad()
        fp_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.fp_layer.parameters(),
                              max_norm=self.config.max_grad_norm)
        self.fp_optimizer.step()

        # ----------------------------------------------------------------------------
        # soft‐update the target network every train() call
        self.soft_target_update()    # uses self.config.soft_tau by default
        # ----------------------------------------------------------------------------


        metrics = {
        "loss/total": loss.item(),
        "loss/huber": hloss.mean().item(),
        "loss/fraction_proposal": fp_loss.item(),
        "entropy/taus": ent.mean().item(),
        "epsilon": self.exp_eps,
        "td_error/mean": delta.abs().mean().item(),
        "td_error/std": delta.std().item(),
        "quantile/spread": (z_pred.max() - z_pred.min()).mean().item(),
        "quantile/mean": z_pred.mean().item(),
        "grad/total_norm": total_norm,
        "param_norm/online_model": sum(p.norm().item() for p in self.online_model.parameters() if p.requires_grad),
        "replay/fill": len(self.memory) / self.config.mem_size,
        "replay/size": len(self.memory),
        "reward/mean": rewards.mean().item(),
        "reward/std": rewards.std().item(),
        "lr/fp_optimizer_lr": self.optimizer.param_groups[0]["lr"],
        "lr/optimizer": self.fp_optimizer.param_groups[0]["lr"],
        }

        # target‐online drift
        drift = sum((p_t - p).norm().item()
            for p_t, p in zip(self.target_model.parameters(),
                              self.online_model.parameters()))
        metrics["target_online_drift"] = drift

        fp_norm = sum(p.grad.norm().item()
              for p in self.online_model.fp_layer.parameters()
              if p.grad is not None)
        metrics["grad/fp_layer_norm"] = fp_norm

        # diagnostics on the *uncamped* distribution
        metrics["tau/raw_min"]  = raw_probs.min().item()
        metrics["tau/raw_max"]  = raw_probs.max().item()
        metrics["tau/raw_zero%"]= (raw_probs < 1e-6).float().mean().item()

        # diagnostics on the *clamped* distribution
        metrics["tau/clamped_min"] = clamped_probs.min().item()
        metrics["tau/clamped_max"] = clamped_probs.max().item()
        metrics["tau/taus_range"]     = (taus.max() - taus.min()).item()
        metrics["tau/tau_hat_spread"] = (tau_hats.max() - tau_hats.min()).item()
        metrics["tau/probs_mean"] = clamped_probs.mean().item()
        metrics["tau/probs_std"]  = clamped_probs.std().item()

        # Bellman‐target stats (use the `target_z` you already computed)
        metrics.update({
        "target_z/mean": target_z.mean().item(),
        "target_z/std":  target_z.std().item(),
        "target_z/min":  target_z.min().item(),
        "target_z/max":  target_z.max().item(),
        })

        if isinstance(self.memory.sampler, PrioritizedSampler):
            metrics["is_weight/mean"] = is_weights.mean().item()
            metrics["is_weight/max"]  = is_weights.max().item()
        return metrics
