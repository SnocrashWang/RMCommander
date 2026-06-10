from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch.distributions import Categorical, Normal

from rules.base.config.robot_config import BASE_ROBOT_TYPE_LIST
from rules.base.environment import ActionBase
from utils.config.game_config import GameTeam
from utils.config.robot_config import ROBOT_ID


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_schema: Dict[str, spaces.Space]):
        super().__init__()
        self.action_schema = action_schema

        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
        )

        self.box_heads = nn.ModuleDict()
        self.box_log_stds = nn.ParameterDict()
        self.discrete_heads = nn.ModuleDict()

        for name, space in self.action_schema.items():
            if isinstance(space, spaces.Box):
                action_dim = int(np.prod(space.shape))
                self.box_heads[name] = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ELU(),
                    nn.Linear(64, action_dim),
                )
                self.box_log_stds[name] = nn.Parameter(torch.full((action_dim,), np.log(0.05)))
            elif isinstance(space, spaces.Discrete):
                self.discrete_heads[name] = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ELU(),
                    nn.Linear(64, space.n),
                )
            else:
                raise TypeError(f"Unsupported action space for {name}: {space}")

    def forward(self, x):
        features = self.shared_network(x)
        outputs = {}

        for name, space in self.action_schema.items():
            if isinstance(space, spaces.Box):
                raw_mean = self.box_heads[name](features)
                mean = self._scale_box_mean(raw_mean, space)
                std = torch.exp(self.box_log_stds[name])
                outputs[name] = {
                    "type": "box",
                    "space": space,
                    "mean": mean,
                    "std": std,
                }
            elif isinstance(space, spaces.Discrete):
                outputs[name] = {
                    "type": "discrete",
                    "space": space,
                    "logits": self.discrete_heads[name](features),
                }

        return outputs

    @staticmethod
    def _scale_box_mean(raw_mean, space: spaces.Box):
        low = torch.as_tensor(space.low.reshape(-1), dtype=raw_mean.dtype, device=raw_mean.device)
        high = torch.as_tensor(space.high.reshape(-1), dtype=raw_mean.dtype, device=raw_mean.device)
        return (torch.tanh(raw_mean) + 1) * (high - low) / 2 + low


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.network(x)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        actor_lr=5e-5,
        critic_lr=5e-4,
        gamma=0.98,
        lmbda=0.95,
        epochs=4,
        eps=0.1,
        batch_size=16,
        device: str = None,
        robot_type_list=None,
        action_cls=ActionBase,
        action_kwargs=None,
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.batch_size = batch_size
        self.robot_type_list = list(robot_type_list or BASE_ROBOT_TYPE_LIST)
        self.action_cls = action_cls
        self.action_schema = action_cls._schema
        self.action_kwargs = action_kwargs or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")

        self.actor = PolicyNet(state_dim, self.action_schema).to(self.device)
        self.critic = ValueNet(state_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    @torch.no_grad()
    def take_action(self, state, team: GameTeam, deterministic: bool = False):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        policy_outputs = self.actor(state)

        actions = {}
        for robot_type in self.robot_type_list:
            action_values = {}
            for name, output in policy_outputs.items():
                if output["type"] == "box":
                    value = output["mean"] if deterministic else Normal(output["mean"], output["std"]).sample()
                    value = self._clamp_box_value(value, output["space"])
                    if team == GameTeam.BLUE and name == "navigation_target_norm":
                        value = value * -1
                    action_values[name] = value.reshape(output["space"].shape).cpu().numpy()
                elif output["type"] == "discrete":
                    if deterministic:
                        value = torch.argmax(output["logits"], dim=-1)
                    else:
                        value = Categorical(logits=output["logits"]).sample()
                    action_values[name] = value.item()

            action_values.update(self.action_kwargs)
            actions[ROBOT_ID[team][robot_type]] = self.action_cls(**action_values)

        return actions

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict["states"]), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict["actions"]), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict["rewards"]), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict["next_states"]), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict["dones"]), dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta)
            advantage_std = advantage.std(unbiased=False)
            advantage = (advantage - advantage.mean()) / (advantage_std + 1e-8)
            old_log_probs, _ = self._get_log_probs(states, actions)

        for _ in range(self.epochs):
            log_probs, entropy = self._get_log_probs(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2) - 0.01 * entropy)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def update_multi_rollout(self, transition_dicts):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        rollout_lengths = []

        for trans_dict in transition_dicts:
            all_states.append(trans_dict["states"])
            all_actions.append(trans_dict["actions"])
            all_rewards.append(trans_dict["rewards"])
            all_next_states.append(trans_dict["next_states"])
            all_dones.append(trans_dict["dones"])
            rollout_lengths.append(len(trans_dict["dones"]))

        states = torch.tensor(np.vstack(all_states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.vstack(all_actions), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.hstack(all_rewards), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.vstack(all_next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.hstack(all_dones), dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            td_delta_split = torch.split(td_delta, rollout_lengths)
            advantages_list = [compute_advantage(self.gamma, self.lmbda, td) for td in td_delta_split]
            advantage = torch.cat(advantages_list, dim=0)
            advantage_std = advantage.std(unbiased=False)
            advantage = (advantage - advantage.mean()) / (advantage_std + 1e-8)
            old_log_probs, _ = self._get_log_probs(states, actions)

        num_samples = states.size(0)
        indices = np.arange(num_samples)

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = torch.from_numpy(indices[start:end]).long().to(self.device)

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_td_target = td_target[batch_indices]
                batch_advantage = advantage[batch_indices]

                log_probs, entropy = self._get_log_probs(batch_states, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantage
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
                critic_loss = F.mse_loss(self.critic(batch_states), batch_td_target.detach())

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def _get_log_probs(self, states, actions):
        policy_outputs = self.actor(states)
        single_action_dim = self._single_action_dim()
        if actions.shape[1] % single_action_dim != 0:
            raise ValueError(f"action shape {actions.shape} is not divisible by schema dim {single_action_dim}")

        log_probs = torch.zeros(states.shape[0], device=self.device)
        entropy = torch.zeros(states.shape[0], device=self.device)

        for robot_offset in range(0, actions.shape[1], single_action_dim):
            field_offset = robot_offset
            for name, space in self.action_schema.items():
                output = policy_outputs[name]

                if isinstance(space, spaces.Box):
                    field_dim = int(np.prod(space.shape))
                    value = actions[:, field_offset:field_offset + field_dim]
                    dist = Normal(output["mean"], output["std"])
                    log_probs = log_probs + dist.log_prob(value).sum(dim=-1)
                    entropy = entropy + dist.entropy().sum(dim=-1)
                    field_offset += field_dim
                elif isinstance(space, spaces.Discrete):
                    value = actions[:, field_offset].long()
                    dist = Categorical(logits=output["logits"])
                    log_probs = log_probs + dist.log_prob(value)
                    entropy = entropy + dist.entropy()
                    field_offset += 1

        return log_probs.unsqueeze(-1), entropy.unsqueeze(-1)

    def _single_action_dim(self):
        dim = 0
        for space in self.action_schema.values():
            if isinstance(space, spaces.Box):
                dim += int(np.prod(space.shape))
            elif isinstance(space, spaces.Discrete):
                dim += 1
            else:
                raise TypeError(f"Unsupported action space: {space}")
        return dim

    @staticmethod
    def _clamp_box_value(value, space: spaces.Box):
        low = torch.as_tensor(space.low.reshape(-1), dtype=value.dtype, device=value.device)
        high = torch.as_tensor(space.high.reshape(-1), dtype=value.dtype, device=value.device)
        return torch.max(torch.min(value.reshape(-1), high), low)

    def save(self, path: str):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self._load_matching_state_dict(self.actor, checkpoint["actor_state_dict"])
        self._load_matching_state_dict(self.critic, checkpoint["critic_state_dict"])
        try:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        except ValueError:
            print("优化器状态与当前模型结构不匹配，已跳过优化器状态加载")

    @staticmethod
    def _load_matching_state_dict(module, state_dict):
        current_state = module.state_dict()
        matched_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and current_state[key].shape == value.shape
        }
        current_state.update(matched_state)
        module.load_state_dict(current_state)


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.squeeze(-1)
    advantages = torch.zeros_like(td_delta)
    advantage = 0.0
    for t in reversed(range(td_delta.size(0))):
        advantage = gamma * lmbda * advantage + td_delta[t]
        advantages[t] = advantage
    return advantages.unsqueeze(-1)
