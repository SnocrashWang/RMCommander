import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np
from typing import List, Dict, Any, Tuple
from base.config.robot_config import BASE_ROBOT_TYPE_LIST
from base.environment import Action
from utils.config.game_config import GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(PolicyNet, self).__init__()
        # 共享特征提取层
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU()
        )
        
        # 速度分支（连续动作）
        self.velocity_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 2),
            nn.Tanh()  # 添加tanh激活函数
        )
        self.velocity_log_std = nn.Parameter(torch.full((2,), np.log(0.05)))
        
        # 攻击目标分支（离散动作）
        self.attack_target_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, len(RobotType))  # 输出可能目标的logits
        )

    def forward(self, x):
        features = self.shared_network(x)
        
        # 速度（连续）
        velocity_mean = self.velocity_mean(features)
        velocity_std = torch.exp(self.velocity_log_std)
        
        # 目标选择（离散）
        attack_target_logits = self.attack_target_network(features)
        
        return velocity_mean, velocity_std, attack_target_logits


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x)

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        actor_lr = 1e-5,
        critic_lr = 1e-4,
        gamma = 0.98,
        lmbda = 0.95,
        epochs = 4,
        eps = 0.1,
        batch_size = 16,
        device: str = None
    ):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.batch_size = batch_size
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        self.actor = PolicyNet(state_dim).to(self.device)
        self.critic = ValueNet(state_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
    
    @torch.no_grad()
    def take_action(self, state, team: GameTeam) -> Dict[str, Action]:
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        velocity_mean, velocity_std, attack_target_logits = self.actor(state)
        # print(velocity_mean, velocity_std)
        # 速度
        velocity_dist = Normal(velocity_mean, velocity_std)
        velocity_action = velocity_dist.sample()
        velocity_norm = torch.norm(velocity_action)
        velocity_action = (velocity_action / velocity_norm) if velocity_norm != 0 else torch.tensor([0, 0])
        # 目标选择
        attack_target_dist = Categorical(logits=attack_target_logits)
        attack_target_action = attack_target_dist.sample()

        actions = {
            ROBOT_ID[team][robot_type]: Action(
                velocity=velocity_action.cpu().numpy(),
                attack_target=attack_target_action.item()
            ) for robot_type in BASE_ROBOT_TYPE_LIST
        }

        return actions

    def update(self, transition_dicts):
        """更新策略，支持多个rollout和shuffle"""
        # 合并所有rollout的数据
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        # 记录每个rollout的长度
        rollout_lengths = []
        
        for trans_dict in transition_dicts:
            all_states.append(trans_dict['states'])
            all_actions.append(trans_dict['actions'])
            all_rewards.append(trans_dict['rewards'])
            all_next_states.append(trans_dict['next_states'])
            all_dones.append(trans_dict['dones'])
            rollout_lengths.append(len(trans_dict['dones']))
        
        # 转换为张量
        states = torch.tensor(np.vstack(all_states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.vstack(all_actions), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.hstack(all_rewards), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.vstack(all_next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.hstack(all_dones), dtype=torch.float).view(-1, 1).to(self.device)
        
        # 计算TD目标和优势函数
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - self.critic(states)
            # 分割每个rollout的td_delta
            td_delta_split = torch.split(td_delta, rollout_lengths)
            advantages_list = []
            for td in td_delta_split:
                # 对每个rollout独立计算优势
                advantages_list.append(compute_advantage(self.gamma, self.lmbda, td))
            # 合并所有rollout的优势函数
            advantage = torch.cat(advantages_list, dim=0)
            old_log_probs = self._get_log_probs(states, actions)  # 使用detach避免梯度冲突

        # 获取总样本数并创建索引
        num_samples = states.size(0)
        indices = np.arange(num_samples)
        
        # 训练多个epoch，每个epoch都shuffle数据
        for _ in range(self.epochs):
            # 打乱数据索引
            np.random.shuffle(indices)
            
            # 创建batches
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                if end > num_samples:
                    end = num_samples  # 处理最后不完整的batch
                    
                # 获取当前batch的索引
                batch_indices = indices[start:end]
                if isinstance(batch_indices, np.ndarray):
                    batch_indices = torch.from_numpy(batch_indices).long().to(self.device)
                
                # 从大数组中提取batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_td_target = td_target[batch_indices]
                batch_advantage = advantage[batch_indices]
                
                # 计算PPO损失
                log_probs = self._get_log_probs(batch_states, batch_actions)
                log_ratio = log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算critic损失
                critic_values = self.critic(batch_states)
                critic_loss = F.mse_loss(critic_values, batch_td_target.detach())
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def _get_log_probs(self, states, actions):
        velocity_mean, velocity_std, attack_target_logits = self.actor(states)

        velocity_action_dists = torch.distributions.Normal(velocity_mean, velocity_std)
        attack_target_action_dists = torch.distributions.Categorical(logits=attack_target_logits)

        # 提取动作的不同部分
        velocity_actions = actions[:, 0:2]
        attack_target_actions = actions[:, 2]

        # 计算各个动作的log概率
        # 对于正态分布，log_prob会返回与输入相同形状的张量
        velocity_log_probs = velocity_action_dists.log_prob(velocity_actions)
        attack_target_log_probs = attack_target_action_dists.log_prob(attack_target_actions)

        # 将所有log概率相加
        log_probs = (velocity_log_probs.sum(dim=-1) # 将速度的log概率在最后一个维度上求和（因为它是2维的）
                     + attack_target_log_probs)
        return log_probs
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.squeeze(-1)
    advantages = torch.zeros_like(td_delta)
    # 反向遍历
    advantage = 0.0
    for t in reversed(range(td_delta.size(0))):
        advantage = gamma * lmbda * advantage + td_delta[t]
        advantages[t] = advantage
    return advantages.unsqueeze(-1)
