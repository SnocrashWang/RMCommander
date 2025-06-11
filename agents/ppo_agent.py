import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple
from utils.config.robot_config import RobotType
from RMUL.config.robot_config import ROBOT_TYPE_RMUL
from torch.distributions import Normal, Categorical

class PPONetwork(nn.Module):
    def __init__(self, state_size: int):
        super(PPONetwork, self).__init__()
        
        # 共享特征提取层
        self.shared_network = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 导航分支（连续动作）
        self.navigation_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出x,y的均值
        )
        self.navigation_std = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出x,y的标准差
        )
        
        # 攻击分支（离散动作）
        self.attack_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出是否攻击的logits
        )
        
        # 目标选择分支（离散动作）
        self.target_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 输出可能目标的logits
        )
        
        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        features = self.shared_network(state)
        
        # 导航动作（连续）
        nav_mean = self.navigation_mean(features)
        nav_std = torch.exp(self.navigation_std(features))  # 确保标准差为正
        
        # 攻击动作（离散）
        attack_logits = self.attack_network(features)
        
        # 目标选择（离散）
        target_logits = self.target_network(features)
        
        # 状态价值
        value = self.value_network(features)
        
        return nav_mean, nav_std, attack_logits, target_logits, value

class PPOAgent:
    def __init__(
        self,
        state_size: int,
        action_space: Dict[str, Tuple],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64
    ):
        self.state_size = state_size
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 创建网络
        self.network = PPONetwork(state_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # 定义可用的目标机器人ID列表
        self.available_targets = [RobotType.NONE] + ROBOT_TYPE_RMUL
    
    def _process_network_output(self, nav_mean, nav_std, attack_logits, target_logits) -> Dict[str, Any]:
        """将网络输出转换为实际动作"""
        # 处理导航坐标
        nav_dist = Normal(nav_mean, nav_std)
        nav_action = nav_dist.sample()
        x = torch.sigmoid(nav_action[0]) * self.action_space['navigation'][1]
        y = torch.sigmoid(nav_action[1]) * self.action_space['navigation'][3]
        
        # 处理攻击决策
        attack_dist = Categorical(logits=attack_logits)
        attack_action = attack_dist.sample()
        should_attack = attack_action.item() == 1
        
        # 处理目标选择
        target_dist = Categorical(logits=target_logits)
        target_action = target_dist.sample()
        target_id = self.available_targets[target_action.item()]
        
        return {
            'navigation': (x.item(), y.item()),
            'attack': int(should_attack),
            'target': target_id
        }
    
    def _get_target_index(self, target_id: RobotType) -> int:
        """获取目标机器人在可用目标列表中的索引"""
        return self.available_targets.index(target_id)
    
    def act(self, state: np.ndarray) -> Dict[str, Any]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            nav_mean, nav_std, attack_logits, target_logits, value = self.network(state_tensor)
            action = self._process_network_output(nav_mean[0], nav_std[0], attack_logits[0], target_logits[0])
            
            # 计算动作概率
            nav_dist = Normal(nav_mean[0], nav_std[0])
            attack_dist = Categorical(logits=attack_logits[0])
            target_dist = Categorical(logits=target_logits[0])
            
            log_prob = (
                nav_dist.log_prob(torch.tensor([action['navigation'][0], action['navigation'][1]]))
                + attack_dist.log_prob(torch.tensor(action['attack']))
                + target_dist.log_prob(torch.tensor(self._get_target_index(action['target'])))
            ).sum()
        
        # 存储轨迹
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        return action
    
    def store_reward(self, reward: float, done: bool):
        """存储奖励和完成状态"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        """更新策略"""
        if len(self.states) < self.batch_size:
            return
        
        # 计算优势
        advantages = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(self.states)
        old_log_probs = torch.FloatTensor(self.log_probs)
        old_values = torch.FloatTensor(self.values)
        
        # 多轮更新
        for _ in range(self.update_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(self.states))
            
            for start_idx in range(0, len(self.states), self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # 获取批次数据
                batch_states = states[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # 前向传播
                nav_mean, nav_std, attack_logits, target_logits, values = self.network(batch_states)
                
                # 计算新的动作概率
                nav_dist = Normal(nav_mean, nav_std)
                attack_dist = Categorical(logits=attack_logits)
                target_dist = Categorical(logits=target_logits)
                
                # 计算策略损失
                ratio = torch.exp(
                    nav_dist.log_prob(batch_states[:, :2]).sum(dim=1) +
                    attack_dist.log_prob(torch.tensor([a['attack'] for a in self.actions])[batch_indices]) +
                    target_dist.log_prob(torch.tensor([self._get_target_index(a['target']) for a in self.actions])[batch_indices])
                )
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 计算价值损失
                value_loss = nn.MSELoss()(values.squeeze(), batch_old_values + batch_advantages)
                
                # 计算熵损失
                entropy_loss = -(
                    nav_dist.entropy().mean() +
                    attack_dist.entropy().mean() +
                    target_dist.entropy().mean()
                )
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 