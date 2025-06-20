import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple
from base.environment import Action
from base.config.robot_config import BASE_ROBOT_TYPE_LIST
from utils.config.game_config import GameTeam
from utils.config.robot_config import ROBOT_ID, RobotType
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
            nn.Linear(64, 2),
            nn.Tanh()  # 添加tanh激活函数
        )
        self.navigation_std = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出x,y的标准差
            nn.Softplus(),  # 确保标准差为正
            nn.Tanh()  # 添加tanh激活函数
        )
        
        # 攻击分支（离散动作）
        self.attack_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出是否攻击的logits
        )
        
        # 目标选择分支（离散动作）
        # 输出维度为1（NONE）+ len(BASE_ROBOT_TYPE_LIST)
        self.target_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1 + len(BASE_ROBOT_TYPE_LIST))  # 输出可能目标的logits
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
        nav_std = self.navigation_std(features) + 1e-6  # 添加小值确保标准差为正
        
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
        team: GameTeam,
        state_size: int,
        field_width: float,
        field_height: float,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        device: str = None  # 新增设备参数
    ):
        self.team = team
        self.state_size = state_size
        self.field_width = field_width
        self.field_height = field_height
        self.target_robot_list = [RobotType.NONE] + BASE_ROBOT_TYPE_LIST
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 创建网络并移动到指定设备
        self.network = PPONetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # 存储轨迹
        self.states = []
        self.actions = []
        self.raw_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.old_dist_params = []  # 存储旧策略的分布参数
    
    def _process_network_output(self, nav_mean, nav_std, attack_logits, target_logits) -> Tuple[Dict[str, Action], torch.Tensor]:
        """将网络输出转换为实际动作，并返回原始采样动作用于概率计算"""
        # 处理导航坐标
        nav_dist = Normal(nav_mean, nav_std)
        nav_action = nav_dist.sample()

        # 将输出映射到场地范围内
        x = (torch.tanh(nav_action[0]) + 1) * self.field_width / 2
        y = (torch.tanh(nav_action[1]) + 1) * self.field_height / 2
        
        # 处理攻击决策
        attack_dist = Categorical(logits=attack_logits)
        attack_action = attack_dist.sample()
        should_attack = attack_action.item() == 1
        
        # 处理目标选择
        target_dist = Categorical(logits=target_logits)
        target_action = target_dist.sample()
        target_id = self.target_robot_list[target_action.item()]
        
        actions = {
            ROBOT_ID[self.team][robot_type]: Action(
                navigation=(x.item(), y.item()),
                attack=should_attack,
                target=target_id
            ) for robot_type in BASE_ROBOT_TYPE_LIST
        }
        
        # 返回动作和原始采样值
        return actions, nav_action, attack_action, target_action
    
    def act(self, state: np.ndarray) -> Dict[str, Action]:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nav_mean, nav_std, attack_logits, target_logits, value = self.network(state_tensor)
            action, nav_action, attack_action, target_action = self._process_network_output(nav_mean[0], nav_std[0], attack_logits[0], target_logits[0])
            
            # 计算动作概率
            nav_dist = Normal(nav_mean[0], nav_std[0])
            attack_dist = Categorical(logits=attack_logits[0])
            target_dist = Categorical(logits=target_logits[0])
            
            # 计算每个机器人的动作概率
            log_probs = []
            for robot_id, robot_action in action.items():
                # 使用原始采样的导航动作计算log概率
                nav_log_prob = nav_dist.log_prob(nav_action).sum()
                
                # 计算攻击动作的log概率
                attack_log_prob = attack_dist.log_prob(attack_action)
                
                # 计算目标选择的log概率
                target_log_prob = target_dist.log_prob(target_action)
                
                # 合并所有log概率
                robot_log_prob = nav_log_prob + attack_log_prob + target_log_prob
                log_probs.append(robot_log_prob)
            
            # 所有机器人的平均对数概率
            log_prob = torch.stack(log_probs).mean()
        
        # 存储轨迹
        self.states.append(state)
        self.actions.append(action)
        self.raw_actions.append((nav_action, attack_action, target_action))
        self.values.append(value.item())
        self.log_probs.append(log_prob.item())
        
        # 保存旧策略的分布参数（用于KL散度计算）
        self.old_dist_params.append({
            'nav_mean': nav_mean[0].detach().clone(),
            'nav_std': nav_std[0].detach().clone(),
            'attack_logits': attack_logits[0].detach().clone(),
            'target_logits': target_logits[0].detach().clone()
        })
        
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
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def update(self):
        """更新策略"""
        if len(self.states) < self.batch_size:
            return
        
        # 计算优势
        advantages = self.compute_gae()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)  # 先转换为numpy数组
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.values).to(self.device)
        
        # 多轮更新
        for _ in range(self.update_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(self.states))
            
            for start_idx in range(0, len(self.states), self.batch_size):
                batch_indices = indices[start_idx:min(start_idx + self.batch_size, len(indices))]
                
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
                batch_actions = [self.actions[i] for i in batch_indices]
                batch_raw_actions = [self.raw_actions[i] for i in batch_indices]
                batch_old_params = [self.old_dist_params[i] for i in batch_indices]
                batch_log_probs = []
                
                # 计算KL散度
                kl_divs = []
                
                for actions, raw_actions, old_params in zip(batch_actions, batch_raw_actions, batch_old_params):
                    robot_log_probs = []
                    nav_action, attack_action, target_action = raw_actions
                    
                    for robot_id, robot_action in actions.items():
                        # 使用存储的原始采样动作计算log概率
                        nav_log_prob = nav_dist.log_prob(nav_action).sum()
                        
                        # 计算攻击动作的log概率
                        attack_log_prob = attack_dist.log_prob(attack_action)
                        
                        # 计算目标选择的log概率
                        target_log_prob = target_dist.log_prob(target_action)
                        
                        # 合并所有log概率
                        robot_log_prob = nav_log_prob + attack_log_prob + target_log_prob
                        robot_log_probs.append(robot_log_prob)
                    
                    batch_log_probs.append(torch.stack(robot_log_probs).mean())
                    
                    # 计算KL散度
                    old_nav_dist = Normal(old_params['nav_mean'], old_params['nav_std'])
                    old_attack_dist = Categorical(logits=old_params['attack_logits'])
                    old_target_dist = Categorical(logits=old_params['target_logits'])
                    
                    nav_kl = torch.distributions.kl_divergence(old_nav_dist, nav_dist).mean()
                    attack_kl = torch.distributions.kl_divergence(old_attack_dist, attack_dist).mean()
                    target_kl = torch.distributions.kl_divergence(old_target_dist, target_dist).mean()
                    
                    kl_divs.append(nav_kl + attack_kl + target_kl)
                
                batch_log_probs = torch.stack(batch_log_probs)
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                
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
                
                # 计算梯度范数（在裁剪之前）
                total_norm = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                # if total_norm > self.max_grad_norm:
                #     print(f"梯度被裁剪: {total_norm:.4f}")
                
                self.optimizer.step()
                
                # 检查KL散度早停
                avg_kl_div = torch.stack(kl_divs).mean().item()
                if avg_kl_div > self.target_kl:
                    # print(f"KL散度早停: {avg_kl_div:.4f} > {self.target_kl}")
                    break
        
        # 清空轨迹
        self.states = []
        self.actions = []
        self.raw_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.old_dist_params = []
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device)  # 保存设备信息
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
