import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Optional, Union

import wandb

from envs.ev_charging.ev_charging import EVCharging
from morl.common.evaluation import seed_everything, log_episode_info, multi_policy_evaluation, single_policy_evaluation
from morl.common.morl_algorithm import MOAgent, MOPolicy
from morl.common.networks import mlp, layer_init
from morl.common.prioritized_buffer import PrioritizedReplayBuffer
from morl.multi_policy.mo_ddpg.mo_ddpg import EarlyStopping, OrnsteinUhlenbeckNoise


class QNetwork(nn.Module):
    '''Critic Network for DDPG.'''

    def __init__(self, obs_dim: int, action_dim: int, net_arch=[1024, 1024, 1024]):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, net_arch)
        self.apply(layer_init)
        # self.net = nn.Sequential(
        #    nn.Linear(obs_dim + action_dim, net_arch[0]),
        #    nn.ReLU(),
        #    nn.Linear(net_arch[0], net_arch[1]),
        #    nn.ReLU(),
        #    nn.Linear(net_arch[1], 1),
        #    nn.ReLU(),
        #    nn.Linear(net_arch[2], action_dim)
        # )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


class Actor(nn.Module):
    '''Actor Network for DDPG.'''

    def __init__(self, obs_dim: int, action_dim: int, action_space: any, net_arch=[1024, 1024, 1024]):
        super().__init__()
        self.net = mlp(obs_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], action_dim)

        self.register_buffer('action_scale',
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias',
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

        self.apply(layer_init)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        x = self.mean(x)
        return torch.tanh(x) * self.action_scale + self.action_bias

    def get_action(self, obs):
        return self.forward(obs)


class DDPGAgent(MOAgent, MOPolicy):
    def __init__(
            self,
            envs,
            learning_rate: float = 1e-4,
            gamma: float = 0.99,
            tau: float = 0.005,
            buffer_size: int = 1000000,
            net_arch: List = [512, 512],
            batch_size: int = 256,
            learning_starts: int = 1000,
            per: bool = True,
            per_alpha: float = 0.6,
            policy_frequency: int = 1,
            env_iterations: int = 10,
            project_name: str = 'MORL-Baselines',
            experiment_name: str = 'DDPG',
            wandb_entity: Optional[str] = None,
            log: bool = True,
            seed: Optional[int] = None,
            device: Union[torch.device, str] = 'auto',
    ):
        if seed is not None:
            seed_everything(seed)
        self.rnd = random.Random(seed)

        MOAgent.__init__(self, envs[0], device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        self.envs = envs
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_starts = learning_starts
        self.per = per
        self.per_alpha = per_alpha
        self.batch_size = batch_size
        self.policy_frequency = policy_frequency
        self.env_iterations = env_iterations
        self.seed = seed

        # Q network (Critic)
        self.qf1 = QNetwork(obs_dim=self.observation_dim, action_dim=self.action_dim,
                            net_arch=net_arch).to(self.device)
        self.qf1_target = deepcopy(self.qf1)
        self.q_optimizer = optim.Adam(self.qf1.parameters(), lr=learning_rate)

        # Actor network
        self.actor = Actor(obs_dim=self.observation_dim, action_dim=self.action_dim,
                           action_space=self.action_space, net_arch=net_arch).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(self.observation_shape, self.action_dim,
                                                     rew_dim=self.reward_dim,
                                                     max_size=self.buffer_size, obs_dtype=np.float32,
                                                     action_dtype=np.float32, min_priority=1e-5)

        # noise
        self.exploration_noise = OrnsteinUhlenbeckNoise(size=self.action_dim)

        self.early_stopper = EarlyStopping(patience=5, restarts=2, min_delta=0.01)

        self.log = log
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)
        self.save_dir = f'models/ddpg'
        self.prev_actor_loss = 0

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, w, rewards, next_states, dones, indices = self.replay_buffer.sample(self.batch_size,
                                                                                             to_tensor=True,
                                                                                             device=self.device)
        reward = torch.sum(w * rewards)
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = reward + (1 - dones) * self.gamma * self.qf1_target(next_states, next_actions)

        current_q_values = self.qf1(states, actions)
        td_errors = target_q_values - current_q_values
        critic_loss = nn.MSELoss()(current_q_values, target_q_values)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update priorities in replay buffer
        priority = td_errors.abs().detach().cpu().numpy()
        priority = (priority + self.replay_buffer.min_priority) ** self.per_alpha
        self.replay_buffer.update_priorities(indices, priority.flatten())

        # Update Actor
        if self.global_step % self.policy_frequency == 0:
            actor_loss = -self.qf1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.prev_actor_loss = actor_loss.item()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            actor_loss = None

        if self.log and self.global_step % 100 == 0:
            wandb.log({
                'losses/q_loss': critic_loss.item(),
                'losses/actor_loss': actor_loss.item() if actor_loss is not None else self.prev_actor_loss,
                'global_step': self.global_step,
            })

    def train(
            self,
            total_timesteps: int,
            eval_envs: List[EVCharging],
            num_eval_episodes_for_front: int = 5,
            eval_freq: int = 100000,
            early_stopping_freq: int = 100000,
            reset_num_timesteps: bool = False,
            save_file_name: str = 'DDPG',
            sub_folder: str = 'scenario_CS05',
    ):
        '''Train the agent.'''
        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        w = [1 / self.reward_dim] * self.reward_dim
        env_iteration = 0
        env = self.envs[0]
        obs, info = env.reset()

        total_training_time = 0
        for _ in range(1, total_timesteps + 1):  # Ensure the loop is defined
            start_time = time.time()
            # If we've run env_iterations episodes on the current environment, switch to the next one
            if env_iteration >= self.env_iterations:
                env = self.rnd.choice(self.envs)
                obs, info = env.reset()
                env_iteration = 0

            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = self.actor.get_action(
                        torch.tensor(obs).float().to(self.device)
                    ).detach().cpu().numpy()[0]

                    noise = self.exploration_noise.sample()
                    action += noise
                    action = action.clip(self.action_space.low, self.action_space.high)

            next_obs, vector_reward, terminated, truncated, info = env.step(action)
            self.replay_buffer.add(obs, action, w, vector_reward, next_obs, terminated)

            if self.global_step >= self.learning_starts:
                self.update()

            if terminated:
                env_iteration += 1  # Increment the episode counter
                obs, info = env.reset()
                self.num_episodes += 1

                if self.log and "episode" in info.keys():
                    log_episode_info(info["episode"], np.dot, w, self.global_step)
            else:
                obs = next_obs

            end_time = time.time()
            total_training_time += (end_time - start_time)

            if eval_envs is not None and self.log:
                if self.global_step % early_stopping_freq == 0:
                    # Evaluation
                    avg_vec_return, avg_disc_vec_return, \
                        avg_constraint_violations, front = single_policy_evaluation(self, eval_envs,
                                                                                    rep=num_eval_episodes_for_front)

                    self.report(
                        0,
                        0,
                        avg_vec_return,
                        avg_disc_vec_return,
                        avg_constraint_violations,
                        reward_utility=True,
                        log_name='eval',
                    )
                    wandb.log({'num_episodes': self.num_episodes, 'training_time': round(total_training_time)})
                    reward = np.sum(w * avg_vec_return)
                    if self.early_stopper(reward, q=self.qf1, actor=self.actor, q_target=self.qf1_target,
                                          actor_target=self.actor_target, q_optimizer=self.q_optimizer,
                                          actor_optimizer=self.actor_optimizer):
                        if self.early_stopper.do_restart():
                            self.learning_rate *= 0.1
                            early_stopping_freq = int(early_stopping_freq / 2)
                            eval_freq = int(eval_freq / 2)
                            self.early_stopper.load_best_model(q=self.qf1, actor=self.actor, q_target=self.qf1_target,
                                                               actor_target=self.actor_target,
                                                               q_optimizer=self.q_optimizer,
                                                               actor_optimizer=self.actor_optimizer)
                        else:
                            print('Early stopping triggered')
                            break  # `break` now correctly inside the loop

                elif self.global_step % eval_freq == 0:
                    avg_vec_return, avg_disc_vec_return, \
                        avg_constraint_violations, front = single_policy_evaluation(self, eval_envs,
                                                                                    rep=num_eval_episodes_for_front)

                    self.report(
                        0,
                        0,
                        avg_vec_return,
                        avg_disc_vec_return,
                        avg_constraint_violations,
                        reward_utility=True,

                        log_name='eval',
                    )
                    wandb.log({'num_episodes': self.num_episodes, 'training_time': round(total_training_time)})
                    self.save(sub_folder=sub_folder, filename=save_file_name, save_replay_buffer=False)

        self.global_step += 1

        if eval_envs is not None and self.log:
            # Evaluation
            avg_vec_return, avg_disc_vec_return, \
                avg_constraint_violations, front = single_policy_evaluation(self, eval_envs,
                                                                            rep=num_eval_episodes_for_front)

            self.report(
                0,
                0,
                avg_vec_return,
                avg_disc_vec_return,
                avg_constraint_violations,
                reward_utility=True,
                log_name='eval',
            )
        self.save(sub_folder=sub_folder, filename=save_file_name, save_replay_buffer=False)
        reward = wandb.run.summary['eval/reward']
        self.close_wandb()
        return reward

    @torch.no_grad()
    def eval(
            self, obs: Union[np.ndarray, torch.Tensor], w: Union[np.ndarray, torch.Tensor], torch_action=False
    ) -> Union[np.ndarray, torch.Tensor]:
        '''Evaluate the policy action for the given observation and weight vector.'''
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs).float().to(self.device)

        action = self.actor.get_action(obs)

        if not torch_action:
            action = action.detach().cpu().numpy()

        return action

    def get_weights(self):
        w = [1 / self.reward_dim] * self.reward_dim
        return [w]

    def save(self, sub_folder: str = None, filename: str = None, save_replay_buffer=False):
        save_dir = f'{self.save_dir}/{sub_folder}'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{filename}.tar')
        save_dict = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer if save_replay_buffer else None
        }
        torch.save(save_dict, save_path)
        print(f'Model saved to {save_path}')

    def load(self, sub_folder: str = None, filename: str = None, load_replay_buffer=False):
        save_dir = f'{self.save_dir}/{sub_folder}'
        load_path = os.path.join(save_dir, f'{filename}.tar')
        if not os.path.exists(load_path):
            raise ValueError(f'No saved model found at {save_dir}/{filename}')
        if torch.cuda.is_available():
            load_dict = torch.load(load_path)
        else:
            load_dict = torch.load(load_path, map_location=torch.device('cpu'))
        self.actor.load_state_dict(load_dict['actor_state_dict'])
        self.actor_target.load_state_dict(load_dict['actor_target_state_dict'])
        self.qf1.load_state_dict(load_dict['qf1_state_dict'])
        self.qf1_target.load_state_dict(load_dict['qf1_target_state_dict'])
        self.actor_optimizer.load_state_dict(load_dict['actor_optimizer'])
        self.q_optimizer.load_state_dict(load_dict['q_optimizer'])
        if load_replay_buffer and 'replay_buffer' in load_dict:
            self.replay_buffer = load_dict['replay_buffer']
        print(f'Model loaded from {load_path}')

    def get_config(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'tau': self.tau,
            'gamma': self.gamma,
            'net_arch': self.net_arch,
            'policy_frequency': self.policy_frequency,
            'buffer_size': self.buffer_size,
            'learning_starts': self.learning_starts,
            'per_alpha': self.per_alpha,
            'env_iterations': self.env_iterations,
            'seed': self.seed,
        }
