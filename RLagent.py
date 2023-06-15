import GridRoboticCleanerEnv as customenv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
import env_util
import pickle
import os




class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 8, kernel_size=4, stride=1)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.gelu2 = nn.GELU()
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=2, stride=1)
        self.gelu3 = nn.GELU()

        self.fc1 = nn.Linear(4 * 4 * 16, 64)
        self.gelu4 = nn.GELU()
        
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.gelu1(self.conv1(x))
        x = self.gelu2(self.conv2(x))
        x = self.gelu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.gelu4(self.fc1(x))
        x = self.fc2(x)
        return x
    



class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in batch])
        return states, actions, rewards, next_states, dones
    


class DQNAgent:
    def __init__(self, state_channels, action_dim, lr, gamma, tau, epsilon, epsilon_decay, epsilon_min, delta, delta_decay, delta_min, memory_capacity, batch_size, agentgrid):
        self.state_channels = state_channels
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.delta = delta
        self.delta_decay = delta_decay
        self.delta_min = delta_min

        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: {}".format(self.device))

        self.behavior_net = DQN(state_channels, action_dim).to(self.device)
        self.target_net = DQN(state_channels, action_dim).to(self.device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.behavior_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(memory_capacity)

        self.loss = 0.0
        self.cnt = 0
        self.print_interval = 100

        self.re_search_mode = False
        self.masked_map = np.full_like(agentgrid, -1)


    def select_clean_action(self, agent_location, agentgrid):
        target_position, _ = env_util.bfs(agent_location, 4, agentgrid)
        return env_util.find_first_move(agent_location, target_position, agentgrid)
    

    def select_visit_action(self, agent_location, agentgrid):
        target_position, _ = env_util.bfs(agent_location, -1, agentgrid)
        return env_util.find_first_move(agent_location, target_position, agentgrid)
    
    def select_return_action(self, agent_location, agentgrid, end_pos):
        current_move = env_util.find_first_move(agent_location, end_pos, agentgrid)
        return current_move

    def select_research_action(self, agent_location, sight, agentgrid):
        self.masked_map = np.where(sight == 1, 1, self.masked_map)
        if -1 not in self.masked_map:
            self.re_search_mode = False
            target_position, _ = env_util.bfs(agent_location, 2, agentgrid)
            return env_util.find_first_move(agent_location, target_position, agentgrid)
        
        target_position, _ = env_util.bfs2(agent_location, -1, agentgrid, self.masked_map)
        return env_util.find_first_move(agent_location, target_position, agentgrid)

    def select_action(self, state, agentgrid, agent_location, final_destination, sight):
        self.masked_map = np.where(sight == 1, 1, self.masked_map)

        if np.random.rand() <= self.delta:
            agent_location = tuple(agent_location)
            if 4 in agentgrid:
                return self.select_clean_action(agent_location, agentgrid)
            elif -1 in agentgrid:
                return self.select_visit_action(agent_location, agentgrid)
            elif -1 not in self.masked_map:
                final_destination = tuple(final_destination)
                return self.select_return_action(agent_location, agentgrid, final_destination)
            else:
                return self.select_research_action(agent_location, sight, agentgrid)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.behavior_net(state_tensor)
                return q_values.argmax(dim=1).item()
                
        
    def train(self):
        if len(self.memory.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_batch = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        state_action_values = self.behavior_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        expected_state_action_values = reward_batch + self.gamma * next_state_values * (1 - done_batch)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss = loss.item()

        self.cnt += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update_target(self):
        for target_param, behavior_param in zip(self.target_net.parameters(), self.behavior_net.parameters()):
            target_param.data.copy_(self.tau * behavior_param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def decay_delta(self):
        if self.delta > self.delta_min:
            self.delta *= self.delta_decay

    def print_info(self):
        if self.cnt != 0:   
            print("Loss: {}".format(self.loss / self.cnt), end=" ")
        print("delta: {}".format(self.delta), end=" ")
        print("epsilon: {}".format(self.epsilon))
        self.cnt = 0
        self.loss = 0.0



    def test(self):
        self.target_net.load_state_dict(torch.load(os.path.join("models", "target_net_latest.pth")))



def main():
    env = customenv.GridRoboticCleanerEnv(render_mode="human", 
                                          width=10, 
                                          height=10, 
                                          detection_range=3, 
                                          min_num_rooms=1, 
                                          max_num_rooms=4, 
                                          min_room_size=2, 
                                          max_room_size=4, 
                                          min_door_size=1, 
                                          max_door_size=2, 
                                          max_change=40, 
                                          change_probability=0.04, 
                                          seed=0,
                                          fps=30)
    





    state_channels = env.state_channels
    action_dim = env.action_space.n

    lr = 0.001  # 학습률
    gamma = 0.99  # 할인율
    tau = 0.001  # 타겟 네트워크 업데이트 비율
    epsilon = 1.0  # 탐색적 행동 비율
    epsilon_decay = 0.995  # 탐색적 행동 비율 감소율
    epsilon_min = 0.01  # 탐색적 행동 비율의 최소값

    delta = 0.9  # 탐색적 행동 비율
    delta_decay = 0.995  # 탐색적 행동 비율 감소율
    delta_min = 0.001  # 탐색적 행동 비율의 최소값

    memory_capacity = 250000  # Replay Memory 용량
    batch_size = 4096  # 배치 크기

    agent = DQNAgent(state_channels, action_dim, lr, gamma, tau, epsilon, epsilon_decay, epsilon_min, delta, delta_decay, delta_min, memory_capacity, batch_size, env.agentgrid)

    reward_list = []
    episodes = 1000000  # 에피소드 수
    for episode in range(episodes):
        #print("New Episode {episode}".format(episode=episode + 1))
        observation, _ = env.reset()
        total_reward = 0
        state = env.to_state(observation)

        agent.masked_map = np.full_like(env.agentgrid, -1)

        done = False

        while not done:
            #print("memory capacity: {}".format(len(agent.memory.memory)))


            """if tuple(env._agent_location) == tuple(env._target_location):
                agent.re_search_mode = True"""
            
            if (-1 not in agent.masked_map) and (4 not in observation["memory"]):
                if tuple(env._agent_location) == tuple(env._target_location):
                    agent.masked_map = np.full_like(env.agentgrid, -1)
                


            action = agent.select_action(state, observation["memory"], env._agent_location, env._target_location, observation["sight"])

            observation, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_state = env.to_state(observation)
            agent.memory.push((state, action, reward, next_state, float(done)))
            state = next_state
            #input("Press Enter to continue...")

            agent.train()
            agent.soft_update_target()

        agent.decay_epsilon()
        agent.decay_delta()
        reward_list.append(total_reward)
        with open("reward_list.pkl", "wb") as f:
            pickle.dump(reward_list, f)

        print("Episode: {}, Score: {}".format(episode + 1, reward_list[-1]))
        agent.print_info()

        if episode % 10 == 0:
            torch.save(agent.behavior_net.state_dict(), os.path.join("models", "behavior_net_latest.pth"))
            torch.save(agent.target_net.state_dict(), os.path.join("models", "target_net_latest.pth"))
        
        if episode % 100 == 0:
            torch.save(agent.behavior_net.state_dict(), os.path.join("models", "behavior_net_{}.pth".format(episode)))
            torch.save(agent.target_net.state_dict(), os.path.join("models", "target_net_{}.pth".format(episode)))

    env.close()






if __name__ == "__main__":
    main()




