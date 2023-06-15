import GridRoboticCleanerEnv as customenv
import numpy as np
import random
import env_util



class SlamAgent:
    def __init__(self, state_channels, action_dim, gridworld):
        self.state_channels = state_channels
        self.action_dim = action_dim

        # -1 로 채워진
        self.masked_map = np.full_like(gridworld, -1)
        
        self.cnt = 0
        self.print_interval = 100

    # issue: 중구난방으로 움직이는 문제가 있음. 방향의 우선성을 정해야 함. (아래 오른쪽 위 왼쪽 순)
    def select_action(self, state, location, map, sight, final_destination, env):
        self.masked_map = np.where(sight == 1, 1, self.masked_map)
        #print(self.masked_map)
        #print(map)
        if 4 in map:
            return self.select_clean_action(location, map)
        elif -1 in map:
            return self.select_visit_action(location, map)
        elif -1 not in self.masked_map:
            return self.select_return_action(location, map, final_destination)
        else:
            return self.select_research_action(location, sight, map, env)
            
    # 단계 1: 방을 돌며 전체 map을 업데이트함.
    # action policy while the room is not fully visited
    def select_visit_action(self, location, map):
        #print("we are going to -1")
        target_position, distance = env_util.bfs(location, -1, map)
        return env_util.find_first_move(location, target_position, map)
    
    # 단계 2: 방을 돌며 확인된 현재 map을 구석구석 청소함.
    # action policy while the room is not fully cleaned
    def select_clean_action(self, location, map):
        #print("we are going to 4")
        target_position, distance = env_util.bfs(location, 4, map)
        return env_util.find_first_move(location, target_position, map)
    
    # 단계 3. target position으로 귀환하는 기능
    def select_return_action(self, location, map, end_pos):
        #print("we are going back")
        current_move = env_util.find_first_move(location, end_pos, map)
        return current_move
    
    def select_research_action(self, location, sight, map, env):
        #print("we are searching again")
        # print(self.masked_map)

        self.masked_map = np.where(sight == 1, 1, self.masked_map)
        if -1 not in self.masked_map:
            #print("in")
            target_position, _ = env_util.bfs(location, 2, map)
            return env_util.find_first_move(location, target_position, map)

        
        
        # 여기서 masked map을 사용했기 때문에, 이미 있는 벽을 무시하고 최단 거리의 블록을 고르게 됨
        target_position, _ = env_util.bfs2(location, -1, map, self.masked_map)

        


        return env_util.find_first_move(location, target_position, map)
        
        # bfs가 none type을 반환함, target이 없을때. 



# RLAgent에서 가져온 main 함수. 
def main():
    env = customenv.GridRoboticCleanerEnv(render_mode="human", 
                                          width=10, 
                                          height=10, 
                                          detection_range=2, 
                                          min_num_rooms=1, 
                                          max_num_rooms=4, 
                                          min_room_size=2, 
                                          max_room_size=4, 
                                          min_door_size=1, 
                                          max_door_size=2, 
                                          max_change=100, 
                                          change_probability=0.1, 
                                          seed=0,
                                          fps=16)
    agent_location = env._agent_location
    state_channels = env.state_channels
    action_dim = env.action_space.n

    is_first_reset = True

    agent = SlamAgent(state_channels, action_dim, env.agentgrid)

    episodes = 120  # 에피소드 수
    reward_list = []
    for episode in range(episodes):
        print("New Episode {episode}".format(episode=episode + 1))
        observation, _ = env.reset()
        if is_first_reset:
            is_first_reset = False
            input("Press Enter to continue...")
        total_reward = 0
        state = env.to_state(observation)
        target_tuple = tuple(env._target_location)

        # -1 로 채워진
        agent.masked_map = np.full_like(env.agentgrid, -1)
        
        
        done = False

        while not done:
            
            # {"sight": robot_sight_map, "memory": self.agentgrid} 
            memory_map = observation["memory"]
            # 보고있다 1 안보인다 0
            sight = observation["sight"]
            # print(memory_map.shape) -> (20, 20)
            agent_location = env._agent_location
            # print(agent_location.shape) -> (2, )
            agent_tuple = tuple(agent_location)

            if (-1 not in agent.masked_map) and (4 not in memory_map):
                if agent_tuple == target_tuple:
                    #print("Checkpoint!")
                    agent.masked_map = np.full_like(env.agentgrid, -1)
            
            
            
            action = agent.select_action(state, agent_tuple, memory_map, sight, target_tuple, env)
            # print(action)

            observation, reward, done, _, _ = env.step(action)
            total_reward += reward

            next_state = env.to_state(observation)
            
            state = next_state

            

        
        reward_list.append(total_reward)

        if episode % 10 == 0:
            print("Episode: {}, Score: {}".format(episode + 1, reward_list[-1]))

        print("return_list: ", reward_list) 

    env.close()
    



if __name__ == "__main__":
    main()