import GridRoboticCleanerEnv as customenv
import numpy as np
import env_util

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&다양하게 돌릴수 있게 terminal 많이 만들어 뒀습니다
masked_map = None

def select_action(location, map, sight, final_destination):
    global masked_map
    masked_map = np.where(sight == 1, 1, masked_map)
    if 4 in map:
        return select_clean_action(location, map)
    elif -1 in map:
        return select_visit_action(location, map)
    elif -1 not in masked_map:
        return select_return_action(location, map, final_destination)
    else:
        return select_research_action(location, sight, map)
            
    # 단계 1: 방을 돌며 전체 map을 업데이트함.
    # action policy while the room is not fully visited
def select_visit_action(location, map):
    #print("we are going to -1")
    target_position, distance = env_util.bfs(location, -1, map)
    return env_util.find_first_move(location, target_position, map)
    
# 단계 2: 방을 돌며 확인된 현재 map을 구석구석 청소함.
# action policy while the room is not fully cleaned
def select_clean_action(location, map):
    #print("we are going to 4")
    target_position, distance = env_util.bfs(location, 4, map)
    return env_util.find_first_move(location, target_position, map)
    
# 단계 3. target position으로 귀환하는 기능
def select_return_action(location, map, end_pos):
    #print("we are going back")
    current_move = env_util.find_first_move(location, end_pos, map)
    return current_move
    
def select_research_action(location, sight, map):
    global masked_map
    masked_map = np.where(sight == 1, 1, masked_map)


    #print("we are searching again")
    masked_map = np.where(sight == 1, 1, masked_map)
    # print(self.masked_map)
    if -1 not in masked_map:
        #print("in")
        target_position, _ = env_util.bfs(location, 2, map)
        return env_util.find_first_move(location, target_position, map)
    # yes
    # print("we search again again")
    #print("out")
    #print(map)
    #print(masked_map)
    target_position, _ = env_util.bfs3(location, -1, map, masked_map)
    return env_util.find_first_move(location, target_position, map)

def main():
    global masked_map
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
    
    
    total_reward_list = []
    full_trace_list = []
    trace_list = [] # trace를 저장할 변수. Back을 위해 사용
    grid = env.agentgrid # agent를 통해서 알게된 그리드 정보
      
    dy = [1, 0, -1, 0]
    dx = [0, 1, 0, -1]

    is_first_reset = True

    episodes = 120 # 에피소드 수
    for episode in range(episodes):
        observation, _ = env.reset()
        if is_first_reset:
            is_first_reset = False
            input("press enter to start")
        done = False
        agent_done = False
        total_reward = 0
        is_first_phase = True

        #masked_map = np.full_like(env.agentgrid, -1)

        agent_location = env._agent_location

        is_all_open_space = True
        for i in range(0,4):
            new_y = agent_location[0]+dy[i]
            new_x = agent_location[1]+dx[i]
            if new_y >= 0 and new_y < env.height and new_x >= 0 and new_x < env.width:
                if grid[new_y][new_x] != 4:
                    is_all_open_space = False
            else:
                is_all_open_space = False

        if is_all_open_space:
            observation, reward, done, _, _ = env.step(i)
            total_reward += reward
            trace_list.append(i)

        
        while not done:
            agent_location = env._agent_location
            step_succeed = False

            

            if is_first_phase:
                for i in range(0,4):
                    # agent 위치의 바로 아래 칸 부터 갈 수 있는 곳인지 체크
                    new_y = agent_location[0]+dy[i]
                    new_x = agent_location[1]+dx[i]
                    if new_y >= 0 and new_y < env.height and new_x >= 0 and new_x < env.width:
                        pass
                    else:
                        continue
                    if grid[new_y][new_x] == 4: #agent_loaction[0]는 agent의 x좌표
                    # 그 다음 agent 위치의 바로 왼쪽 칸이 갈 수 있는 곳인지 체크
                        new_yy = agent_location[0]+dy[(i-1)%4]
                        new_xx = agent_location[1]+dx[(i-1)%4]
                        coor_outside = False
                        if new_yy >= 0 and new_yy < env.height and new_xx >= 0 and new_xx < env.width:
                            pass
                        else:
                            coor_outside = True
                        if coor_outside or grid[new_yy][new_xx] != 4:
                            #print("success")

                            observation, reward, done, _, _ = env.step(i)
                            total_reward += reward
                            
                            trace_list.append(i)
                            step_succeed = True
                            break

                    
                    # 주위의 모든 칸이 갈 수 없다면 되돌아 가는 back 알고리즘

                if not step_succeed:
                    if len(trace_list) == 0:
                        check_agent_think_it_is_done = True
                        for i in range(0,4):
                            check_y = agent_location[0]+dy[i]
                            check_x = agent_location[1]+dx[i]  

                            if check_y >= 0 and check_y < env.height and check_x >= 0 and check_x < env.width:
                                pass
                            else:
                                continue

                            if grid[check_y][check_x] == 4:
                                check_agent_think_it_is_done = False

                        if check_agent_think_it_is_done:
                            agent_done == True
                            if done == False:
                                is_first_phase = False
                                agent_done = False
                                

                    else:
                        back_new_y = agent_location[0]+dy[(trace_list[-1] + 2) % 4]
                        back_new_x = agent_location[1]+dx[(trace_list[-1] + 2) % 4]

                        if grid[back_new_y][back_new_x] == 5: #back을 실행하는 도중에 장해물 5를 만나는 경우
                            idx = 0

                            # A* 알고리즘 계속 실행
                            while True:
                                target_y = back_new_y
                                target_x = back_new_x

                                idx = 0
                                while True:
                                    target_y += dy[(trace_list[-2 - idx] + 2) % 4]
                                    target_x += dx[(trace_list[-2 - idx] + 2) % 4]
                                    idx += 1
                                    if grid[target_y][target_x] != 5:
                                        break
                                
                                stopover = [target_y, target_x]
                                
                                action_number = env_util.find_first_move(tuple(agent_location), tuple(stopover), grid)
                                observation, reward, done, _, _ = env.step(action_number)
                                total_reward += reward
                                

                                if agent_location[0] == stopover[0] and agent_location[1] == stopover[1]:
                                    break
                            trace_list = trace_list[:-idx - 1] 


                        else:    
                            observation, reward, done, _, _ = env.step((trace_list[-1] + 2) % 4)
                            total_reward += reward
                            

                            trace_list = trace_list[:-1]  # back을 한번 실행하고 가장 마지막 element를 제거

            else: # 2페이즈 시작

                
                masked_map = np.full_like(env.agentgrid, -1)
                target_tuple = tuple(env._target_location)

                
                while not done:    

                    

                    # {"sight": robot_sight_map, "memory": self.agentgrid} 
                    memory_map = observation["memory"]
                    # 보고있다 1 안보인다 0
                    sight = observation["sight"]
                    # print(memory_map.shape) -> (20, 20)
                    agent_location = env._agent_location
                    # print(agent_location.shape) -> (2, )
                    agent_tuple = tuple(agent_location)

                    if (-1 not in masked_map) and (4 not in memory_map):
                        if agent_tuple == target_tuple:
                            masked_map = np.full_like(env.agentgrid, -1)
                                        
                    action = select_action(agent_tuple, memory_map, sight, target_tuple)

                    observation, reward, done, _, _ = env.step(action)
                    total_reward += reward

                

        total_reward_list.append(total_reward)
        print("return_list: ", total_reward_list) 
            
            


if __name__ == "__main__":
    main()