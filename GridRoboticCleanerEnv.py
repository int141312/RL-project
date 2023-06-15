import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import numpy as np
import pygame
import copy

import gymnasium as gym
from gymnasium import spaces

# 안녕하세요!
class GridRoboticCleanerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=20, height=20, detection_range=4, min_num_rooms=2, max_num_rooms=5, min_room_size=3, max_room_size=6, min_door_size=2, max_door_size=4, max_change=20, change_probability=0.01, seed=0, fps=10):
        self.width = width  # The width of the square grid
        self.height = height  # The height of the square grid
        self.window_width = None  # The width of the pygame window
        self.window_height = None  # The height of the pygame window
        self.grid_size = None  # The size of each grid cell in pixels

        self._agent_location = None # current loc. of agent  ex) self._agent_location = np.array([x, y]) -> grid[y][x]
        self._target_location = None

        self.detection_range = detection_range

        self.state_channels = 2

        self.gridworld = np.zeros((self.height, self.width), dtype=int) # 벽: 0, agent 위치: 1, target 위치: 2, 청소한 곳: 3, 청소하지 않은 곳: 4, 장애물: 5
        self.room_mask = np.zeros((self.height, self.width), dtype=int) # 벽: False, 방: True
        self.interruption_map = np.zeros((self.height, self.width), dtype=int) # 장애물: True, 나머지: False
        self.agentgrid = np.zeros((self.height, self.width), dtype=int) # unknown: -1, 벽: 0, agent 위치: 1, target 위치: 2, 청소한 곳: 3, 청소하지 않은 곳: 4, 장애물: 5
        self.agentgrid.fill(-1)

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.fps = fps

        self.next_seed = seed
        self.max_change = max_change
        self.curr_change = 0
        self.change_probability = change_probability


        self.min_num_rooms = min_num_rooms
        self.max_num_rooms = max_num_rooms

        self.min_room_size = min_room_size
        self.max_room_size = max_room_size

        self.min_door_size = min_door_size
        self.max_door_size = max_door_size

        self.curr_step_cnt = 0
        self.step_limit = 10000


    def path_exists(self, start_x, start_y, end_x, end_y, grid):
        start = [start_x, start_y]
        end = [end_x, end_y]
        queue = [start]
        visited = []
        while queue:
            

            node = queue.pop(0)
            if node == end:
                return True
            if node not in visited:
                visited.append(node)

                curr_x = node[0]
                curr_y = node[1]
                

                if curr_x - 1 >= 0:
                    if grid[curr_x - 1, curr_y] == 0:
                        queue.append([curr_x - 1, curr_y])
                if curr_x + 1 < 3:
                    if grid[curr_x + 1, curr_y] == 0:
                        queue.append([curr_x + 1, curr_y])
                if curr_y - 1 >= 0:
                    if grid[curr_x, curr_y - 1] == 0:
                        queue.append([curr_x, curr_y - 1])
                if curr_y + 1 < 3:
                    if grid[curr_x, curr_y + 1] == 0:
                        queue.append([curr_x, curr_y + 1])
        
        
        return False


    def valid_obstacle_check(self, x, y):
        temp_grid_for_obstacle_check = np.zeros((3, 3), dtype=int)
        temp_grid_for_obstacle_check[1, 1] = 1
        if x == 0:
            temp_grid_for_obstacle_check[0, 0] = -1
            temp_grid_for_obstacle_check[0, 1] = -1
            temp_grid_for_obstacle_check[0, 2] = -1
        elif x == self.height - 1:
            temp_grid_for_obstacle_check[2, 0] = -1
            temp_grid_for_obstacle_check[2, 1] = -1
            temp_grid_for_obstacle_check[2, 2] = -1
        if y == 0:
            temp_grid_for_obstacle_check[0, 0] = -1
            temp_grid_for_obstacle_check[1, 0] = -1
            temp_grid_for_obstacle_check[2, 0] = -1
        elif y == self.width - 1:
            temp_grid_for_obstacle_check[0, 2] = -1
            temp_grid_for_obstacle_check[1, 2] = -1
            temp_grid_for_obstacle_check[2, 2] = -1 # -1은 안쓴다는거

        if x - 1 >= 0:
            if y - 1 >= 0:
                if self.gridworld[x - 1, y - 1] == 5 or self.gridworld[x - 1, y - 1] == 0:
                    temp_grid_for_obstacle_check[0, 0] = 1
            if self.gridworld[x - 1, y] == 5 or self.gridworld[x - 1, y] == 0:
                temp_grid_for_obstacle_check[0, 1] = 1
            if y + 1 < self.width:
                if self.gridworld[x - 1, y + 1] == 5 or self.gridworld[x - 1, y + 1] == 0:
                    temp_grid_for_obstacle_check[0, 2] = 1
        
        if y - 1 >= 0:
            if self.gridworld[x, y - 1] == 5 or self.gridworld[x, y - 1] == 0:
                temp_grid_for_obstacle_check[1, 0] = 1
        if y + 1 < self.width:
            if self.gridworld[x, y + 1] == 5 or self.gridworld[x, y + 1] == 0:
                temp_grid_for_obstacle_check[1, 2] = 1

        if x + 1 < self.height:
            if y - 1 >= 0:
                if self.gridworld[x + 1, y - 1] == 5 or self.gridworld[x + 1, y - 1] == 0:
                    temp_grid_for_obstacle_check[2, 0] = 1
            if self.gridworld[x + 1, y] == 5 or self.gridworld[x + 1, y] == 0:
                temp_grid_for_obstacle_check[2, 1] = 1
            if y + 1 < self.width:
                if self.gridworld[x + 1, y + 1] == 5 or self.gridworld[x + 1, y + 1] == 0:
                    temp_grid_for_obstacle_check[2, 2] = 1

        if temp_grid_for_obstacle_check[0, 0] == 1 and temp_grid_for_obstacle_check[0, 1] == 1 and temp_grid_for_obstacle_check[1, 0] == 1 and temp_grid_for_obstacle_check[1, 1] == 1:
            return False
        if temp_grid_for_obstacle_check[0, 1] == 1 and temp_grid_for_obstacle_check[0, 2] == 1 and temp_grid_for_obstacle_check[1, 1] == 1 and temp_grid_for_obstacle_check[1, 2] == 1:
            return False
        if temp_grid_for_obstacle_check[1, 0] == 1 and temp_grid_for_obstacle_check[1, 1] == 1 and temp_grid_for_obstacle_check[2, 0] == 1 and temp_grid_for_obstacle_check[2, 1] == 1:
            return False
        if temp_grid_for_obstacle_check[1, 1] == 1 and temp_grid_for_obstacle_check[1, 2] == 1 and temp_grid_for_obstacle_check[2, 1] == 1 and temp_grid_for_obstacle_check[2, 2] == 1:
            return False
        


        

        empty_space = list()
        for i in range(3):
            for j in range(3):
                if temp_grid_for_obstacle_check[i, j] == 0:
                    empty_space.append((i, j))

        
        
        connected = True
        for i in range(len(empty_space)):
            for j in range(i + 1, len(empty_space)):
               
               connected = connected and self.path_exists(empty_space[i][0], empty_space[i][1], empty_space[j][0], empty_space[j][1], temp_grid_for_obstacle_check)



        

        return connected


    def visibility_check_order_list(self, x, y, grid, height, width):
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]

        unit_grid_len = 2
        result = list()
        total_num = grid.reshape((-1, )).shape[0] - 1
        if total_num < 0:
            total_num = 0
        while len(result) != total_num:
            x = x - 1
            y = y - 1
            for i in range(4):
                for _ in range(unit_grid_len):
                    x = x + dx[i]
                    y = y + dy[i]
                    if x >= 0 and x < height and y >= 0 and y < width:
                        result.append([x, y])
            unit_grid_len += 2
        return result


    def obstacle_between(self, x1, y1, x2, y2, grid):
        skip_x = x2
        skip_y = y2

        if x1 == x2:
            if y1 > y2:
                temp = y1
                y1 = y2
                y2 = temp
            for i in range(y1 + 1, y2):
                if grid[x1, i] == 5 or grid[x1, i] == 0:
                    return True
        elif y1 == y2:
            if x1 > x2:
                temp = x1
                x1 = x2
                x2 = temp
            for i in range(x1 + 1, x2):
                if grid[i, y1] == 5 or grid[i, y1] == 0:
                    return True
        else:
            if x1 > x2:
                temp = x1
                x1 = x2
                x2 = temp
                temp = y1
                y1 = y2
                y2 = temp
            if grid[x1, y1] == 5 or grid[x1, y1] == 0:
                if x1 != skip_x or y1 != skip_y:
                    return True
            if grid[x2, y2] == 5 or grid[x2, y2] == 0:
                if x2 != skip_x or y2 != skip_y:
                    return True
            if y1 < y2:
                slope = (y2 - y1) / (x2 - x1)
                bias = y1 + 0.5 - slope * (x1 + 0.5)
                for i in range(x1 + 1, x2):
                    curr_value = int(slope * i + bias)
                    if grid[i, curr_value] == 5 or grid[i, curr_value] == 0:
                        if i != skip_x or curr_value != skip_y:
                            return True
                slope = (x2 - x1) / (y2 - y1)
                bias = x1 + 0.5 - slope * (y1 + 0.5)
                for i in range(y1 + 1, y2):
                    curr_value = int(slope * i + bias)
                    if grid[curr_value, i] == 5 or grid[curr_value, i] == 0:
                        if curr_value != skip_x or i != skip_y:
                            return True
            else:
                slope = (y2 - y1) / (x2 - x1)
                bias = y1 + 0.5 - slope * (x1 + 0.5)
                for i in range(x1 + 1, x2 + 1):
                    curr_value = int(slope * i + bias)
                    if abs(slope * i + bias - curr_value) < 1e-10:
                        curr_value -= 1
                    if grid[i, curr_value] == 5 or grid[i, curr_value] == 0:
                        if i != skip_x or curr_value != skip_y:
                            return True
                slope = (x2 - x1) / (y2 - y1)
                bias = x1 + 0.5 - slope * (y1 + 0.5)
                for i in range(y1, y2 + 1, -1):
                    curr_value = int(slope * i + bias)
                    if grid[curr_value, i - 1] == 5 or grid[curr_value, i - 1] == 0:
                        if curr_value != skip_x or i - 1 != skip_y:
                            return True

        return False


    def _get_obs(self):
        top_cutoff = False
        bottom_cutoff = False
        left_cutoff = False
        right_cutoff = False

        x_upper = self._agent_location[0] - self.detection_range
        x_lower = self._agent_location[0] + self.detection_range + 1
        y_left = self._agent_location[1] - self.detection_range
        y_right = self._agent_location[1] + self.detection_range + 1
        
        if self._agent_location[0] - self.detection_range < 0:
            x_upper = 0
            top_cutoff = True
        if self._agent_location[0] + self.detection_range >= self.height:
            x_lower = self.height
            bottom_cutoff = True
        if self._agent_location[1] - self.detection_range < 0:
            y_left = 0
            left_cutoff = True
        if self._agent_location[1] + self.detection_range >= self.width:
            y_right = self.width
            right_cutoff = True


        obs = self.gridworld[x_upper:x_lower, y_left:y_right].copy()

        

        agent_relative_x = self.detection_range
        if top_cutoff:
            agent_relative_x = self._agent_location[0]
        agent_relative_y = self.detection_range
        if left_cutoff:
            agent_relative_y = self._agent_location[1]

        check_list = self.visibility_check_order_list(agent_relative_x, agent_relative_y, obs, obs.shape[0], obs.shape[1])


        # agent_relative_x, agent_relative_y, obs, obs.shape[0], obs.shape[1]

        robot_sight = np.zeros_like(obs)
        robot_sight.fill(-1)

        for curr_node in range(len(check_list)):
            x = check_list[curr_node][0]
            y = check_list[curr_node][1]
            if robot_sight[x, y] != -1:
                continue
            if self.obstacle_between(agent_relative_x, agent_relative_y, x, y, obs):
                robot_sight[x, y] = 0
                x_diff = x - agent_relative_x
                y_diff = y - agent_relative_y
                while True:
                    x = x + x_diff
                    y = y + y_diff
                    if x < 0 or x >= obs.shape[0] or y < 0 or y >= obs.shape[1]:
                        break
                    robot_sight[x, y] = 0
            elif obs[x, y] == 5 or obs[x, y] == 0:
                robot_sight[x, y] = 1
                x_diff = x - agent_relative_x
                y_diff = y - agent_relative_y
                while True:
                    x = x + x_diff
                    y = y + y_diff
                    if x < 0 or x >= obs.shape[0] or y < 0 or y >= obs.shape[1]:
                        break
                    robot_sight[x, y] = 0
            else:
                robot_sight[x, y] = 1

        
        robot_sight[agent_relative_x, agent_relative_y] = 1

        robot_sight_map = np.zeros_like(self.gridworld)
        robot_sight_map[x_upper:x_lower, y_left:y_right] = robot_sight
        
        self.agentgrid[robot_sight_map == 1] = self.gridworld[robot_sight_map == 1].copy()

        
        


        return {
            "sight": robot_sight_map,
            "memory": self.agentgrid
        } 
    




    def _get_info(self):
        return {
            "grid": self.gridworld
        }



    def draw_room(self, room_x_up, room_x_down, room_y_left, room_y_right, width, height):
        self.gridworld[room_x_up:room_x_down + 1, room_y_left:room_y_right + 1] = 3

        # change outside border of room to 0
        if room_x_up - 1 >= 0:
            y_left = room_y_left - 1
            y_right = room_y_right + 2
            if y_left < 0:
                y_left = 0
            if y_right > width:
                y_right = width
            wall_grid = self.gridworld[room_x_up - 1, y_left:y_right]
            wall_grid[wall_grid == 4] = 0
        if room_x_down + 1 < height:
            y_left = room_y_left - 1
            y_right = room_y_right + 2
            if y_left < 0:
                y_left = 0
            if y_right > width:
                y_right = width
            wall_grid = self.gridworld[room_x_down + 1, y_left:y_right]
            wall_grid[wall_grid == 4] = 0
        if room_y_left - 1 >= 0:
            x_up = room_x_up - 1
            x_down = room_x_down + 2
            if x_up < 0:
                x_up = 0
            if x_down > height:
                x_down = height
            wall_grid = self.gridworld[x_up:x_down, room_y_left - 1]
            wall_grid[wall_grid == 4] = 0
        if room_y_right + 1 < width:
            x_up = room_x_up - 1
            x_down = room_x_down + 2
            if x_up < 0:
                x_up = 0
            if x_down > height:
                x_down = height
            wall_grid = self.gridworld[x_up:x_down, room_y_right + 1]
            wall_grid[wall_grid == 4] = 0


    def generated_map(self, width, height):
        self.gridworld = np.zeros((height, width), dtype=int)
        self.gridworld.fill(4)

        num_rooms = self.np_random.integers(
            self.min_num_rooms, self.max_num_rooms, size=1, dtype=int
        )

        possible_room_locations = [(0, 0, 0), (0, width - 1, 1), (height - 1, width - 1, 2), (height - 1, 0, 3)]  # (x, y, direction)
    

        for i in range(num_rooms.item()):
            possible_room_locations_num = len(possible_room_locations)
            possible_room_locations_permutation = self.np_random.permutation(possible_room_locations_num)

            room_not_decided = True
            room_grid = None
            room_x_up = None
            room_x_down = None
            room_y_right = None
            room_y_left = None
            room_width = None
            room_height = None
            room_direction = None

            try_limit = 50
            cnt = 0
            end_bool = False

            while room_not_decided:
                cnt += 1
                room_width = self.np_random.integers(
                    self.min_room_size, self.max_room_size, size=1, dtype=int
                )
                room_height = self.np_random.integers(
                    self.min_room_size, self.max_room_size, size=1, dtype=int
                )
                for location_index in possible_room_locations_permutation:
                    room_x_up = possible_room_locations[location_index][0]
                    room_y_left = possible_room_locations[location_index][1]
                    room_direction = possible_room_locations[location_index][2]

                    room_x_prev = room_x_up
                    room_y_prev = room_y_left

                    if room_direction == 0:
                        room_x_down = room_x_up + room_height.item() - 1
                        room_y_right = room_y_left + room_width.item() - 1
                    elif room_direction == 1:
                        room_x_down = room_x_up + room_height.item() - 1
                        room_y_right = room_y_left
                        room_y_left = room_y_right - room_width.item() + 1
                    elif room_direction == 2:
                        room_x_down = room_x_up
                        room_y_right = room_y_left
                        room_x_up = room_x_down - room_height.item() + 1
                        room_y_left = room_y_right - room_width.item() + 1
                    elif room_direction == 3:
                        room_x_down = room_x_up
                        room_y_right = room_y_left + room_width.item() - 1
                        room_x_up = room_x_down - room_height.item() + 1


                    room_grid = self.gridworld[room_x_up:room_x_down + 1, room_y_left:room_y_right + 1]
                    if np.all(room_grid == 4):
                        possible_room_locations.remove((room_x_prev, room_y_prev, room_direction))
                        room_not_decided = False


                        rand_num = self.np_random.integers(
                            1, 4, size=1, dtype=int
                        )





                        if rand_num.item() == 1:
                            rand_num = self.np_random.integers(
                                0, 1, size=1, dtype=int
                            )
                            if rand_num.item() == 1:
                                if room_direction == 0 or room_direction == 1:
                                    y_left = room_y_left
                                    y_right = room_y_right + 1
                                    if y_left < 0:
                                        y_left = 0
                                    if y_right > width:
                                        y_right = width
                                    while room_x_down < height - 1 and (self.gridworld[room_x_down + 1, y_left:y_right] == 4).all():
                                        room_x_down += 1
                                else:
                                    y_left = room_y_left
                                    y_right = room_y_right + 1
                                    if y_left < 0:
                                        y_left = 0
                                    if y_right > width:
                                        y_right = width
                                    while room_x_up > 0 and (self.gridworld[room_x_up - 1, y_left:y_right] == 4).all():
                                        room_x_up -= 1
                            else:
                                if room_direction == 0 or room_direction == 3:
                                    x_up = room_x_up
                                    x_down = room_x_down + 1
                                    if x_up < 0:
                                        x_up = 0
                                    if x_down > height:
                                        x_down = height
                                    while room_y_right < width - 1 and (self.gridworld[x_up:x_down, room_y_right + 1] == 4).all():
                                        room_y_right += 1
                                else:
                                    x_up = room_x_up
                                    x_down = room_x_down + 1
                                    if x_up < 0:
                                        x_up = 0
                                    if x_down > height:
                                        x_down = height
                                    while room_y_left > 0 and (self.gridworld[x_up:x_down, room_y_left - 1] == 4).all():
                                        room_y_left -= 1
     
                            room_grid = self.gridworld[room_x_up:room_x_down + 1, room_y_left:room_y_right + 1]

                        
                        for possible_location in possible_room_locations:
                            if possible_location[0] >= room_x_up and possible_location[0] <= room_x_down and possible_location[1] >= room_y_left and possible_location[1] <= room_y_right:
                                possible_room_locations.remove(possible_location)




                        if room_direction == 0:
                            if room_y_right + 2 < width and self.gridworld[room_x_up, room_y_right + 2] == 4:
                                if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_right + 2] == 4:
                                    temp_y = room_y_right + 2
                                    while True:
                                        if temp_y >= 0:
                                            if self.gridworld[room_x_up - 2, temp_y] == 4:
                                                temp_y -= 1
                                            else:
                                                possible_room_locations.append((room_x_up - 2, temp_y + 1, 3))
                                                break
                                        else:
                                            possible_room_locations.append((room_x_up - 2, temp_y + 1, 3))
                                            break
                                else:
                                    possible_room_locations.append((room_x_up, room_y_right + 2, 0))
                            else:
                                if room_y_right + 2 >= width:
                                    if room_x_down + 2 < height and self.gridworld[room_x_down + 2, width - 1] == 4:
                                        possible_room_locations.append((room_x_down + 2, width - 1, 1))
                                
                                else:
                                    if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_right] == 4:
                                        if room_y_right + 1 < width and self.gridworld[room_x_down + 2, room_y_right + 1] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_down + 2, room_y_right, 1))






                        elif room_direction == 1:
                            if room_y_left - 2 >= 0 and self.gridworld[room_x_up, room_y_left - 2] == 4:
                                if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_left - 2] == 4:
                                    temp_y = room_y_left - 2
                                    while True:
                                        if temp_y < width:
                                            if self.gridworld[room_x_up - 2, temp_y] == 4:
                                                temp_y += 1
                                            else:
                                                possible_room_locations.append((room_x_up - 2, temp_y - 1, 2))
                                                break
                                        else:
                                            possible_room_locations.append((room_x_up - 2, temp_y - 1, 2))
                                            break
                                else:
                                    possible_room_locations.append((room_x_up, room_y_left - 2, 1))
                            else:
                                if room_y_left - 2 < 0:
                                    if room_x_down + 2 < height and self.gridworld[room_x_down + 2, 0] == 4:
                                        possible_room_locations.append((room_x_down + 2, 0, 0))
                                
                                else:
                                    if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_left] == 4:
                                        if room_y_left - 1 >= 0 and self.gridworld[room_x_down + 2, room_y_left - 1] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_down + 2, room_y_left, 0))


                        elif room_direction == 2:
                            if room_y_left - 2 >= 0 and self.gridworld[room_x_down, room_y_left - 2] == 4:
                                if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_left - 2] == 4:
                                    temp_y = room_y_left - 2
                                    while True:
                                        if temp_y < width:
                                            if self.gridworld[room_x_down + 2, temp_y] == 4:
                                                temp_y += 1
                                            else:
                                                possible_room_locations.append((room_x_down + 2, temp_y - 1, 1))
                                                break
                                        else:
                                            possible_room_locations.append((room_x_down + 2, temp_y - 1, 1))
                                            break
                                else:
                                    possible_room_locations.append((room_x_down, room_y_left - 2, 2))

                            else:
                                if room_y_left - 2 < 0:
                                    if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, 0] == 4:
                                        possible_room_locations.append((room_x_up - 2, 0, 3))
                                
                                else:
                                    if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_left] == 4:
                                        if room_y_left - 1 >= 0 and self.gridworld[room_x_up - 2, room_y_left - 1] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_up - 2, room_y_left, 3))


                        elif room_direction == 3:
                            if room_y_right + 2 < width and self.gridworld[room_x_down, room_y_right + 2] == 4:
                                if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_right + 2] == 4:
                                    temp_y = room_y_right + 2
                                    while True:
                                        if temp_y >= 0:
                                            if self.gridworld[room_x_down + 2, temp_y] == 4:
                                                temp_y -= 1
                                            else:
                                                possible_room_locations.append((room_x_down + 2, temp_y + 1, 0))
                                                break
                                        else:
                                            possible_room_locations.append((room_x_down + 2, temp_y + 1, 0))
                                            break
                                else:
                                    possible_room_locations.append((room_x_down, room_y_right + 2, 3))
                            else:
                                if room_y_right + 2 >= width:
                                    if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, width - 1] == 4:
                                        possible_room_locations.append((room_x_up - 2, width - 1, 2))
                                
                                else:
                                    if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_right] == 4:
                                        if room_y_right + 1 < width and self.gridworld[room_x_up - 2, room_y_right + 1] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_up - 2, room_y_right, 2))

                        if room_direction == 0:
                            if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_left] == 4:
                                if room_y_left - 2 >= 0 and self.gridworld[room_x_down + 2, room_y_left - 2] == 4:
                                    temp_x = room_x_down + 2
                                    while True:
                                        if temp_x >= 0:
                                            if self.gridworld[temp_x, room_y_left - 2] == 4:
                                                temp_x -= 1
                                            else:
                                                possible_room_locations.append((temp_x + 1, room_y_left - 2, 1))
                                                break
                                        else:
                                            possible_room_locations.append((temp_x + 1, room_y_left - 2, 1))
                                            break
                                else:
                                    possible_room_locations.append((room_x_down + 2, room_y_left, 0))
                            else:
                                if room_x_down + 2 >= height:
                                    if room_y_right + 2 < width and self.gridworld[height - 1, room_y_right + 2] == 4:
                                        possible_room_locations.append((height - 1, room_y_right + 2, 3))
                                
                                else:
                                    if room_y_right + 2 < width and self.gridworld[room_x_down, room_y_right + 2] == 4:
                                        if room_x_down + 1 < height and self.gridworld[room_x_down + 1, room_y_right + 2] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_down, room_y_right + 2, 3))


                        elif room_direction == 1:
                            if room_x_down + 2 < height and self.gridworld[room_x_down + 2, room_y_right] == 4:
                                if room_y_right + 2 < width and self.gridworld[room_x_down + 2, room_y_right + 2] == 4:
                                    temp_x = room_x_down + 2
                                    while True:
                                        if temp_x >= 0:
                                            if self.gridworld[temp_x, room_y_right + 2] == 4:
                                                temp_x -= 1
                                            else:
                                                possible_room_locations.append((temp_x + 1, room_y_right + 2, 0))
                                                break
                                        else:
                                            possible_room_locations.append((temp_x + 1, room_y_right + 2, 0))
                                            break
                                else:
                                    possible_room_locations.append((room_x_down + 2, room_y_right, 1))

                            else:
                                if room_x_down + 2 >= height:
                                    if room_y_left - 2 >= 0 and self.gridworld[height - 1, room_y_left - 2] == 4:
                                        possible_room_locations.append((height - 1, room_y_left - 2, 2))
                                
                                else:
                                    if room_y_left - 2 >= 0 and self.gridworld[room_x_down, room_y_left - 2] == 4:
                                        if room_x_down + 1 < height and self.gridworld[room_x_down + 1, room_y_left - 2] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_down, room_y_left - 2, 2))


                        elif room_direction == 2:
                            if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_right] == 4:
                                if room_y_right + 2 < width and self.gridworld[room_x_up - 2, room_y_right + 2] == 4:
                                    temp_x = room_x_up - 2
                                    while True:
                                        if temp_x < height:
                                            if self.gridworld[temp_x, room_y_right + 2] == 4:
                                                temp_x += 1
                                            else:
                                                possible_room_locations.append((temp_x - 1, room_y_right + 2, 3))
                                                break
                                        else:
                                            possible_room_locations.append((temp_x - 1, room_y_right + 2, 3))
                                            break
                                else:
                                    possible_room_locations.append((room_x_up - 2, room_y_right, 2))
                            else:
                                if room_x_up - 2 < 0:
                                    if room_y_left - 2 >= 0 and self.gridworld[0, room_y_left - 2] == 4:
                                        possible_room_locations.append((0, room_y_left - 2, 1))
                                
                                else:
                                    if room_y_left - 2 >= 0 and self.gridworld[room_x_up, room_y_left - 2] == 4:
                                        if room_x_up - 1 >= 0 and self.gridworld[room_x_up - 1, room_y_left - 2] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_up, room_y_left - 2, 1))


                        elif room_direction == 3:
                            if room_x_up - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_left] == 4:
                                if room_y_left - 2 >= 0 and self.gridworld[room_x_up - 2, room_y_left - 2] == 4:
                                    temp_x = room_x_up - 2
                                    while True:
                                        if temp_x < height:
                                            if self.gridworld[temp_x, room_y_left - 2] == 4:
                                                temp_x += 1
                                            else:
                                                possible_room_locations.append((temp_x - 1, room_y_left - 2, 2))
                                                break
                                        else:
                                            possible_room_locations.append((temp_x - 1, room_y_left - 2, 2))
                                            break
                                else:
                                    possible_room_locations.append((room_x_up - 2, room_y_left, 3))
                            else:
                                if room_x_up - 2 < 0:
                                    if room_y_right + 2 < width and self.gridworld[0, room_y_right + 2] == 4:
                                        possible_room_locations.append((0, room_y_right + 2, 0))
                                
                                else:
                                    if room_y_right + 2 < width and self.gridworld[room_x_up, room_y_right + 2] == 4:
                                        if room_x_up - 1 >= 0 and self.gridworld[room_x_up - 1, room_y_right + 2] == 4:
                                            pass
                                        else:
                                            possible_room_locations.append((room_x_up, room_y_right + 2, 0))


                        possible_locations = []
                        for location in possible_room_locations:
                            if (location[0], location[1]) not in possible_locations:
                                possible_locations.append((location[0], location[1]))
                            else:
                                possible_room_locations.remove(location)

                        break
                if cnt >= try_limit:
                    end_bool = True
                    break

            if end_bool:
                self.gridworld[self.gridworld == 9] = 4
                self.gridworld[self.gridworld == 3] = 4
                #print("fail to make room")
                return
            


            assert room_grid is not None
            assert room_x_up is not None
            assert room_x_down is not None
            assert room_y_right is not None
            assert room_y_left is not None
            assert room_width is not None
            assert room_height is not None
            assert room_direction is not None

            gridworld_backup = self.gridworld.copy()
            

            # make door
            self.draw_room(room_x_up, room_x_down, room_y_left, room_y_right, width, height)
            
            door_error = True

            door_num = self.np_random.integers(
                1, 2, size=1, dtype=int
            )
            for _ in range(door_num.item()):
                door_position_np = self.np_random.permutation(4)
                for door_side_position in door_position_np:
                    door_size = self.np_random.integers(
                        self.min_door_size, self.max_door_size, size=1, dtype=int
                    )
                    door_size = door_size.item()
                    if door_side_position == 0:
                        if room_x_up - 1 >= 0 and (self.gridworld[room_x_up - 1, room_y_left:room_y_right + 1] == 0).all():
                            door_size = min(door_size, room_y_right - room_y_left + 1)
                            if room_y_left == (room_y_right - door_size + 1):
                                door_position = room_y_left
                            else:
                                door_position = self.np_random.integers(
                                    room_y_left, room_y_right - door_size + 1, size=1, dtype=int
                                )
                                door_position = door_position.item()
                            connection_bool = False
                            if room_x_up - 2 >= 0:
                                connection_bool = (self.gridworld[room_x_up - 2, door_position:door_position + door_size] == 3).any() or (self.gridworld[room_x_up - 2, door_position:door_position + door_size] == 4).any()
                            if connection_bool:
                                self.gridworld[room_x_up - 1, door_position:door_position + door_size] = 9
                                door_error = False
                                break
                            else:
                                if room_x_up - 3 >= 0:
                                    connection_bool = (self.gridworld[room_x_up - 3, door_position:door_position + door_size] == 3).any() or (self.gridworld[room_x_up - 3, door_position:door_position + door_size] == 4).any()
                                if connection_bool:
                                    self.gridworld[room_x_up - 1, door_position:door_position + door_size] = 9
                                    self.gridworld[room_x_up - 2, door_position:door_position + door_size] = 9
                                    door_error = False
                                    break
                    elif door_side_position == 1:
                        if room_x_down + 1 < height and (self.gridworld[room_x_down + 1, room_y_left:room_y_right + 1] == 0).all():
                            door_size = min(door_size, room_y_right - room_y_left + 1)
                            if room_y_left == (room_y_right - door_size + 1):
                                door_position = room_y_left
                            else:
                                door_position = self.np_random.integers(
                                    room_y_left, room_y_right - door_size + 1, size=1, dtype=int
                                )
                                door_position = door_position.item()
                            connection_bool = False
                            if room_x_down + 2 < height:
                                connection_bool = (self.gridworld[room_x_down + 2, door_position:door_position + door_size] == 3).any() or (self.gridworld[room_x_down + 2, door_position:door_position + door_size] == 4).any()
                            if connection_bool:
                                self.gridworld[room_x_down + 1, door_position:door_position + door_size] = 9
                                door_error = False
                                break
                            else:
                                if room_x_down + 3 < height:
                                    connection_bool = (self.gridworld[room_x_down + 3, door_position:door_position + door_size] == 3).any() or (self.gridworld[room_x_down + 3, door_position:door_position + door_size] == 4).any()
                                if connection_bool:
                                    self.gridworld[room_x_down + 1, door_position:door_position + door_size] = 9
                                    self.gridworld[room_x_down + 2, door_position:door_position + door_size] = 9
                                    door_error = False
                                    break
                    elif door_side_position == 2:
                        if room_y_left - 1 >= 0 and (self.gridworld[room_x_up:room_x_down + 1, room_y_left - 1] == 0).all():
                            door_size = min(door_size, room_x_down - room_x_up + 1)
                            if room_x_up == (room_x_down - door_size + 1):
                                door_position = room_x_up
                            else:
                                door_position = self.np_random.integers(
                                    room_x_up, room_x_down - door_size + 1, size=1, dtype=int
                                )
                                door_position = door_position.item()
                            connection_bool = False
                            if room_y_left - 2 >= 0:
                                connection_bool = (self.gridworld[door_position:door_position + door_size, room_y_left - 2] == 3).any() or (self.gridworld[door_position:door_position + door_size, room_y_left - 2] == 4).any()
                            if connection_bool:
                                self.gridworld[door_position:door_position + door_size, room_y_left - 1] = 9
                                door_error = False
                                break
                            else:
                                if room_y_left - 3 >= 0:
                                    connection_bool = (self.gridworld[door_position:door_position + door_size, room_y_left - 3] == 3).any() or (self.gridworld[door_position:door_position + door_size, room_y_left - 3] == 4).any()
                                if connection_bool:
                                    self.gridworld[door_position:door_position + door_size, room_y_left - 1] = 9
                                    self.gridworld[door_position:door_position + door_size, room_y_left - 2] = 9
                                    door_error = False
                                    break
                    elif door_side_position == 3:
                        if room_y_right + 1 < width and (self.gridworld[room_x_up:room_x_down + 1, room_y_right + 1] == 0).all():
                            door_size = min(door_size, room_x_down - room_x_up + 1)
                            if room_x_up == (room_x_down - door_size + 1):
                                door_position = room_x_up
                            else:
                                door_position = self.np_random.integers(
                                    room_x_up, room_x_down - door_size + 1, size=1, dtype=int
                                )
                                door_position = door_position.item()
                            connection_bool = False
                            if room_y_right + 2 < width:
                                connection_bool = (self.gridworld[door_position:door_position + door_size, room_y_right + 2] == 3).any() or (self.gridworld[door_position:door_position + door_size, room_y_right + 2] == 4).any()
                            if connection_bool:
                                self.gridworld[door_position:door_position + door_size, room_y_right + 1] = 9
                                door_error = False
                                break
                            else:
                                if room_y_right + 3 < width:
                                    connection_bool = (self.gridworld[door_position:door_position + door_size, room_y_right + 3] == 3).any() or (self.gridworld[door_position:door_position + door_size, room_y_right + 3] == 4).any()
                                if connection_bool:
                                    self.gridworld[door_position:door_position + door_size, room_y_right + 1] = 9
                                    self.gridworld[door_position:door_position + door_size, room_y_right + 2] = 9
                                    door_error = False
                                    break
                        

            if door_error:
                self.gridworld = gridworld_backup.copy()

                self.gridworld[self.gridworld == 9] = 4
                self.gridworld[self.gridworld == 3] = 4

                print("Door error")

                return

        self.door_mask = self.gridworld == 9
        self.gridworld[self.door_mask] = 4
        self.gridworld[self.gridworld == 3] = 4



                
            
    def check_door_direction_up(self, curr_x, curr_y):
        if curr_x - 1 >= 0 and self.gridworld[curr_x - 1][curr_y] == 0:
            self.gridworld[curr_x - 1][curr_y] = 4
            curr_x -= 1
            return curr_x, curr_y, True
        else:
            return curr_x, curr_y, False
        

    def check_door_direction_down(self, curr_x, curr_y):
        if curr_x + 1 < self.height and self.gridworld[curr_x + 1][curr_y] == 0:
            self.gridworld[curr_x + 1][curr_y] = 4
            curr_x += 1
            return curr_x, curr_y, True
        else:
            return curr_x, curr_y, False
        
    def check_door_direction_left(self, curr_x, curr_y):
        if curr_y - 1 >= 0 and self.gridworld[curr_x][curr_y - 1] == 0:
            self.gridworld[curr_x][curr_y - 1] = 4
            curr_y -= 1
            return curr_x, curr_y, True
        else:
            return curr_x, curr_y, False
        
    def check_door_direction_right(self, curr_x, curr_y):
        if curr_y + 1 < self.width and self.gridworld[curr_x][curr_y + 1] == 0:
            self.gridworld[curr_x][curr_y + 1] = 4
            curr_y += 1
            return curr_x, curr_y, True
        else:
            return curr_x, curr_y, False


    def grid_all_observable_check(self):   # 이거 해야함!!!!!
        check_observability = np.zeros_like(self.gridworld)
        for i in range(self.height):
            for j in range(self.width):
                if self.gridworld[i][j] == 4:
                    if i - 1 >= 0:
                        check_observability[i - 1][j] = 1
                    if i + 1 < self.height:
                        check_observability[i + 1][j] = 1
                    if j - 1 >= 0:
                        check_observability[i][j - 1] = 1
                    if j + 1 < self.width:
                        check_observability[i][j + 1] = 1
        if (check_observability == 0).any():
            return False
        else:
            return True
                    


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=self.next_seed)
        self.next_seed += 1

        self.curr_change = 0
        self.curr_step_cnt = 0

        while True:
            while True:
                gen_succeed = False
                while not gen_succeed:
                    try:
                        while True:
                            self.generated_map(self.width, self.height)
                            if (self.gridworld == 4).any():
                                gen_succeed = True
                                break
                    except:
                        gen_succeed = False


            

               
            
                while True:
                    all_connected_x = 0
                    all_connected_y = 0
                    while True:
                        if self.gridworld[all_connected_x][all_connected_y] == 4:
                            break
                        else:
                            all_connected_y += 1
                            if all_connected_y == self.height:
                                all_connected_x += 1
                                all_connected_y = 0
                                if all_connected_x == self.width:
                                    print("something wrong 1")
                                    exit(16446856458654)
                    
                    all_connected_check_queue = list()
                    all_connected_check_queue.append([all_connected_x, all_connected_y])
                    all_connected_check_list = list()
                    all_connected_check_list.append([all_connected_x, all_connected_y])
                    while len(all_connected_check_queue) != 0:
                        all_connected_check_x = all_connected_check_queue[0][0]
                        all_connected_check_y = all_connected_check_queue[0][1]
                        all_connected_check_queue.pop(0)
                        if all_connected_check_x - 1 >= 0 and self.gridworld[all_connected_check_x - 1][all_connected_check_y] == 4 and [all_connected_check_x - 1, all_connected_check_y] not in all_connected_check_list:
                            all_connected_check_queue.append([all_connected_check_x - 1, all_connected_check_y])
                            all_connected_check_list.append([all_connected_check_x - 1, all_connected_check_y])
                        if all_connected_check_x + 1 < self.height and self.gridworld[all_connected_check_x + 1][all_connected_check_y] == 4 and [all_connected_check_x + 1, all_connected_check_y] not in all_connected_check_list:
                            all_connected_check_queue.append([all_connected_check_x + 1, all_connected_check_y])
                            all_connected_check_list.append([all_connected_check_x + 1, all_connected_check_y])
                        if all_connected_check_y - 1 >= 0 and self.gridworld[all_connected_check_x][all_connected_check_y - 1] == 4 and [all_connected_check_x, all_connected_check_y - 1] not in all_connected_check_list:
                            all_connected_check_queue.append([all_connected_check_x, all_connected_check_y - 1])
                            all_connected_check_list.append([all_connected_check_x, all_connected_check_y - 1])
                        if all_connected_check_y + 1 < self.width and self.gridworld[all_connected_check_x][all_connected_check_y + 1] == 4 and [all_connected_check_x, all_connected_check_y + 1] not in all_connected_check_list:
                            all_connected_check_queue.append([all_connected_check_x, all_connected_check_y + 1])
                            all_connected_check_list.append([all_connected_check_x, all_connected_check_y + 1])
                    
                    temp_gridworld = np.zeros_like(self.gridworld)

                    for coordinate in all_connected_check_list:
                        temp_gridworld[coordinate[0]][coordinate[1]] = 1
                        

                    temp_gridworld_backup = temp_gridworld.copy()

                    temp_gridworld[self.gridworld == 0] = 1
                

                    if (temp_gridworld == 1).all():
                        break
                    else:
                        


                        coor_adjacent_to_disconnected = list()
                        for curr_x in range(self.height):
                            for curr_y in range(self.width):
                                if temp_gridworld[curr_x][curr_y] == 1:
                                    adjacent_to_disconnected = False
                                    if curr_x - 1 >= 0 and temp_gridworld[curr_x - 1][curr_y] == 0:
                                        adjacent_to_disconnected = True
                                    if curr_x + 1 < self.height and temp_gridworld[curr_x + 1][curr_y] == 0:
                                        adjacent_to_disconnected = True
                                    if curr_y - 1 >= 0 and temp_gridworld[curr_x][curr_y - 1] == 0:
                                        adjacent_to_disconnected = True
                                    if curr_y + 1 < self.width and temp_gridworld[curr_x][curr_y + 1] == 0:
                                        adjacent_to_disconnected = True
                                    if adjacent_to_disconnected:
                                        coor_adjacent_to_disconnected.append([curr_x, curr_y])


                        if len(coor_adjacent_to_disconnected) == 0:
                            print("something wrong 2")
                            exit(164468564145658654)
                        elif len(coor_adjacent_to_disconnected) == 1:
                            disconnected_index = 0
                        else:
                            disconnected_index = self.np_random.integers(
                                0, len(coor_adjacent_to_disconnected) - 1, size=1, dtype=int
                            )
                            disconnected_index = disconnected_index.item()
                            
                        curr_x = coor_adjacent_to_disconnected[disconnected_index][0]
                        curr_y = coor_adjacent_to_disconnected[disconnected_index][1]


                        self.gridworld[curr_x][curr_y] = 4
                        door_size = self.np_random.integers(
                            self.min_door_size, self.max_door_size, size=1, dtype=int
                        )
                        door_size = door_size.item()
                        for _ in range(1, door_size):
                            door_direction_permutation = self.np_random.permutation(4)
                            door_direction_function = [
                                self.check_door_direction_up,
                                self.check_door_direction_down,
                                self.check_door_direction_left,
                                self.check_door_direction_right,
                            ]
                            
                            changed = False
                            for door_direction_index in door_direction_permutation:
                                curr_x, curr_y, changed = door_direction_function[door_direction_index](curr_x, curr_y)
                                if changed:
                                    break
                            if not changed:
                                break


                

                self.room_mask = self.gridworld == 4

            # 모든 공간이 observable 한지 확인 - 이거 해야함
                if self.grid_all_observable_check():
                    break


            # 여기까지 방 생성 끝남


            # 여기에 장애물 넣어야 함

            obstacle_gen_num = 10
            for _ in range(obstacle_gen_num):
                obstacle_probability = self.np_random.uniform(0, 1, size=1).item()
                obstacle_x = None
                obstacle_y = None
                if obstacle_probability < 0.3:
                    if (self.gridworld != 4).all():
                        continue
                    while True:
                        obstacle_x = self.np_random.integers(0, self.height - 1, size=1, dtype=int)
                        obstacle_y = self.np_random.integers(0, self.width - 1, size=1, dtype=int)
                        if self.gridworld[obstacle_x.item()][obstacle_y.item()] == 4:
                            break

                    if self.valid_obstacle_check(obstacle_x.item(), obstacle_y.item()):
                        self.gridworld[obstacle_x.item()][obstacle_y.item()] = 5
                        self.interruption_map[obstacle_x.item()][obstacle_y.item()] = 1
                        #print("obstacle generated", obstacle_x.item(), obstacle_y.item())
                

            if (self.gridworld != 4).all() == False:
                break

        # 여기서부터 에이전트와 타겟의 위치를 정함  
        agent_x = None
        agent_y = None

        while True:
            agent_x = self.np_random.integers(0, self.width - 1, size=1, dtype=int) # 방 벽면에서만 시작을 할지는 고민을 해봐야 함
            agent_y = self.np_random.integers(0, self.height - 1, size=1, dtype=int)
            if self.gridworld[agent_x.item()][agent_y.item()] == 4:
                break
        
        self._agent_location = np.concatenate((agent_x, agent_y))
        self._target_location = self._agent_location.copy() # 타겟의 위치를 에이전트의 위치로 설정
        self.gridworld[self._target_location[0]][self._target_location[1]] = 2 # 타겟의 위치를 2으로 설정
        
        



        observation = self._get_obs()  # 이거 바꿔야함 - 이게 에이전트 시야 구현해야 하는 부분
        info = self._get_info() # 이게 전체 맵의 정보를 담아야 하는 부분

        if self.render_mode == "human":
            self._render_frame() # 이거는 나중에 해
        
        print_map = False
        if self.render_mode == "human":
            print_map = False
        if print_map:
            temp_list = list()
            for i in range(self.width):
                t_list = list()
                for j in range(self.height):
                    if self.gridworld[i][j] == 0:
                        t_list.append("■")
                    elif self.gridworld[i][j] == 4:
                        t_list.append("□")
                    elif self.gridworld[i][j] == 5:
                        t_list.append("★")
                temp_list.append(t_list)

            for tt in temp_list:
                print(tt)

        return observation, info


    def to_state(self, observation):
        memory_to_state = observation["memory"][np.newaxis, :]
        sight_to_state = observation["sight"][np.newaxis, :]
        state = np.concatenate((memory_to_state, sight_to_state), axis=0)

        return state


    def is_terminal(self):  
        first_check = self._agent_location[0] == self._target_location[0] and self._agent_location[1] == self._target_location[1]
        second_check = (self.gridworld != 4).all()
        return first_check and second_check


    def step(self, action):

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action] # action은 0, 1, 2, 3
        # 순서대로 아래, 오른쪽, 위, 왼쪽


        new_location = self._agent_location + direction
        if new_location[0] >= 0 and new_location[0] < self.height and new_location[1] >= 0 and new_location[1] < self.width:
            new_grid_value = self.gridworld[new_location[0]][new_location[1]]
            if new_grid_value == 0 or new_grid_value == 5: # 0은 벽, 5 장애물
                pass
            else:
                self._agent_location += direction

        new_cleaned = False
        if self.gridworld[self._agent_location[0]][self._agent_location[1]] != 2:
            if self.gridworld[self._agent_location[0]][self._agent_location[1]] == 4:
                new_cleaned = True
            self.gridworld[self._agent_location[0]][self._agent_location[1]] = 3 # 에이전트가 있던 위치를 3으로 바꿈, 청소 완료
            

        # We use `np.clip` to make sure we don't leave the grid


        
        # An episode is done iff the agent has reached the target
        terminated = self.is_terminal()  

        reward = 1 if new_cleaned else -1
        reward = 100 if terminated else reward  # constant negative reward


        # 환경 변화 시키는 코드 여기에 넣어야 함
        # 최대 변경 횟수를 제한 두어야 episode를 보장할 수 있음
        if self.curr_change < self.max_change:
            curr_probability = self.np_random.uniform(0, 1, size=1).item()
            if curr_probability < self.change_probability:
                obstacle_gen_num = 1
                while True:
                    curr_probability = self.np_random.uniform(0, 1, size=1).item()
                    if curr_probability < 0.5:
                        obstacle_gen_num += 1
                        if obstacle_gen_num >= 10:
                            break
                    else:
                        break

                for _ in range(obstacle_gen_num):
                    if (self.gridworld != 4).all():
                        continue
                    obstacle_x = None
                    obstacle_y = None
                    while True:
                        obstacle_x = self.np_random.integers(0, self.height - 1, size=1, dtype=int)
                        obstacle_y = self.np_random.integers(0, self.width - 1, size=1, dtype=int)
                        if self.gridworld[obstacle_x.item()][obstacle_y.item()] == 4 or self.gridworld[obstacle_x.item()][obstacle_y.item()] == 3:
                            if self._agent_location[0] == obstacle_x.item() and self._agent_location[1] == obstacle_y.item():
                                continue
                            break
                    if self.valid_obstacle_check(obstacle_x.item(), obstacle_y.item()):
                        self.gridworld[obstacle_x.item()][obstacle_y.item()] = 5
                        self.interruption_map[obstacle_x.item()][obstacle_y.item()] = 1
                        self.curr_change += 1


        if self.curr_change < self.max_change:
            curr_probability = self.np_random.uniform(0, 1, size=1).item()
            if curr_probability < self.change_probability:
                obstacle_num = sum(self.interruption_map.flatten())
                if obstacle_num > 0:
                    obstacle_delete_num = 1
                    while True:
                        curr_probability = self.np_random.uniform(0, 1, size=1).item()
                        if curr_probability < 0.5:
                            obstacle_delete_num += 1
                            if obstacle_delete_num >= obstacle_num:
                                break
                        else:
                            break

                    obstacle_index_permutation = self.np_random.permutation(obstacle_num)
                    obstacle_index_permutation = obstacle_index_permutation[:obstacle_delete_num]
                    obstacle_index_permutation = np.sort(obstacle_index_permutation)

                    curr_index = -1
                    for i in range(self.height):
                        for j in range(self.width):
                            if self.interruption_map[i][j] == 1:
                                curr_index += 1
                                if curr_index in obstacle_index_permutation:
                                    self.gridworld[i][j] = 4
                                    self.interruption_map[i][j] = 0
                                    self.curr_change += 1


        

        self.gridworld[self._target_location[0]][self._target_location[1]] = 2




        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        

        print_map = False
        if self.render_mode == "human":
            print_map = False
        if print_map:
            temp_list = list()
            for i in range(self.width):
                t_list = list()
                for j in range(self.height):
                    if self.gridworld[i][j] == 0:
                        t_list.append("■")
                    elif self.gridworld[i][j] == 4:
                        t_list.append("□")
                    elif self.gridworld[i][j] == 5:
                        t_list.append("★")
                temp_list.append(t_list)

            for tt in temp_list:
                print(tt)
            print()
            print()
            print()


        self.curr_step_cnt += 1
        if self.curr_step_cnt >= self.step_limit:
            terminated = True
            reward = -1000
            print("step limit")

        #input("Press Enter to continue...")
        return observation, reward, terminated, False, info  # 4번째는 done
    

    def _render_frame(self):
        margin_pixels = 3
        max_window_width = 1280
        max_window_height = 720

        if self.grid_size is None:

            grid_size_candidate1 = (max_window_width - margin_pixels * (self.width - 1)) // self.width
            grid_size_candidate2 = (max_window_height - margin_pixels * (self.height - 1)) // self.height

            self.grid_size = min(grid_size_candidate1, grid_size_candidate2)

        if self.window_width is None:
            self.window_width = self.grid_size * self.width + margin_pixels * (self.width - 1)

        if self.window_height is None:
            self.window_height = self.grid_size * self.height + margin_pixels * (self.height - 1)

        if self.window is None and self.render_mode == "human":
            pygame.init()
            #pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        pix_square_size = self.grid_size + margin_pixels

        color_list = [
            (120, 120, 120),
            (255, 0, 0),
            (0, 170, 255),
            (64, 255, 0),
            (255, 160, 0),
            (0, 0, 0),
            (255, 255, 0)
        ]
        for y_cor in range(self.height):
            for x_cor in range(self.width):
                curr_color = color_list[self.gridworld[y_cor][x_cor]]
                if self._agent_location[1] == x_cor and self._agent_location[0] == y_cor:
                    curr_color = color_list[1]

                #print(curr_color)
                pygame.draw.rect(
                    canvas,
                    curr_color,
                    pygame.Rect(pix_square_size * x_cor, pix_square_size * y_cor, self.grid_size, self.grid_size)
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.fps) # 3 frames per second
            #input("Press Enter to continue...")

        


def close(self):
    if self.window is not None:
        pygame.display.quit()
        pygame.quit()

import random
import time

if __name__ == "__main__":
    env = GridRoboticCleanerEnv(render_mode="human", width=20, height=20, detection_range=4, min_num_rooms=3, max_num_rooms=6, min_room_size=4, max_room_size=10, min_door_size=2, max_door_size=4, max_change=40, change_probability=0.01, seed=0)
    for _ in range(1):
        env.reset()
        for _ in range(5):
            # step 에서는 유효한, 즉 벽이나 장애물, 맵 밖으로 나가지 않는 action을 줘야함
            # step은 그런거 예외처리 안하고 그냥 이동함
            # 각자 env.action_space.sample_1~3()을 만들어서 필요한대로 하면 될듯

            print(random.randint(0, 3))
            


            # print(env.np_random.integers(0, 100, size=1, dtype=int))  방 생성 전용. 이 함수는 다른 곳에서 쓰면 안됨
            print(random.randint(0, 3)) # 랜덤 함수를 쓸거면 이걸 쓰시오
            print(random.randint(0, 3))
            time.sleep(1)
            print(np.random.rand())  # 랜덤 함수를 쓸거면 이걸 쓰시오

            env.step(0)
    env.close()




