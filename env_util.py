import numpy as np
import heapq
from collections import deque

movements = [(1, 0), (0, 1), (-1, 0), (0, -1)]
movements_extended = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# Define the movement costs
movement_costs = {
    (1, 0): 1,  # Down
    (0, 1): 1,  # Right
    (-1, 0): 1,  # Up
    (0, -1): 1,  # Left
}

movement_directions = {
    (1, 0): 0,  # Down
    (0, 1): 1,  # Right
    (-1, 0): 2,  # Up
    (0, -1): 3,  # Left
}

def count_integers(grid):
    count_0 = np.count_nonzero(grid == 0)
    count_5 = np.count_nonzero(grid == 5)
    return count_0, count_5

# Function to check if a position is valid within the grid
def is_valid_position(position, grid):
    row, col = position
    rows, cols = grid.shape
    not_five = False # Not an obstacle
    not_zero = False # Not a wall 
    if 0 <= row < rows and 0 <= col < cols:
        not_five = grid[row, col] != 5 # Not an obstacle
        not_zero = grid[row, col] != 0 # Not a wall 
    not_obstacle = not_five and not_zero
    return 0 <= row < rows and 0 <= col < cols and not_obstacle


# Function to check if a position is valid within the grid
def is_valid_position2(position, grid):
    row, col = position
    rows, cols = grid.shape
    return 0 <= row < rows and 0 <= col < cols

def is_valid_move(grid, visited, row, col):
    rows, cols = grid.shape
    pos = (row, col)
    if_valid = is_valid_position(pos, grid)
    if if_valid:
        return not visited[row, col]
    else: 
        return False

# Function to calculate the Euclidean distance heuristic
def euclidean_distance(position, end):
    if end is None:
        return 0
    else:
        return np.linalg.norm(np.array(position) - np.array(end))

# Function to find the path using A* algorithm
def find_full_path(start, end, grid):
    rows, cols = grid.shape
    open_list = []
    closed_set = set()
    parent = {}

    g_score = {start: 0}
    f_score = {start: euclidean_distance(start, end)}

    heapq.heappush(open_list, (f_score[start], start))

    while open_list:


        current = heapq.heappop(open_list)[1]

        if current == end:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)

        for dx, dy in movement_costs:
            neighbor = current[0] + dx, current[1] + dy
            if is_valid_position(neighbor, grid):
                tentative_g_score = g_score[current] + movement_costs[(dx, dy)]
                if neighbor in closed_set and tentative_g_score >= g_score.get(neighbor, 0):
                    continue
                if tentative_g_score < g_score.get(neighbor, 0) or neighbor not in [i[1] for i in open_list]:
                    parent[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + euclidean_distance(neighbor, end)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None
    
    # returns list of tuples (y, x)
def return_four_integers(list):
    list_int = []
    for i in (0, len(list)-2):
        init_point = np.array(list[i])
        next_point = np.array(list[i+1])
        movement = next_point - init_point
        four_out = movement_directions[tuple(movement)]
        list_int.append(four_out)

    return list_int

# returns first move for shrtest path with 0~3 integer output
def find_first_move(start, end, grid):
    full_path = find_full_path(start, end, grid)
    # print(full_path)
    #print(start, end, full_path)
    if full_path is None:
        return np.random.randint(0, 4)
    if len(full_path) >= 2:
        init_point = np.array(full_path[0])
        next_point = np.array(full_path[1])
        movement = next_point - init_point
        four_out = movement_directions[tuple(movement)]
    else:
        four_out = np.random.randint(0, 4)

    return four_out

    # returns an integer from zero to three

# Function to perform breadth-first search
def bfs(start, target_int, grid):
    
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        position, distance = queue.popleft()
        row, col = position

        if grid[row, col] == target_int:
            return position, distance

        for dx, dy in movements:
            new_position = row + dx, col + dy

            if is_valid_position(new_position, grid) and new_position not in visited:
                queue.append((new_position, distance + 1))
                visited.add(new_position)

    """print("No target found.")
    print("start: ", start)
    print("target_int: ", target_int)
    print("grid: ", grid)"""
    # No target found
    return None, -1

def bfs2(start, target_int, grid, target_grid):

    # 길은 grid로 찾고 목표는 target_grid로 찾는다.
    
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        position, distance = queue.popleft()
        row, col = position

        
        

        for dx, dy in movements_extended:
            new_position = row + dx, col + dy

            if is_valid_position2(new_position, target_grid):
                if target_grid[row + dx, col + dy] == target_int:
                    x_prime, y_prime = new_position
                    x_original, y_original = position
                    if not is_valid_position(new_position, grid):
                        return position, distance
                    else:
                        dist_kind = abs(x_prime - x_original) + abs(y_prime - y_original)
                        if dist_kind == 1:
                            return new_position, distance + 1
                        elif dist_kind == 2:
                            if is_valid_position((x_prime, y_original), grid) or is_valid_position((x_original, y_prime), grid):
                                return new_position, distance + 2
                            else:
                                return position, distance
                        else:
                            assert False, "dist_kind should be 1 or 2"


            if (dx, dy) in movements:
                if is_valid_position(new_position, grid) and new_position not in visited:
                    queue.append((new_position, distance + 1))
                    visited.add(new_position)

    # No target found
    return None, -1


def bfs3(start, target_int, grid, target_grid):

    # 길은 grid로 찾고 목표는 target_grid로 찾는다.
    
    queue = deque([(start, 0)])
    visited = set([start])

    while queue:
        position, distance = queue.popleft()
        row, col = position

        
        

        for dx, dy in movements_extended:
            new_position = row + dx, col + dy

            if is_valid_position2(new_position, target_grid):
                if target_grid[row + dx, col + dy] == target_int:
                    x_prime, y_prime = new_position
                    x_original, y_original = position
                    if not is_valid_position(new_position, grid):
                        return position, distance
                    else:
                        dist_kind = abs(x_prime - x_original) + abs(y_prime - y_original)
                        if dist_kind == 1:
                            return new_position, distance + 1
                        elif dist_kind == 2:
                            if is_valid_position((x_prime, y_original), grid) or is_valid_position((x_original, y_prime), grid):
                                return new_position, distance + 2
                            else:
                                return position, distance
                        else:
                            assert False, "dist_kind should be 1 or 2"


            if (dx, dy) in movements:
                if is_valid_position(new_position, grid) and new_position not in visited:
                    queue.append((new_position, distance + 1))
                    visited.add(new_position)

    # No target found
    return None, -1
    
def dfs_visit(grid, visited, path, row, col, full_len):
    visited[row, col] = True
    path.append((row, col))

    # rows, cols = grid.shape

    if len(path) == full_len:
        return True

    moves = movements

    for dr, dc in moves:
        new_row, new_col = row + dr, col + dc

        if is_valid_move(grid, visited, new_row, new_col):
            if dfs_visit(grid, visited, path, new_row, new_col, full_len):
                return True

    visited[row, col] = False
    path.pop()

    return False

# Function to perform depth-first search
def visit_all_cells(start, end, grid):
    rows, cols = grid.shape
    zeros, fives = count_integers(grid)

    full_length_maybe = rows * cols - zeros - fives

    visited = np.zeros((rows, cols), dtype=bool)
    path = []

    start_row, start_col = start
    end_row, end_col = end



    dfs_visit(grid, visited, path, start_row, start_col, full_length_maybe)

    path.append((end_row, end_col))

    return path


def apply_values(input_array, mask_array):
    output_array = np.where(mask_array == 1, 1, input_array)
    return output_array


def main():
    # Example usage

    # Example usage
    input_array = np.array([[-1, -1, -1],
                        [-1, -1, -1]])

    mask_array = np.array([[1, 0, 1],
                        [0, 1, 0]])

    output_array = apply_values(input_array, mask_array)

    print("Output Array:")
    print(output_array)

    start = (15, 17) # tuple
    end = (13, 17) # tuple
    print(type(start))
    print(type(end))
    grid = np.array([[3, 3, 3, 5, 3, 3, 5, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 3, 3, 5, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3],
                    [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0],
                    [3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
                    [3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3],
                    [3, 3, 3, 3, 0, 3, 5, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 3],
                    [3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 8, 8, 8, 3],
                    [0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 3, 3, 0, 0, 5, 3, 0],
                    [3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                    [3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
    

    dddd = bfs2(start, 8, grid, grid)
    print(dddd)
    exit(54)

    path = find_full_path(start, end, grid)

    if path:
        print("Path type: ", type(path))
        print("position type: ", type(path[0]))
        print("Path found:")
        for position in path:
            print(position)
        print("first move: ")
        print()
        print()
        print()
        print()
        print()
        move = find_first_move(start, end, grid)
        print(type(move))
        print(move)
    else:
        print("No path found.")
    '''
    target_position, distance = bfs(start, 4, grid)

    if target_position is not None:
        print("Nearest target found at position:", target_position)
        print("Distance:", distance)
    else:
        print("No target found.")
    '''


    

if __name__ == "__main__":
    main()