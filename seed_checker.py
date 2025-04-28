import random
import numpy as np
from collections import deque

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 40
ROWS, COLS = SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE
exit_x, exit_y = COLS - 2, ROWS - 2

def generate_complex_maze(seed):
    random.seed(seed)
    np.random.seed(seed)
    button_safe_zone = [(r, c) for r in range(ROWS//3, 2*ROWS//3) for c in range(COLS//3, 2*COLS//3)]
    maze = np.zeros((ROWS, COLS))
    for r in range(ROWS):
        for c in range(COLS):
            if random.random() < 0.1 and (r, c) not in [(1,1), (exit_x, exit_y)] and (r, c) not in button_safe_zone:
                maze[r][c] = 1
    return maze

def get_random_object_position(maze):
    button_safe_zone = [(r, c) for r in range(ROWS//3, 2*ROWS//3) for c in range(COLS//3, 2*COLS//3)]
    while True:
        x, y = random.randint(1, COLS-2), random.randint(1, ROWS-2)
        if maze[y][x] == 0 and all(
            0 <= x + dx < COLS and 0 <= y + dy < ROWS and maze[y + dy][x + dx] == 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ) and (y, x) not in button_safe_zone:
            return x, y

def can_agent_reach(agent_pos, target_pos, maze, obj_pos=None):
    # BFS for agent movement. If obj_pos is given, treat it as a wall.
    queue = deque([agent_pos])
    visited = set()
    visited.add(agent_pos)
    while queue:
        x, y = queue.popleft()
        if (x, y) == target_pos:
            return True
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<COLS and 0<=ny<ROWS and maze[ny][nx]==0 and (nx,ny) not in visited:
                if obj_pos is not None and (nx,ny) == obj_pos:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False

def is_pushpull_solvable(maze, agent_start, obj_start):
    # BFS state: (agent_x, agent_y, obj_x, obj_y)
    start = (agent_start[0], agent_start[1], obj_start[0], obj_start[1])
    queue = deque([start])
    visited = set()
    visited.add(start)
    # For each BFS step, try every move (push, pull, move).
    while queue:
        ax, ay, ox, oy = queue.popleft()
        # Check for goal
        if (ox, oy) == (exit_x, exit_y):
            return True
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            # PUSH: agent moves into object, pushes it forward
            if (ax + dx, ay + dy) == (ox, oy):
                n_ox, n_oy = ox + dx, oy + dy
                n_ax, n_ay = ox, oy
                if (0 <= n_ox < COLS and 0 <= n_oy < ROWS and maze[n_oy][n_ox]==0 and maze[ay+dy][ax+dx]==0):
                    new_state = (n_ax, n_ay, n_ox, n_oy)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
            # PULL: agent moves away, object comes to agent's old position
            elif (ax - dx, ay - dy) == (ox, oy):
                n_ax, n_ay = ax + dx, ay + dy
                n_ox, n_oy = ax, ay
                if (0 <= n_ax < COLS and 0 <= n_ay < ROWS and
                    maze[n_ay][n_ax]==0 and
                    0 <= n_ox < COLS and 0 <= n_oy < ROWS and maze[n_oy][n_ox]==0):
                    new_state = (n_ax, n_ay, n_ox, n_oy)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
            # NORMAL: agent moves (no object movement)
            else:
                n_ax, n_ay = ax + dx, ay + dy
                if (0 <= n_ax < COLS and 0 <= n_ay < ROWS and
                    maze[n_ay][n_ax]==0 and (n_ax, n_ay)!=(ox, oy)):
                    new_state = (n_ax, n_ay, ox, oy)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
    return False

def check_seed(seed):
    maze = generate_complex_maze(seed)
    obj_x, obj_y = get_random_object_position(maze)
    agent_start = (1,1)
    obj_start = (obj_x, obj_y)
    # Check agent can reach object
    agent_to_obj = can_agent_reach(agent_start, obj_start, maze)
    # Check agent can solve the puzzle (move/push/pull object to goal)
    if not agent_to_obj:
        return False
    can_solve = is_pushpull_solvable(maze, agent_start, obj_start)
    return can_solve

if __name__ == "__main__":
    good_seeds = []
    for seed in range(1, 201):  # You can adjust the upper bound as needed
        try:
            if check_seed(seed):
                print(f"[SOLVABLE] Seed {seed} is solvable.")
                good_seeds.append(seed)
            else:
                print(f"[UNSOLVABLE] Seed {seed} is NOT solvable.")
        except Exception as e:
            print(f"[ERROR] Seed {seed}: {e}")
    # Save to file
    with open("solvable_seeds.txt", "w") as f:
        f.write(repr(good_seeds))
    print(f"Found {len(good_seeds)} solvable seeds.")
