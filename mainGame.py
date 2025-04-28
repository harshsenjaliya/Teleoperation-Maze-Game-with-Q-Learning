import pygame
import random
import numpy as np
import time
import sys
import pickle
import os
from collections import defaultdict, deque

# Game config
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 40
ROWS, COLS = SCREEN_HEIGHT // GRID_SIZE, SCREEN_WIDTH // GRID_SIZE

WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
BLUE   = (0, 0, 255)
RED    = (255, 0, 0)
GREEN  = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY   = (100, 100, 100)
ORANGE = (255, 127, 0)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Teleoperation Maze Game")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 28)

STATE_WELCOME = "welcome"
STATE_MODE_SELECT = "mode_select"
STATE_MANUAL_INPUT = "manual_input"
STATE_MANUAL = "manual"
STATE_MANUAL_COMPLETE = "manual_complete"
STATE_RL_SETUP = "rl_setup"
STATE_RL_TRAIN = "rl_train"
STATE_RL_TEST = "rl_test"
STATE_RL_COMPLETE = "rl_complete"
STATE_ERROR = "error"
current_state = STATE_WELCOME
state_start_time = time.time()

button_manual = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 40, 200, 50)
button_rl_train = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 40, 200, 50)
button_rl_test = pygame.Rect(SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 110, 200, 50)
button_back = pygame.Rect(SCREEN_WIDTH - 120, SCREEN_HEIGHT - 50, 100, 40)

Q_TABLE_FILE = "q_table.pkl"

# === Load solvable seeds ===
with open("solvable_seeds.txt", "r") as f:
    solvable_seeds = eval(f.read())
assert len(solvable_seeds) > 0, "No solvable seeds loaded!"

split_idx = int(0.8 * len(solvable_seeds))
TRAIN_SEEDS = solvable_seeds[:split_idx]
TEST_SEEDS  = solvable_seeds[split_idx:]
MAX_EPISODES = 500
MAX_STEPS = 1000
RL_STEP_DELAY = 0.001

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

def get_object_position(maze):
    button_safe_zone = [(r, c) for r in range(ROWS//3, 2*ROWS//3) for c in range(COLS//3, 2*COLS//3)]
    while True:
        x, y = random.randint(1, COLS-2), random.randint(1, ROWS-2)
        if maze[y][x] == 0 and all(
            0 <= x + dx < COLS and 0 <= y + dy < ROWS and maze[y + dy][x + dx] == 0
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ) and (y, x) not in button_safe_zone:
            return x, y

class Player:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.color = BLUE
    def move(self, dx, dy, maze, obj):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0:
            # Push
            if (new_x, new_y) == (obj.x, obj.y):
                if obj.move(dx, dy, maze):
                    self.x, self.y = new_x, new_y
            # Pull
            elif (self.x - dx, self.y - dy) == (obj.x, obj.y):
                old_x, old_y = self.x, self.y
                if (0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0
                        and 0 <= old_x < COLS and 0 <= old_y < ROWS and maze[old_y][old_x] == 0):
                    self.x, self.y = new_x, new_y
                    obj.x, obj.y = old_x, old_y
            # Normal move
            else:
                self.x, self.y = new_x, new_y

class Object:
    def __init__(self, maze, x=None, y=None):
        if x is not None and y is not None:
            self.x, self.y = x, y
        else:
            self.x, self.y = get_object_position(maze)
        self.color = RED
    def move(self, dx, dy, maze):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0:
            self.x, self.y = new_x, new_y
            return True
        return False

def draw_text(text, x, y, font, color=WHITE):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def draw_button(rect, text, font, bg_color=GRAY, text_color=WHITE):
    pygame.draw.rect(screen, bg_color, rect)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def draw_env(maze, player, obj, highlight_goal=True):
    for r in range(ROWS):
        for c in range(COLS):
            if maze[r][c] == 1:
                pygame.draw.rect(screen, WHITE, (c*GRID_SIZE, r*GRID_SIZE, GRID_SIZE, GRID_SIZE))
    if highlight_goal:
        pygame.draw.rect(screen, GREEN, (exit_x*GRID_SIZE, exit_y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.draw.rect(screen, obj.color, (obj.x*GRID_SIZE, obj.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))
    pygame.draw.rect(screen, player.color, (player.x*GRID_SIZE, player.y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

# --- Manual mode helpers
manual_input_seed = ""
manual_input_error = ""
manual_input_active = False
manual_player = None
manual_obj = None
manual_maze = None
manual_moves = 0
manual_collisions = 0
manual_task_complete = False
manual_seed_in_play = None

def reset_manual_mode(seed):
    global manual_player, manual_obj, manual_maze, manual_moves, manual_collisions, manual_task_complete, manual_seed_in_play
    manual_maze = generate_complex_maze(seed)
    ox, oy = get_object_position(manual_maze)
    manual_player = Player(1, 1)
    manual_obj = Object(manual_maze, ox, oy)
    manual_moves = 0
    manual_collisions = 0
    manual_task_complete = False
    manual_seed_in_play = seed
    print(f"[INFO] Manual Mode | SEED: {seed} | Robot Start: ({manual_player.x}, {manual_player.y}) | Object Start: ({manual_obj.x}, {manual_obj.y}) | Goal: ({exit_x}, {exit_y})")

def handle_manual_move(dx, dy):
    global manual_moves, manual_collisions, manual_task_complete
    prev_x, prev_y = manual_player.x, manual_player.y
    manual_player.move(dx, dy, manual_maze, manual_obj)
    manual_moves += 1
    if (manual_player.x, manual_player.y) == (prev_x, prev_y):
        manual_collisions += 1
    if (manual_obj.x, manual_obj.y) == (exit_x, exit_y):
        manual_task_complete = True

##############################################################################
# Q-learning RL Implementation + Baseline Reward Dynamic
##############################################################################

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # left, right, up, down
ACTION_IDS = [0, 1, 2, 3]

def state_to_tuple(player, obj):
    return (player.x, player.y, obj.x, obj.y)

class QLearningAgent:
    def __init__(self, maze):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.maze = maze

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTION_IDS)
        else:
            qs = [self.get_q(state, a) for a in ACTION_IDS]
            max_q = max(qs)
            actions = [a for a, q in zip(ACTION_IDS, qs) if q == max_q]
            return random.choice(actions)

    def update(self, state, action, reward, next_state, done):
        old_q = self.get_q(state, action)
        next_q = max([self.get_q(next_state, a) for a in ACTION_IDS]) if not done else 0
        new_q = old_q + self.lr * (reward + self.gamma * next_q - old_q)
        self.q_table[(state, action)] = new_q

    def set_epsilon(self, e):
        self.epsilon = e

def env_step(player, obj, maze, action, visited_obj_history, contact_state):
    dx, dy = ACTIONS[action]
    reward = -1                      # Step penalty
    prev_dist = abs(obj.x - exit_x) + abs(obj.y - exit_y)
    made_contact = False
    first_contact = False
    collided = False

    # PUSH
    if (player.x + dx, player.y + dy) == (obj.x, obj.y):
        made_contact = True
        if not contact_state[0]:
            first_contact = True
            contact_state[0] = True
        if object_move(obj, dx, dy, maze):
            player.x += dx
            player.y += dy
        else:
            collided = True

    # PULL
    elif (player.x - dx, player.y - dy) == (obj.x, obj.y):
        made_contact = True
        if not contact_state[0]:
            first_contact = True
            contact_state[0] = True
        new_x = player.x + dx
        new_y = player.y + dy
        obj_new_x = player.x
        obj_new_y = player.y
        if (0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0 and
            0 <= obj_new_x < COLS and 0 <= obj_new_y < ROWS and maze[obj_new_y][obj_new_x] == 0):
            old_x, old_y = player.x, player.y
            player.x, player.y = new_x, new_y
            obj.x, obj.y = old_x, old_y
        else:
            collided = True

    # NORMAL MOVE
    else:
        new_x, new_y = player.x + dx, player.y + dy
        if 0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0:
            player.x, player.y = new_x, new_y
        else:
            collided = True

    curr_dist = abs(obj.x - exit_x) + abs(obj.y - exit_y)

    # Reward Shaping
    if made_contact:
        if first_contact:
            reward += 50
        else:
            reward += 5
    else:
        if contact_state[0]:
            reward -= 15

    if curr_dist < prev_dist:
        reward += 10 * (prev_dist - curr_dist)
    elif curr_dist > prev_dist:
        reward -= 5 * (curr_dist - prev_dist)

    if collided:
        reward -= 10

    # Jiggle penalty (for object position repeated in last 5 moves)
    if (obj.x, obj.y) in list(visited_obj_history)[-5:]:
        reward -= 15

    done = (obj.x, obj.y) == (exit_x, exit_y)
    if done:
        reward += 1000

    return reward, done, collided

def object_move(obj, dx, dy, maze):
    new_x, new_y = obj.x + dx, obj.y + dy
    if 0 <= new_x < COLS and 0 <= new_y < ROWS and maze[new_y][new_x] == 0:
        obj.x, obj.y = new_x, new_y
        return True
    return False

def save_q_table(q_table):
    with open(Q_TABLE_FILE, "wb") as f:
        pickle.dump(q_table, f)

def load_q_table():
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def run_qlearning(train_seeds, test_seeds, train_episodes, max_steps):
    agent = None
    q_table = load_q_table()
    total_train_success = 0
    total_train_fail = 0
    baseline_window = 25
    episode_rewards = deque(maxlen=baseline_window)
    best_reward_so_far = -float("inf")

    # Training phase
    for seed in train_seeds:
        print(f"\n[INFO] === Training on Seed {seed} ===")
        maze = generate_complex_maze(seed)
        fixed_obj = Object(maze)
        fixed_obj_x, fixed_obj_y = fixed_obj.x, fixed_obj.y
        seed_success = 0
        for ep in range(train_episodes):
            player = Player(1, 1)
            obj = Object(maze, fixed_obj_x, fixed_obj_y)
            agent = QLearningAgent(maze)
            agent.q_table = q_table
            agent.set_epsilon(0.2)
            state = state_to_tuple(player, obj)
            moves = 0
            collisions = 0
            contact_state = [False]  # [first_contact_flag]
            visited_obj_history = deque(maxlen=20)
            total_reward = 0
            start_time = time.time()
            done = False
            for step in range(max_steps):
                screen.fill(BLACK)
                draw_env(maze, player, obj)
                draw_text(f"Training Seed {seed} Ep {ep+1}/{train_episodes}", 10, 10, font, YELLOW)
                pygame.display.flip()
                time.sleep(RL_STEP_DELAY)

                action = agent.select_action(state)
                reward, done, collided = env_step(player, obj, maze, action, visited_obj_history, contact_state)
                next_state = state_to_tuple(player, obj)
                agent.update(state, action, reward if not done else 100, next_state, done)
                state = next_state
                total_reward += reward
                if collided:
                    collisions += 1
                moves += 1
                visited_obj_history.append((obj.x, obj.y))
                if done:
                    elapsed = time.time() - start_time
                    print(f"[INFO] TRAIN GOAL REACHED: Seed {seed}, Ep {ep+1}, Moves: {moves}, Collisions: {collisions}, Time: {elapsed:.2f}s, Total Reward: {total_reward:.2f}")
                    seed_success += 1
                    break
            else:
                # Failed to reach goal
                total_reward -= 500  # Absolute failure penalty
                print(f"[INFO] TRAINING MAX STEPS REACHED: Seed {seed}, Ep {ep+1}, Moves: {moves}, Collisions: {collisions}, Total Reward: {total_reward:.2f}")

            # --- Baseline Reward Dynamic
            if len(episode_rewards) == baseline_window:
                running_avg = np.mean(episode_rewards)
                if total_reward > running_avg:
                    total_reward += 50  # Bonus for beating the recent baseline
                    print(f"[INFO] BONUS: Reward {total_reward:.2f} beats baseline {running_avg:.2f}. +50 bonus!")
                else:
                    total_reward -= 25  # Penalty for below baseline
                    print(f"[INFO] PENALTY: Reward {total_reward:.2f} below baseline {running_avg:.2f}. -25 penalty!")
            episode_rewards.append(total_reward)
            if total_reward > best_reward_so_far:
                best_reward_so_far = total_reward

            q_table = agent.q_table

        print(f"[INFO] TRAINING Seed {seed} Success Rate: {seed_success/train_episodes:.2%}")
        total_train_success += seed_success
        total_train_fail += (train_episodes - seed_success)

    overall_train_acc = total_train_success / (len(train_seeds)*train_episodes)
    print(f"\n[INFO] === Training Summary ===")
    print(f"[INFO] TRAINING Overall Success Rate: {overall_train_acc:.2%}")
    print(f"[INFO] Highest Training Reward: {best_reward_so_far:.2f}")

    save_q_table(q_table)

    # Testing phase
    total_test_success = 0
    total_test_fail = 0
    best_test_reward = -float("inf")
    for seed in test_seeds:
        print(f"\n[INFO] === Testing on Seed {seed} ===")
        maze = generate_complex_maze(seed)
        fixed_obj = Object(maze)
        fixed_obj_x, fixed_obj_y = fixed_obj.x, fixed_obj.y
        player = Player(1, 1)
        obj = Object(maze, fixed_obj_x, fixed_obj_y)
        agent = QLearningAgent(maze)
        agent.q_table = q_table
        agent.set_epsilon(0.0)
        state = state_to_tuple(player, obj)
        moves = 0
        collisions = 0
        contact_state = [False]
        visited_obj_history = deque(maxlen=20)
        total_reward = 0
        start_time = time.time()
        success = False
        for step in range(max_steps):
            screen.fill(BLACK)
            draw_env(maze, player, obj)
            draw_text(f"Testing Seed {seed}", 10, 10, font, YELLOW)
            pygame.display.flip()
            time.sleep(RL_STEP_DELAY)

            action = agent.select_action(state)
            reward, done, collided = env_step(player, obj, maze, action, visited_obj_history, contact_state)
            state = state_to_tuple(player, obj)
            total_reward += reward
            if collided:
                collisions += 1
            moves += 1
            visited_obj_history.append((obj.x, obj.y))
            if done:
                elapsed = time.time() - start_time
                print(f"[INFO] TEST GOAL REACHED: Seed {seed}, Moves: {moves}, Collisions: {collisions}, Time: {elapsed:.2f}s, Total Reward: {total_reward:.2f}")
                total_test_success += 1
                best_test_reward = max(best_test_reward, total_reward)
                success = True
                break
        if not success:
            total_reward -= 500  # Absolute failure penalty
            print(f"[INFO] TEST MAX STEPS REACHED: Seed {seed}, Moves: {moves}, Collisions: {collisions}, Total Reward: {total_reward:.2f}")
            total_test_fail += 1

    overall_test_acc = total_test_success / len(test_seeds)
    print(f"\n[INFO] === Testing Summary ===")
    print(f"[INFO] TESTING Overall Success Rate: {overall_test_acc:.2%}")
    print(f"[INFO] Highest Test Reward: {best_test_reward:.2f}")

    print("\n[INFO] RL Training & Testing Complete.")
    return q_table

##############################################################################
#                              Main Game Loop                                #
##############################################################################
running = True
rl_train_once = False
rl_test_once = False
error_msg = ""
input_box_rect = pygame.Rect(SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 15, 160, 40)

while running:
    screen.fill(BLACK)
    if current_state == STATE_WELCOME:
        draw_text("Teleoperation Maze Game", SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//4, font, WHITE)
        if time.time() - state_start_time > 3:
            current_state = STATE_MODE_SELECT

    elif current_state == STATE_MODE_SELECT:
        draw_text("Select Mode", SCREEN_WIDTH//2 - 110, 100, font, WHITE)
        draw_button(button_manual, "Manual Mode", font)
        draw_button(button_rl_train, "Train Q-learning", font, bg_color=YELLOW, text_color=BLACK)
        draw_button(button_rl_test, "Test Q-learning", font, bg_color=YELLOW, text_color=BLACK)

    elif current_state == STATE_MANUAL_INPUT:
        draw_text("Enter a valid seed for manual mode:", SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2 - 70, font, WHITE)
        pygame.draw.rect(screen, ORANGE if manual_input_active else WHITE, input_box_rect, 2)
        draw_text(manual_input_seed, input_box_rect.x + 10, input_box_rect.y + 5, font, WHITE)
        if manual_input_error:
            draw_text(manual_input_error, SCREEN_WIDTH//2 - 180, SCREEN_HEIGHT//2 + 40, font_small, RED)
        draw_button(button_back, "Back", font)

    elif current_state == STATE_MANUAL:
        draw_env(manual_maze, manual_player, manual_obj)
        draw_text(f"Seed: {manual_seed_in_play} | Moves: {manual_moves} | Collisions: {manual_collisions}", 20, 10, font_small, YELLOW)
        draw_text("Move: Arrow Keys or WASD | Back: ESC", 20, SCREEN_HEIGHT - 40, font_small, YELLOW)
        draw_button(button_back, "Back", font)
        if manual_task_complete:
            draw_text("Task Complete!", SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2, font, GREEN)
            draw_text("Press any key to return.", SCREEN_WIDTH//2 - 130, SCREEN_HEIGHT//2 + 40, font, WHITE)

    elif current_state == STATE_RL_TRAIN:
        draw_text("RL Training in Progress...", SCREEN_WIDTH//2 - 180, SCREEN_HEIGHT//2, font, YELLOW)
        pygame.display.flip()
        if not rl_train_once:
            run_qlearning(TRAIN_SEEDS, TEST_SEEDS, MAX_EPISODES, MAX_STEPS)
            rl_train_once = True
            current_state = STATE_RL_COMPLETE

    elif current_state == STATE_RL_TEST:
        draw_text("RL Testing in Progress...", SCREEN_WIDTH//2 - 180, SCREEN_HEIGHT//2, font, YELLOW)
        pygame.display.flip()
        if not rl_test_once:
            run_qlearning([], TEST_SEEDS, 1, MAX_STEPS)  # Only test seeds, don't retrain
            rl_test_once = True
            current_state = STATE_RL_COMPLETE

    elif current_state == STATE_RL_COMPLETE:
        draw_text("RL Training & Testing Complete!", SCREEN_WIDTH//2 - 200, SCREEN_HEIGHT//2, font, GREEN)
        draw_text("Press any key to return.", SCREEN_WIDTH//2 - 130, SCREEN_HEIGHT//2 + 40, font, WHITE)

    elif current_state == STATE_ERROR:
        draw_text(error_msg, SCREEN_WIDTH//2 - 220, SCREEN_HEIGHT//2, font, RED)
        draw_text("Press any key to return.", SCREEN_WIDTH//2 - 130, SCREEN_HEIGHT//2 + 40, font, WHITE)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if current_state == STATE_MODE_SELECT:
                if button_manual.collidepoint(event.pos):
                    manual_input_seed = ""
                    manual_input_error = ""
                    manual_input_active = True
                    current_state = STATE_MANUAL_INPUT
                elif button_rl_train.collidepoint(event.pos):
                    rl_train_once = False
                    current_state = STATE_RL_TRAIN
                elif button_rl_test.collidepoint(event.pos):
                    rl_test_once = False
                    current_state = STATE_RL_TEST
            elif current_state == STATE_MANUAL_INPUT:
                if input_box_rect.collidepoint(event.pos):
                    manual_input_active = True
                elif button_back.collidepoint(event.pos):
                    current_state = STATE_MODE_SELECT
            elif current_state == STATE_MANUAL:
                if button_back.collidepoint(event.pos):
                    current_state = STATE_MODE_SELECT

        elif event.type == pygame.KEYDOWN:
            if current_state == STATE_MANUAL_INPUT and manual_input_active:
                if event.key == pygame.K_RETURN:
                    try:
                        entered_seed = int(manual_input_seed)
                        if entered_seed in solvable_seeds:
                            reset_manual_mode(entered_seed)
                            current_state = STATE_MANUAL
                            manual_input_error = ""
                        else:
                            manual_input_error = "Seed not in solvable_seeds.txt!"
                    except ValueError:
                        manual_input_error = "Please enter a valid integer seed!"
                elif event.key == pygame.K_BACKSPACE:
                    manual_input_seed = manual_input_seed[:-1]
                else:
                    if len(manual_input_seed) < 8 and event.unicode.isdigit():
                        manual_input_seed += event.unicode
            elif current_state == STATE_MANUAL:
                if not manual_task_complete:
                    if event.key in [pygame.K_LEFT, pygame.K_a]:
                        handle_manual_move(-1, 0)
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        handle_manual_move(1, 0)
                    elif event.key in [pygame.K_UP, pygame.K_w]:
                        handle_manual_move(0, -1)
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        handle_manual_move(0, 1)
                if manual_task_complete or event.key == pygame.K_ESCAPE:
                    current_state = STATE_MODE_SELECT
            elif current_state == STATE_MANUAL_INPUT and not manual_input_active:
                current_state = STATE_MODE_SELECT
            elif current_state == STATE_RL_COMPLETE:
                current_state = STATE_MODE_SELECT
            elif current_state == STATE_ERROR:
                current_state = STATE_MODE_SELECT

    time.sleep(0.05)

pygame.quit()
sys.exit()
