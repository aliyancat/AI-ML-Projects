import pygame
import random
import time
import sys

pygame.init()

GRID_SIZE = 10
CELL_SIZE = 60
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
PURPLE = (128, 0, 128)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Grid Car RL Game")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

road = [[0 for i in range(10)] for j in range(10)]
startpos = [0,0]
Goal = [9,9]

road[2][3] = 1
road[5][4] = 1
road[7][2] = 1
road[9][8] = 1

x = 0
y = 0

def move_car(action):
    global x, y
    if action == 0:
        if y > 0:
            y = y - 1
    elif action == 1:
        if y < 9:
            y = y + 1
    elif action == 2:
        if x > 0:
            x = x - 1
    elif action == 3:
        if x < 9:
            x = x + 1

def get_reward():
    if road[y][x] == 1:
        return -10
    elif x == 9 and y == 9:
        return 100
    else:
        return -1

def is_game_over():
    if road[y][x] == 1 or (x == 9 and y == 9):
        return True
    return False

def reset_game():
    global x, y
    x = 0
    y = 0

q_table = []
for row in range(10):
    row_list = []
    for col in range(10):
        actions = [0.0, 0.0, 0.0, 0.0]
        row_list.append(actions)
    q_table.append(row_list)

learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

def choose_action():
    random_number = random.random()
    if random_number < epsilon:
        return random.randint(0, 3)
    else:
        best_action = 0
        best_value = q_table[y][x][0]
        for i in range(4):
            if q_table[y][x][i] > best_value:
                best_value = q_table[y][x][i]
                best_action = i
        return best_action

def update_q_table(old_x, old_y, action, reward, new_x, new_y):
    old_q = q_table[old_y][old_x][action]
    max_future_q = max(q_table[new_y][new_x])
    new_q = old_q + learning_rate * (reward + discount_factor * max_future_q - old_q)
    q_table[old_y][old_x][action] = new_q

def draw_grid():
    screen.fill(WHITE)
    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, GRID_SIZE * CELL_SIZE))
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (GRID_SIZE * CELL_SIZE, i * CELL_SIZE))
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if road[row][col] == 1:
                pygame.draw.rect(screen, RED, 
                               (col * CELL_SIZE + 2, row * CELL_SIZE + 2, 
                                CELL_SIZE - 4, CELL_SIZE - 4))
    pygame.draw.rect(screen, GREEN, 
                    (9 * CELL_SIZE + 2, 9 * CELL_SIZE + 2, 
                     CELL_SIZE - 4, CELL_SIZE - 4))
    pygame.draw.circle(screen, BLUE, 
                      (x * CELL_SIZE + CELL_SIZE // 2, 
                       y * CELL_SIZE + CELL_SIZE // 2), 
                      CELL_SIZE // 3)

def draw_info(test_num, step, total_reward, successes, action_name):
    info_y = GRID_SIZE * CELL_SIZE + 10
    pygame.draw.rect(screen, GRAY, (0, GRID_SIZE * CELL_SIZE, WINDOW_WIDTH, 100))
    test_text = small_font.render(f"Test: {test_num}/100", True, BLACK)
    step_text = small_font.render(f"Step: {step}", True, BLACK)
    reward_text = small_font.render(f"Reward: {total_reward}", True, BLACK)
    success_text = small_font.render(f"Successes: {successes}", True, BLACK)
    action_text = small_font.render(f"Action: {action_name}", True, BLACK)
    pos_text = small_font.render(f"Position: ({x}, {y})", True, BLACK)
    screen.blit(test_text, (10, info_y))
    screen.blit(step_text, (120, info_y))
    screen.blit(reward_text, (200, info_y))
    screen.blit(success_text, (300, info_y))
    screen.blit(action_text, (10, info_y + 25))
    screen.blit(pos_text, (150, info_y + 25))

def train_silent():
    for episode in range(1000):
        reset_game()
        while not is_game_over():
            old_x = x
            old_y = y
            action = choose_action()
            move_car(action)
            reward = get_reward()
            update_q_table(old_x, old_y, action, reward, x, y)

def run_visual_test(test_num, successes):
    reset_game()
    steps = 0
    total_reward = 0
    action_names = ["Up", "Down", "Left", "Right"]
    last_action = 0
    while not is_game_over() and steps < 100:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    waiting = True
                    while waiting:
                        for pause_event in pygame.event.get():
                            if pause_event.type == pygame.QUIT:
                                pygame.quit()
                                sys.exit()
                            elif pause_event.type == pygame.KEYDOWN:
                                if pause_event.key == pygame.K_SPACE:
                                    waiting = False
        action = choose_action()
        last_action = action
        move_car(action)
        reward = get_reward()
        total_reward += reward
        draw_grid()
        draw_info(test_num, steps + 1, total_reward, successes, action_names[action])
        if reward == -10:
            crash_text = font.render("CRASHED!", True, RED)
            screen.blit(crash_text, (WINDOW_WIDTH // 2 - 60, WINDOW_HEIGHT // 2))
        elif reward == 100:
            success_text = font.render("SUCCESS!", True, GREEN)
            screen.blit(success_text, (WINDOW_WIDTH // 2 - 60, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        if reward == -10 or reward == 100:
            time.sleep(1)
            break
        steps += 1
        clock.tick(3)
    if steps >= 100:
        timeout_text = font.render("TIMEOUT!", True, PURPLE)
        screen.blit(timeout_text, (WINDOW_WIDTH // 2 - 60, WINDOW_HEIGHT // 2))
        pygame.display.flip()
        time.sleep(1)
    return reward == 100

def main():
    train_silent()
    screen.fill(WHITE)
    title_text = font.render("Grid Car RL Game", True, BLACK)
    instruction1 = small_font.render("Watch the blue car learn to reach the green goal!", True, BLACK)
    instruction2 = small_font.render("Red squares are obstacles", True, BLACK)
    instruction3 = small_font.render("Press SPACE to pause, ESC to quit", True, BLACK)
    instruction4 = small_font.render("Starting tests in 3 seconds...", True, BLACK)
    screen.blit(title_text, (WINDOW_WIDTH // 2 - 120, 100))
    screen.blit(instruction1, (WINDOW_WIDTH // 2 - 180, 200))
    screen.blit(instruction2, (WINDOW_WIDTH // 2 - 100, 230))
    screen.blit(instruction3, (WINDOW_WIDTH // 2 - 120, 260))
    screen.blit(instruction4, (WINDOW_WIDTH // 2 - 120, 290))
    pygame.display.flip()
    time.sleep(3)
    test_count = 0
    successes = 0
    try:
        while test_count < 100:
            test_count += 1
            success = run_visual_test(test_count, successes)
            if success:
                successes += 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    screen.fill(WHITE)
    final_text = font.render(f"Final Results:", True, BLACK)
    tests_text = small_font.render(f"Tests completed: {test_count}", True, BLACK)
    success_text = small_font.render(f"Successes: {successes}", True, BLACK)
    rate_text = small_font.render(f"Success rate: {successes/test_count*100:.1f}%", True, BLACK)
    screen.blit(final_text, (WINDOW_WIDTH // 2 - 80, 200))
    screen.blit(tests_text, (WINDOW_WIDTH // 2 - 80, 250))
    screen.blit(success_text, (WINDOW_WIDTH // 2 - 80, 280))
    screen.blit(rate_text, (WINDOW_WIDTH // 2 - 80, 310))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False
    pygame.quit()

if __name__ == "__main__":
    main()
