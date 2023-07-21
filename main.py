import os
import pygame
import random
import numpy as np
import math
from collections import deque
import copy


# Game settings
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400
SNAKE_SIZE = 20
APPLE_SIZE = 20
GAME_SPEED = 100

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

class Snake:
    def __init__(self):
        self.length = 1
        self.body = [[SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]]
        self.direction = 0  # 0: up, 1: right, 2: down, 3: left
        self.grow = False
        self.vision_distance = 5

    @property
    def head(self):
        return self.body[0]

    def move(self):
        if self.direction == 0:  # up
            new_head = [self.head[0], self.head[1] - SNAKE_SIZE]
        elif self.direction == 1:  # right
            new_head = [self.head[0] + SNAKE_SIZE, self.head[1]]
        elif self.direction == 2:  # down
            new_head = [self.head[0], self.head[1] + SNAKE_SIZE]
        elif self.direction == 3:  # left
            new_head = [self.head[0] - SNAKE_SIZE, self.head[1]]

        self.body.insert(0, new_head)

        if not self.grow:
            self.body.pop()
        else:
            self.grow = False


        print(f"Snake's new position: {self.head}")

    def get_direction_vector(self, relative_direction):
        print(f"Relative direction: {relative_direction}")
        direction_vec = (0, 0)  # default direction
        if self.direction == 0:  # up
            direction_vec = (0, -1) if relative_direction == 0 else (-1, 0) if relative_direction == -1 else (1, 0)
        elif self.direction == 2:  # down
            direction_vec = (0, 1) if relative_direction == 0 else (1, 0) if relative_direction == -1 else (-1, 0)
        elif self.direction == 3:  # left
            direction_vec = (-1, 0) if relative_direction == 0 else (0, 1) if relative_direction == -1 else (0, -1)
        elif self.direction == 1:  # right
            direction_vec = (1, 0) if relative_direction == 0 else (0, -1) if relative_direction == -1 else (0, 1)
        print(f"Direction vector: {direction_vec}")
        return direction_vec


class Apple:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH - APPLE_SIZE) // APPLE_SIZE * APPLE_SIZE
        self.y = random.randint(0, SCREEN_HEIGHT - APPLE_SIZE) // APPLE_SIZE * APPLE_SIZE
        self.size = APPLE_SIZE

class State:
    def __init__(self):
        self.is_ahead_clear = True
        self.is_left_clear = True
        self.is_right_clear = True
        self.is_apple_ahead = False
        self.is_apple_left = False
        self.is_apple_right = False
        self.apple_x = 0
        self.apple_y = 0
        self.snake_x = 0
        self.snake_y = 0
        self.apple_distance = 0
        self.apple_angle = 0
        self.snake_length = 1
        self.snake_direction = 0

    def update(self, snake, apple):
        # Reset state
        self.is_ahead_clear = True
        self.is_left_clear = True
        self.is_right_clear = True
        self.is_apple_ahead = False
        self.is_apple_left = False
        self.is_apple_right = False

        self.apple_x = apple.x
        self.apple_y = apple.y
        self.snake_x = snake.head[0]
        self.snake_y = snake.head[1]

        # Update apple distance and angle
        dx, dy = snake.get_direction_vector(0)  # direction vector of the snake
        apple_dx = apple.x - snake.head[0]  # x difference to apple
        apple_dy = apple.y - snake.head[1]  # y difference to apple
        self.apple_distance = ((apple_dx) ** 2 + (apple_dy) ** 2) ** 0.5
        dot_product = dx * apple_dx + dy * apple_dy  # dot product
        norm_product = (dx ** 2 + dy ** 2) ** 0.5 * (apple_dx ** 2 + apple_dy ** 2) ** 0.5  # product of norms
        self.apple_angle = math.acos(dot_product / norm_product)  # angle in radians

        # Update the snake's length
        self.snake_length = len(snake.body)

        self.direction = snake.direction

        # Check for danger and apple in each direction
        for i in range(-1, 2):
            dx, dy = snake.get_direction_vector(i)

            # Check for danger
            for step in range(1, snake.vision_distance + 1):
                x = snake.head[0] + dx * step
                y = snake.head[1] + dy * step

                if (x, y) in snake.body or x < 0 or y < 0 or x >= SCREEN_WIDTH or y >= SCREEN_HEIGHT:
                    if i == -1:
                        self.is_left_clear = False
                    elif i == 0:
                        self.is_ahead_clear = False
                    elif i == 1:
                        self.is_right_clear = False
                    break

                # Check for apple
                if (x, y) == (apple.x, apple.y):
                    if i == -1:
                        self.is_apple_left = True
                    elif i == 0:
                        self.is_apple_ahead = True
                    elif i == 1:
                        self.is_apple_right = True

        print(f"Updated state: {self.__dict__}")



class RLSnake:
    def __init__(self):
        self.q_table = np.zeros((2 ** 6, 3))
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.alpha = 0.3
        self.gamma = 0.2
        self.snake_history = deque(maxlen=100)  # stores the last 100 positions and directions

    def state_to_int(self, state):
        return (int(state.is_ahead_clear) * 2 ** 0 +
                int(state.is_left_clear) * 2 ** 1 +
                int(state.is_right_clear) * 2 ** 2 +
                int(state.is_apple_ahead) * 2 ** 3 +
                int(state.is_apple_left) * 2 ** 4 +
                int(state.is_apple_right) * 2 ** 5)

    def softmax(self, state):
        state_index = self.state_to_int(state)
        q_values = self.q_table[state_index]
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))  # softmax calculation
        action = np.random.choice(len(q_values), p=probabilities)
        return action

    def update(self, state, snake, apple):
        # Reset state
        self.is_ahead_clear = True
        self.is_left_clear = True
        self.is_right_clear = True
        self.is_apple_ahead = False
        self.is_apple_left = False
        self.is_apple_right = False
        self.direction = snake.direction

        # Update apple distance and angle
        self.apple_distance = ((apple.x - snake.head[0]) ** 2 + (apple.y - snake.head[1]) ** 2) ** 0.5
        self.apple_angle = math.atan2(apple.y - snake.head[1], apple.x - snake.head[0])

        # Update snake length
        self.snake_length = snake.length
        state_index = self.state_to_int(state)
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(self.q_table[state_index]).item()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return action

    def learn(self, old_state, new_state, action, reward, game_over):
        old_state_index = self.state_to_int(old_state)
        new_state_index = self.state_to_int(new_state)

        # Check if the game ended because the snake hit the wall
        if game_over:
            if (snake.head[0] < 0 or snake.head[0] >= SCREEN_WIDTH or
                    snake.head[1] < 0 or snake.head[1] >= SCREEN_HEIGHT):
                reward = -100
            else:
                reward = -100
        elif reward > 0:
            reward = 100
        else:
            # Calculate the distance to the apple in the old and new states
            old_distance = ((old_state.apple_x - old_state.snake_x) ** 2 + (
                    old_state.apple_y - old_state.snake_y) ** 2) ** 0.5
            new_distance = ((new_state.apple_x - new_state.snake_x) ** 2 + (
                    new_state.apple_y - new_state.snake_y) ** 2) ** 0.5

            # Give a higher reward if the snake moved closer to the apple, and a lower reward if it moved further away
            if new_distance < old_distance:
                reward = 1
            else:
                reward = -1

        # Give a small negative reward for changing direction
        if old_state.snake_direction != new_state.snake_direction:
            reward -= 2

        # Check if the snake has made a loop
        current_state = (new_state.snake_x, new_state.snake_y, new_state.snake_direction)

        if current_state in self.snake_history:
            reward -= 50
        self.snake_history.append(current_state)

        # Update Q-value using alpha and gamma
        old_value = self.q_table[old_state_index, action]
        next_max = np.max(self.q_table[new_state_index])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[old_state_index, action] = new_value

        print(f"Old Q-value: {old_value}")
        print(f"New Q-value: {new_value}")

        print(f"Old state: {old_state}, New state: {new_state}, Action: {action}, Reward: {reward}")
        print(f"Updated Q-table: {self.q_table}")

    def simulate_action(self, state, action):
        # Create a copy of the current state
        new_state = copy.deepcopy(state)

        # Simulate the action based on the current state
        if action == 0:  # Go straight
            new_state.snake_x += new_state.direction[0]
            new_state.snake_y += new_state.direction[1]
        elif action == 1:  # Turn left
            new_state.direction = ((new_state.direction[0] - 1) % 4, (new_state.direction[1] - 1) % 4)
        elif action == 2:  # Turn right
            new_state.direction = ((new_state.direction[0] + 1) % 4, (new_state.direction[1] + 1) % 4)

        return new_state

    def does_action_hit_wall(self, state, action):
        # Simulate the action based on the current state
        new_state = self.simulate_action(state, action)

        # Check if the new state is a wall state
        if new_state.is_wall_state():
            return True

        return False

    def choose_action(self, state):
        state_index = self.state_to_int(state)
        q_values = self.q_table[state_index]
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))  # softmax calculation

        # Sort the actions based on their softmax probabilities
        sorted_actions = np.argsort(probabilities)[::-1]

        for action in sorted_actions:
            # Simulate the action and check if it leads to hitting the wall
            # If it does, continue to the next action
            if self.does_action_hit_wall(state, action):
                continue

            # If the action does not lead to hitting the wall, choose it
            return action

        # If all actions lead to hitting the wall, choose the action with the highest softmax probability
        return sorted_actions[0]


# Game loop
while True:
    # Initialize game objects
    snake = Snake()
    apple = Apple()
    state = State()
    rl_snake = RLSnake()

    # Load Q-table from file at start of each game
    if os.path.isfile('q_table.txt') and os.path.getsize('q_table.txt') > 0:
        rl_snake.q_table = np.loadtxt('q_table.txt').reshape((2 ** 6, 3))

    old_state = state
    old_action = 0

    # Game loop
    while True:
        # Update game state
        state.update(snake, apple)

        # Update RL agent
        action = rl_snake.update(state, snake, apple)

        # Update snake direction based on action
        if action == 0:  # Go straight
            pass
        elif action == 1:  # Turn left
            if snake.direction != 1:  # Prevent from going right if currently going left
                snake.direction = (snake.direction - 1) % 4
        elif action == 2:  # Turn right
            if snake.direction != 3:  # Prevent from going left if currently going right
                snake.direction = (snake.direction + 1) % 4

        # Move snake
        snake.move()

        # Check if snake ate apple
        if snake.head == [apple.x, apple.y]:
            apple = Apple()
            snake.grow = True

        state.update(snake, apple)

        if snake.grow:
            # Give the agent a reward and update the Q-table.
            rl_snake.learn(old_state, state, old_action, 1, False)
        else:
            # If the snake didn't eat an apple, give a negative reward.
            rl_snake.learn(old_state, state, old_action, -1, False)

        state.update(snake, apple)

        old_state = state
        old_action = action

        # Check for game over
        if (snake.head[0] < 0 or snake.head[0] >= SCREEN_WIDTH or
                snake.head[1] < 0 or snake.head[1] >= SCREEN_HEIGHT or
                snake.head in snake.body[:-1]):
            rl_snake.learn(old_state, state, old_action, -1, True)
            break

        # Draw everything
        screen.fill(BLACK)
        for x, y in snake.body:
            pygame.draw.rect(screen, WHITE, pygame.Rect(x, y, SNAKE_SIZE, SNAKE_SIZE))
        pygame.draw.rect(screen, RED, pygame.Rect(apple.x, apple.y, APPLE_SIZE, APPLE_SIZE))

        # Flip the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(GAME_SPEED)

    # Save Q-table to file after each game
    np.savetxt('q_table.txt', rl_snake.q_table)

pygame.quit()