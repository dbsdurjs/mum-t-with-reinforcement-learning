import numpy as np
from dddqn import Agent
from custom_env import ENV
import tensorflow as tf
import tensorboard
import pygame
import sys
from pygame.locals import *
import matplotlib.pyplot as plt
import math
from tensorflow.python.client import device_lib
from dijkstra_main import Spot

## 컬러 세팅 ##
PURPLE = (128, 0, 128)  #uav

ORANGE = (255, 165, 0)  #drone1
YELLOW = (255, 255, 0)  #drone2
BLUE = (0, 0, 255)  #drone3
BLACK = (0, 0, 0)   #drone4

RED = (255, 0, 0)   #obstacle
GREEN = (0, 255, 0) #target
WHITE = (255, 255, 255) #background

drone_color = [ORANGE, YELLOW, BLUE, BLACK]
def main():
    print('checking use gpu', device_lib.list_local_devices())

    pygame.init()
    FPS = 180
    FramePerSec = pygame.time.Clock()

    ## 게임 창 설정 ##
    GameDisplay = pygame.display.set_mode((300, 300))
    GameDisplay.fill(WHITE)  # 하얀색으로 배경 채우기
    pygame.display.set_caption("MUM-T")  # 창 이름 설정
    num_agent = 4

    back_img = pygame.image.load('C:/Users/desktop/anaconda3/envs/temp/map.png')
    back_img = pygame.transform.scale(back_img, (300, 300))

    env = ENV()
    agents = [Agent(env=env, lr=1e-4, gamma=0.99, n_actions=4, num_agents=num_agent, epsilon=1.0, batch_size=32,
                    input_dims=[2, ]) for _ in range(num_agent)]

    n_games = 170  #170
    ddqn_scores = [[i] for i in range(num_agent)]
    optimal_paths = [[i] for i in range(num_agent)]
    optimal_drones_coordinate = [[i] for i in range(num_agent)]
    optimal_obstacles_coordinate = [[i] for i in range(num_agent)]
    temp = [0 for _ in range(num_agent)]

    evaluate = False
    print(evaluate)
    if evaluate is True: n_games = 1

    # 두 점 간의 유클리드 거리 계산
    def calculate_distance(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return int(distance)

    for i in range(n_games):
        print('{}번 째 학습'.format(i))
        dones = [False, False, False, False]
        scores = [0, 0, 0, 0]
        rewards = np.zeros((num_agent,))
        checking_obstacles = [0, 0, 0, 0]
        checking_target = [0, 0, 0, 0]
        observations = [env.reset(i) for i in range(num_agent)]
        print(
            f"initial state1 : {observations[0]} | initial state2 : {observations[1]} | initial state3 : {observations[2]} | initial state4 : {observations[3]} ")

        while not all(dones):
            # if you train model, change choose_action's evaluate to False
            actions = [agents[i].choose_action(observations[i], evaluate=False) for i in range(num_agent)]

            observations_ = []
            for k in range(num_agent):
                obs, r, d, co, paths, odc, obc, checking = env.step(actions[k], k, dones[k])
                observations_.append(obs)
                normalized_reward = (r + 10) / 200
                rewards[k] = normalized_reward
                dones[k] = d
                checking_obstacles[k] += co
                drones_coordinate = odc
                obstacles_coordinate = obc
                checking_target[k] = checking

            print('checking target', checking_target)
            # 각 드론끼리의 거리 계산
            dist_drones = []
            for i in range(num_agent - 1):
                for j in range(i + 1, num_agent):
                    dist_drones.append(calculate_distance(observations_[i], observations_[j]))

            # 일정 거리 시 보상 감소
            for i in range(num_agent):
                if dist_drones[i] < 5:
                    print(str(i + 1) + '번 드론 충돌!')
                    rewards[i] -= 10

            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # 창의 닫기 버튼(X 버튼)을 누르면 종료
                    pygame.quit()

            GameDisplay.fill(WHITE)
            GameDisplay.blit(back_img, (0, 0))

            for i in range(num_agent):
                pygame.draw.circle(GameDisplay, drone_color[i], env.agent_pos[i], 5)
                pygame.draw.circle(GameDisplay, RED, env.obstacle_list[i][:2], env.obstacle_radius)

            for k in range(len(env.target_state)):
                pygame.draw.circle(GameDisplay, GREEN, env.target_state[k], 15)

            pygame.draw.circle(GameDisplay, PURPLE, env.uav_state, 10)


            pygame.display.update()
            FramePerSec.tick(FPS)
            for i in range(num_agent):
                print(
                    f"  - action{i}: {actions[i]} | state{i}: {observations_[i]} | reward{i}: {rewards[i] : .2f} | epsilon{i}: {agents[i].epsilon : .2f} | ")
                scores[i] += rewards[i]
                agents[i].store_transition(observations[i], actions[i], rewards[i], observations_[i], dones[i], i)
                if not evaluate:
                    agents[i].learn(i)
                observations[i] = observations_[i]
                ddqn_scores[i].append(scores[i])
            print('===========================================================================')

        # 평균과 표준 편차 구하기
        mean_val = np.mean(scores)
        std_val = np.std(scores)

        # 정규화
        normalized_scores = [(item - mean_val) / std_val for item in scores]

        # 장애물을 만나지 않고 최고의 보상값을 가지고 있다면 최적의 경로
        for j in range(num_agent):
            if (checking_target[j] > 0): #checking_obstacles[j] == 0 and
                print('최적의 경로 탐색!')
                temp[j] += 1
                optimal_paths[j] = paths[j]
                optimal_obstacles_coordinate[j] = obstacles_coordinate[j][:2]
                optimal_drones_coordinate[j] = drones_coordinate[j]
            print(f"| episode{i}: {i} | score{j}: {normalized_scores[j] : .2f}")
            print(f"| last state{j}: {observations[j]} | rewards{j}: {rewards[j] : .2f} | actions{j}: {actions[j]} | epsilon{j} : {agents[j].epsilon : .2f} | temp : {temp[j]}")

    spot = Spot(env, GameDisplay)
    dikjstra_path = spot.run_dijkstra()
    print('다익스트라 종료!')
    pygame.display.update()

    print('모든 경로가 존재하는가?', all(c == 0 for c in checking_obstacles))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2x2 서브플롯 생성
    axs = axs.flatten()  # 서브플롯 1차원 배열로 평탄화

    for j in range(num_agent):
        del ddqn_scores[j][0]
        axs[j].plot(ddqn_scores[j])
        axs[j].set_xlabel('episode', fontsize=12)
        axs[j].set_ylabel(f'agent{j + 1} reward', fontsize=12)
        axs[j].set_title(f'agent{j + 1} reward plot', fontsize=14)

    plt.tight_layout()
    plt.savefig('agent_reward_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    pygame.display.flip()
    FramePerSec.tick(60)

    return optimal_paths, optimal_drones_coordinate, optimal_obstacles_coordinate, checking_obstacles, dikjstra_path, env.target_state

# 최적의 경로 시각화 함수
def visualize_optimal_paths(optimal_paths, optimal_drones_coordinate, optimal_obstacles_coordinate, dijkstra_path, target_coordinate):
        pygame.init()
        OptimalPathDisplay = pygame.display.set_mode((300, 300))
        OptimalPathDisplay.fill(WHITE)  # 하얀색으로 배경 채우기
        pygame.display.set_caption("Optimal Paths Visualization")  # 창 이름 설정

        back_img = pygame.image.load('C:/Users/desktop/anaconda3/envs/temp/map.png')
        back_img = pygame.transform.scale(back_img, (300, 300))

        result = [[]]
        for item in dijkstra_path:
            if item == '.':
                result.append([])
            else:
                result[-1].append(item)

        # 마지막에 추가된 빈 리스트 제거
        if result[-1] == []:
            result.pop()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            OptimalPathDisplay.fill(WHITE)
            OptimalPathDisplay.blit(back_img, (0, 0))

            for i in range(len(optimal_paths)):
                pygame.draw.circle(OptimalPathDisplay, drone_color[i], optimal_drones_coordinate[i], 5)
                pygame.draw.circle(OptimalPathDisplay, RED, optimal_obstacles_coordinate[i], 20)

                path = optimal_paths[i][1:]
                # Draw the paths for each drone
                for j in range(len(optimal_paths[i]) - 2):
                    pygame.draw.line(OptimalPathDisplay, drone_color[i], path[j], path[j + 1], 3)

            # Draw the Dijkstra path
            for di_path in result:
                for k in range(len(di_path) - 1):
                    pygame.draw.line(OptimalPathDisplay, GREEN, di_path[k], di_path[k + 1], 3)

            for j in range(len(target_coordinate)):
                pygame.draw.circle(OptimalPathDisplay, GREEN, target_coordinate[j], 15)

            pygame.display.update()

if __name__ == '__main__':
    optimal_paths, optimal_drones_coordinate, optimal_obstacles_coordinate, checking_obstacles, dijkstra_path, target_coordinate = main()

    if all(c == 0 for c in checking_obstacles):
        print('1', optimal_paths[0])
        print('2', optimal_paths[1])
        print('3', optimal_paths[2])
        print('4', optimal_paths[3])
        print('optimal_drones_coordinate', optimal_drones_coordinate)
        print('optimal_obstacles_coordinate', optimal_obstacles_coordinate)
        print('dijkstra_path', dijkstra_path)

        fifth_elements = [[i] for i in range(len(optimal_paths))]
        for idx, path_info in enumerate(optimal_paths):
            path = path_info[1:]  # 서브리스트만 추출

            for i in range(len(path)):
                if i == 0 or (i + 1) % 5 == 0 or i == len(path) - 1:  # 첫 번째 요소 또는 5의 배수 위치에 있는 요소 추가
                    fifth_elements[idx].append(path[i])

        visualize_optimal_paths(fifth_elements, optimal_drones_coordinate, optimal_obstacles_coordinate, dijkstra_path, target_coordinate)

