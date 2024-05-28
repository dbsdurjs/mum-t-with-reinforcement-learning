import random
from gym import Env, spaces
from gym.spaces import Box
import numpy as np
import math
import copy

class ENV(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([299, 299]), dtype=int)
        self.target_state = np.array([[80, 100], [270, 30], [260, 260]])
        self.uav_state = np.array([5, 5])  # uav position
        self.agent_pos = [[random.randrange(25, 30), random.randrange(25, 30)],  # drone1
                          [random.randrange(55, 60), random.randrange(25, 30)],  # drone2
                          [random.randrange(25, 30), random.randrange(55, 60)],  # drone3
                          [random.randrange(55, 60), random.randrange(55, 60)]]  # drone4
        self.state = np.array(self.agent_pos)
        self.prev_state = self.state
        self.obstacle_radius = 20
        self.obstacle_list = [
            [180, 65, self.obstacle_radius],  # end target1
            [55, 180, self.obstacle_radius],  # end target2
            [200, 200, self.obstacle_radius],  # end target3
            [random.randrange(120, 140), random.randrange(120, 140), self.obstacle_radius]]  # end target4
        self.reward_list = np.zeros((len(self.agent_pos),))
        self.epi_length = [150 for _ in range(len(self.agent_pos))]
        self.checking_obstacle = 0
        self.path = [[i] for i in range(len(self.agent_pos))]
        self.target_coordinate = [[0, 0] for _ in range(len(self.agent_pos))]
        self.optimal_obstacle_coordinate = [[0, 0, _] for _ in range(len(self.agent_pos))]
        self.optimal_drone_coordinate = [[0, 0] for _ in range(len(self.agent_pos))]

    def calculate_distance(self, point1, point2):
        # 두 점 간의 유클리드 거리 계산
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def step(self, action, number_agent, dones):
        self.state = self.agent_pos[number_agent]
        self.end_target = self.obstacle_list[number_agent][:2]
        self.max_reward = 100
        self.min_reward = -50
        checking = 0
        self.action = action
        self.epi_length[number_agent] -= 1

        if dones:
            print(dones, str(number_agent + 1) + ' 이미 도달하였습니다.')
            self.reward_list[number_agent] += self.max_reward
            checking = 1
            return self.target_coordinate[number_agent], self.reward_list[number_agent], True, self.checking_obstacle, self.path, self.optimal_drone_coordinate, self.optimal_obstacle_coordinate, checking
        else:
            self.path[number_agent].append(copy.copy(self.state))

        if self.action == 0:  # 위로 이동 시
            self.state[0] = np.clip(self.state[0], 0, 299)
            self.state[1] = np.clip(self.state[1] - 10, 0, 299)
        elif self.action == 1:  # 아래로 이동 시
            self.state[0] = np.clip(self.state[0], 0, 299)
            self.state[1] = np.clip(self.state[1] + 10, 0, 299)
        elif self.action == 2:  # 왼쪽으로 이동 시
            self.state[0] = np.clip(self.state[0] - 10, 0, 299)
            self.state[1] = np.clip(self.state[1], 0, 299)
        elif self.action == 3:  # 오른쪽로 이동 시
            self.state[0] = np.clip(self.state[0] + 10, 0, 299)
            self.state[1] = np.clip(self.state[1], 0, 299)

        # wall collision penalty
        if (self.state[0] < 5 or self.state[0] > 295) or (self.state[1] < 5 or self.state[1] > 295):
            print(str(number_agent + 1) + 'out of bound')
            self.reward_list[number_agent] += self.min_reward
            self.checking_obstacle += 1
        else:
            dist_to_obstacle_target = self.calculate_distance(self.state, self.end_target)

            # 각 드론이 타겟에 명중 시 보상
            if dist_to_obstacle_target < 20:
                self.reward_list[number_agent] = self.max_reward
                self.optimal_obstacle_coordinate = self.obstacle_list
                self.optimal_drone_coordinate = self.agent_pos
                print(str(number_agent + 1) + ' 번의 타겟 명중!')
                done = True
                self.target_coordinate[number_agent] = self.state
                checking = 1
                return self.target_coordinate[number_agent], self.reward_list[number_agent], done, self.checking_obstacle, self.path, self.optimal_drone_coordinate, self.optimal_obstacle_coordinate, checking

            # 목표물에 가까워진다면 보상값 추가
            elif dist_to_obstacle_target >= 10 and dist_to_obstacle_target < 70:
                print(str(number_agent + 1) + ' near by the target')
                self.reward_list[number_agent] += 5

            # 다른 타겟에 명중 시 보상 감소
            for obstacle in self.obstacle_list:
                if obstacle[:2] == self.end_target:
                    continue
                obstacle_center = obstacle[:2]
                distance_to_obstacle = self.calculate_distance(self.state, obstacle_center)

                if distance_to_obstacle < self.obstacle_radius:
                    print(str(number_agent + 1) + '장애물 충돌!')
                    self.reward_list[number_agent] += self.min_reward
                    self.checking_obstacle += 1

        # Checking if episode is done
        if self.epi_length[number_agent] <= 0:
            print(str(number_agent + 1) + ' end the episode step')
            self.reward_list[number_agent] -= 1
            done = True
        else:
            done = False

        # 매 step penalty
        self.reward_list[number_agent] -= 3

        return self.get_obs(number_agent), self.reward_list[number_agent], done, self.checking_obstacle, self.path, self.optimal_drone_coordinate, self.optimal_obstacle_coordinate, checking
    def reset(self, number_agent):
        self.action_space = spaces.Discrete(4)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([299, 299]), dtype=int)
        self.target_state = np.array([[80, 100], [270, 30], [260, 260]])
        self.uav_state = np.array([5, 5])  # uav position
        self.agent_pos = [[random.randrange(25, 30), random.randrange(25, 30)],  # drone1
                          [random.randrange(55, 60), random.randrange(25, 30)],  # drone2
                          [random.randrange(25, 30), random.randrange(55, 60)],  # drone3
                          [random.randrange(55, 60), random.randrange(55, 60)]]  # drone4
        self.state = np.array(self.agent_pos)
        self.prev_state = self.state
        self.obstacle_radius = 20
        self.obstacle_list = [
            [180, 65, self.obstacle_radius],  # end target1
            [55, 180, self.obstacle_radius],  # end target2
            [200, 200, self.obstacle_radius],  # end target3
            [random.randrange(120, 140), random.randrange(120, 140), self.obstacle_radius]]  # end target4
        self.reward_list = np.zeros((len(self.agent_pos),))
        self.epi_length = [150 for _ in range(len(self.agent_pos))]
        self.checking_obstacle = 0
        self.path = [[i] for i in range(len(self.agent_pos))]
        self.target_coordinate = [[0, 0] for _ in range(len(self.agent_pos))]
        self.optimal_obstacle_coordinate = [[0, 0, _] for _ in range(len(self.agent_pos))]
        self.optimal_drone_coordinate = [[0, 0] for _ in range(len(self.agent_pos))]

        return self.get_obs(number_agent, 1)
    def get_obs(self, number_agent, z=0):
        if z == 1:
            return np.array(self.state[number_agent], dtype=int)
        return np.array(self.state, dtype=int)

