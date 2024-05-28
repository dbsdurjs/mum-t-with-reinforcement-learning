import pygame, sys, random, math
from collections import deque
from tkinter import messagebox, Tk
import numpy as np

queue = deque()
path = []

class Spot():
    def __init__(self, env, GameDisplay):
        self.env = env
        self.uav_state = tuple(self.env.uav_state)  # Ensure tuples for consistency
        self.goals = self.env.target_state  # Ensure tuples for consistency
        self.obstacle_list = [tuple(obstacle[:2]) for obstacle in self.env.obstacle_list]  # Convert to tuples
        self.neighbors = []
        self.visited = []
        self.prev = {}  # Dictionary to store previous nodes
        self.GameDisplay = GameDisplay
        self.max_boundary = int(self.env.observation_space.high[0])
        self.min_boundary = int(self.env.observation_space.low[0])

    def add_neighbors(self, i, j):
        self.neighbors = []  # Reset neighbors list

        if i < self.max_boundary - 1:
            self.neighbors.append((i + 1, j))
        if i > self.min_boundary:
            self.neighbors.append((i - 1, j))
        if j < self.max_boundary - 1:
            self.neighbors.append((i, j + 1))
        if j > self.min_boundary:
            self.neighbors.append((i, j - 1))
        # Add Diagonals
        if i < self.max_boundary - 1 and j < self.max_boundary - 1:
            self.neighbors.append((i + 1, j + 1))
        if i < self.max_boundary - 1 and j > self.min_boundary:
            self.neighbors.append((i + 1, j - 1))
        if i > self.min_boundary and j < self.max_boundary - 1:
            self.neighbors.append((i - 1, j + 1))
        if i > self.min_boundary and j > self.min_boundary:
            self.neighbors.append((i - 1, j - 1))

    # 두 점 간의 유클리드 거리 계산
    def calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return int(distance)

    def dijkstra(self, flag, q_index):
        global queue
        di_path = []

        if len(queue) > 0:
            current = queue.popleft()
            self.add_neighbors(*current)

            if np.array_equal(np.array(current), self.goals[q_index]):
                print('타겟 도착!')
                temp = current
                while temp in self.prev:  # Check if temp is in prev dictionary
                    di_path.append(list(temp))
                    temp = self.prev[temp]  # Get the previous node

                if not flag:
                    if q_index < len(self.goals):
                        q_index += 1
                    print("Done")
                    queue = deque()
                    queue.append(current)
                    self.prev = {}
                    self.visited = []
                    self.visited.append(current)

                    return di_path, q_index  # Reverse the path to get correct order

            if not flag:
                for i in self.neighbors:
                    dist_to_obstacle = [self.calculate_distance(i, obstacle) for obstacle in self.obstacle_list]

                    # 방문하지 않은 노드이고 장애물의 범위가 아니어야함
                    if i not in self.visited and all(dist > self.env.obstacle_radius for dist in dist_to_obstacle):
                        self.visited.append(i)
                        self.prev[i] = current  # Store the previous node
                        queue.append(i)

        return di_path, q_index

    def run_dijkstra(self):
        global path
        flag = False
        queue.append(self.uav_state)
        self.visited.append(self.uav_state)
        q_index = 0

        while True:
            # 목표물 개수보다 적으면 다익스트라 알고리즘 실행
            if q_index < len(self.goals):
                di_path, q_index = self.dijkstra(flag, q_index)
                if len(di_path) > 1:
                    path.extend(di_path)
                    path.append('.')

            cnt = path.count('.')
            if cnt == len(self.goals):
                break
        print(path)
        return path
