import numpy as np

"""class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state, action, reward, state_, done, number_agent):
        if number_agent == 0:
            index = int(self.mem_cntr % self.mem_size)
        elif number_agent == 1:
            index = int((self.mem_cntr % self.mem_size) + (self.mem_size // 3))
        elif number_agent == 2:
            index = int((self.mem_cntr % self.mem_size) + (self.mem_size // 1.5))
        else:
            print('buffer error : over the number of agents')
            return
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size, num_agent):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones"""
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, num_agents):
        self.mem_size = max_size
        self.mem_cntr = np.zeros(num_agents, dtype=np.int32)
        self.num_agents = num_agents

        self.state_memory = np.zeros((num_agents, self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((num_agents, self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((num_agents, self.mem_size), dtype=np.int32)
        self.reward_memory = np.zeros((num_agents, self.mem_size), dtype=np.float32)
        self.terminal_memory = np.zeros((num_agents, self.mem_size), dtype=np.bool8)

    def store_transition(self, state, action, reward, state_, done, number_agent):
        if number_agent >= self.num_agents:
            print('buffer error : over the number of agents')
            return

        index = self.mem_cntr[number_agent] % self.mem_size
        self.state_memory[number_agent, index] = state
        self.new_state_memory[number_agent, index] = state_
        self.action_memory[number_agent, index] = action
        self.reward_memory[number_agent, index] = reward
        self.terminal_memory[number_agent, index] = done

        self.mem_cntr[number_agent] += 1

    def sample_buffer(self, batch_size, num_agent):
        if num_agent >= self.num_agents:
            print('buffer error : invalid agent number')
            return

        max_mem = min(self.mem_cntr[num_agent], self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[num_agent, batch]
        new_states = self.new_state_memory[num_agent, batch]
        actions = self.action_memory[num_agent, batch]
        rewards = self.reward_memory[num_agent, batch]
        dones = self.terminal_memory[num_agent, batch]

        return states, actions, rewards, new_states, dones
