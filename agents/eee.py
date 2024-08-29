from simple_rl.agents.AgentClass import Agent
from agents.folk_egal import FolkEgalPunishAgent
import numpy as np
import random
from typing import Dict


class EEE(Agent):
    def __init__(self, name: str, agents: Dict[str, Agent], player: int, log=False, demo=False):
        Agent.__init__(self, name=name, actions=[])

        self.name = name
        self.agents = agents
        self.player = player
        self.log = log
        self.demo = demo
        self.log_message = ''
        self.m_e = {}
        self.n_e = {}
        self.s_e = {}
        self.in_phase = False
        self.phase_counter = 0
        self.phase_rewards = []
        self.n_i = 0
        self.agent_in_use = random.choice(list(self.agents.values()))
        self.explore_prob = np.random.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]) if not self.demo else 0.50

        for agent_name in self.agents.keys():
            self.m_e[agent_name] = 0
            self.n_e[agent_name] = 0
            self.s_e[agent_name] = 0

        self.prev_round = 0
        self.curr_reward = 0

    def reset(self):
        self.in_phase = False
        self.phase_counter = 0
        self.phase_rewards = []
        self.n_i = 0
        self.agent_in_use = random.choice(list(self.agents.values()))
        self.explore_prob = np.random.choice([0.05, 0.06, 0.07, 0.08, 0.09, 0.10]) if not self.demo else 0.50

        for agent_name in self.agents.keys():
            self.m_e[agent_name] = 0
            self.n_e[agent_name] = 0
            self.s_e[agent_name] = 0

        for expert in self.agents.values():
            if isinstance(expert, FolkEgalPunishAgent):
                expert.start_round, expert.should_attack = 0, False

        self.prev_round = 0
        self.curr_reward = 0
        self.log_message = ''

    def act(self, state, reward, round_num):
        if round_num >= self.prev_round:
            self.curr_reward = 0
            self.prev_round = round_num + 1

            if self.log:
                self.log_message = f'{round_num};{self.in_phase};'

            if self.in_phase:
                if self.phase_counter < self.n_i:
                    self.phase_counter += 1

                    if self.log:
                        self.log_message += f'{self.phase_counter};{self.n_i};'

                else:
                    avg_phase_reward = sum(self.phase_rewards) / len(self.phase_rewards)
                    self.n_e[self.agent_in_use.name] += 1
                    self.s_e[self.agent_in_use.name] += self.n_i
                    self.m_e[self.agent_in_use.name] = self.m_e[self.agent_in_use.name] + (self.n_i / self.s_e[self.agent_in_use.name]) * (avg_phase_reward - self.m_e[self.agent_in_use.name])
                    self.phase_rewards = []
                    self.phase_counter = 0
                    self.n_i = 0
                    self.in_phase = False

            if not self.in_phase:
                explore = np.random.choice([0, 1], p=[1 - self.explore_prob, self.explore_prob])

                if self.log:
                    self.log_message += f'{explore};'

                if explore:
                    if self.demo:
                        self.explore_prob /= 2

                    new_agent = random.choice(list(self.agents.values()))

                    if self.agent_in_use.name != new_agent.name and isinstance(new_agent, FolkEgalPunishAgent):
                        new_agent.start_round, new_agent.should_attack = round_num + 1, False

                    self.agent_in_use = new_agent

                else:
                    max_reward = -np.inf

                    for key, val in self.m_e.items():
                        if val > max_reward:
                            max_reward = val

                    agents_to_consider = []

                    for key, val in self.m_e.items():
                        if val == max_reward:
                            agents_to_consider.append(self.agents[key])

                    if self.log:
                        for agent in agents_to_consider:
                            self.log_message += f'{agent.name};'

                    new_agent = random.choice(agents_to_consider)

                    if self.agent_in_use.name != new_agent.name and isinstance(new_agent, FolkEgalPunishAgent):
                        new_agent.start_round, new_agent.should_attack = round_num + 1, False

                    self.agent_in_use = new_agent

                self.n_i = np.random.choice([2, 3, 4, 5]) if not self.demo else np.random.choice([1, 2, 3])

                self.in_phase = True

            if self.log:
                self.log_message += f'{self.agent_in_use.name}'

        self.curr_reward += reward
        self.phase_rewards.append(self.curr_reward)

        return self.agent_in_use.act(state, reward, round_num)