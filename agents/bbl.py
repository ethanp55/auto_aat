from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from agents.alegaatr import AlegAATr
from agents.folk_egal import FolkEgalPunishAgent
from utils.utils import P1
import numpy as np
from typing import List
from copy import deepcopy
import random


class BBL(Agent):
    def __init__(self, name: str, game: MarkovGameMDP, player: int, n_samples=10, log=False):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.game = deepcopy(game)
        self.player = player
        self.n_samples = n_samples
        self.log = log
        self.log_message = ''
        self.experts = AlegAATr.create_aat_experts(self.game, self.player)
        self.reset()

    def reset(self):
        for expert in self.experts.values():
            if isinstance(expert, FolkEgalPunishAgent):
                expert.start_round, expert.should_attack = 0, False

        self.probs_prod = [1] * len(self.experts)
        self.agent_probs = [1] * len(self.experts)
        self.priors = np.array([1 / len(self.experts)] * len(self.experts))
        self.next_round_num = 1
        self.expert_to_use = random.choice(list(self.experts.values()))
        self.prev_reward = 0
        self.log_message = ''

    def _record_opp_actions_demo(self, round_num, opp_reward):
        agents = list(self.experts.values())

        for i in range(len(agents)):
            agent = deepcopy(agents[i])

            prev_player = agent.player
            agent.player = 1 - self.player
            n_matches = 0

            for _ in range(self.n_samples):
                self.game.reset()
                curr_state = deepcopy(self.game.get_init_state())
                action_map = dict()
                opp_actions = []
                curr_reward = 0
                opp_name = '2' if self.player == P1 else '1'

                while not curr_state.is_terminal():
                    action_p1, action_p2 = self.expert_to_use.act(curr_state, 0, round_num)
                    opp_action_p1, opp_action_p2 = agent.act(curr_state, 0, round_num)
                    opp_action = opp_action_p1 if self.player != P1 else opp_action_p2

                    if self.player == P1:
                        action_map['1'], action_map['2'] = action_p1, opp_action_p2

                    else:
                        action_map['1'], action_map['2'] = opp_action_p1, action_p2

                    if curr_state.turn is None or curr_state.turn != self.player:
                        opp_actions.append(opp_action)

                    updated_rewards_map, next_curr_state = self.game.execute_agent_action(action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        if agent_name == opp_name:
                            curr_reward += new_reward

                    curr_state = next_curr_state

                n_matches += 1 if curr_reward == opp_reward else 0

            n_matches = max(n_matches, 1)
            self.agent_probs[i] = n_matches / self.n_samples
            self.probs_prod[i] *= self.agent_probs[i]
            agent.player = prev_player

    def record_opp_actions(self, round_num, actions: List):
        agents = list(self.experts.values())

        for i in range(len(agents)):
            agent = deepcopy(agents[i])

            prev_player = agent.player
            agent.player = 1 - self.player
            n_matches = 0

            for _ in range(self.n_samples):
                self.game.reset()
                curr_state = deepcopy(self.game.get_init_state())
                action_map = dict()
                opp_actions = []

                while not curr_state.is_terminal():
                    action_p1, action_p2 = self.expert_to_use.act(curr_state, 0, round_num)
                    opp_action_p1, opp_action_p2 = agent.act(curr_state, 0, round_num)
                    opp_action = opp_action_p1 if self.player != P1 else opp_action_p2

                    if self.player == P1:
                        action_map['1'], action_map['2'] = action_p1, opp_action_p2

                    else:
                        action_map['1'], action_map['2'] = opp_action_p1, action_p2

                    if curr_state.turn is None or curr_state.turn != self.player:
                        opp_actions.append(opp_action)

                    updated_rewards_map, next_curr_state = self.game.execute_agent_action(action_map)

                    curr_state = next_curr_state

                n_matches += 1 if opp_actions == actions else 0

            self.agent_probs[i] = n_matches / self.n_samples
            self.probs_prod[i] *= self.agent_probs[i]
            agent.player = prev_player

    def _update_expert(self, round_num):
        def _expected_val(agent, h=1):
            opp_agents = list(self.experts.values())
            expected_reward = 0

            for i in range(len(opp_agents)):
                opp_agent = deepcopy(opp_agents[i])
                expected_reward += self.priors[i] * self.agent_probs[i] * _plan_ahead(agent, opp_agent, h - 1)

            return expected_reward

        def _plan_ahead(agent, opp_agent, h, samples=3):
            agent.player = self.player

            if isinstance(opp_agent, FolkEgalPunishAgent):
                opp_agent.change_player(1 - self.player)

            else:
                opp_agent.player = 1 - self.player

            if agent.name == opp_agent.name:
                opp_agent.name += '1'

            rewards = []

            for _ in range(samples):
                self.game.reset()
                curr_state = deepcopy(self.game.get_init_state())
                action_map = dict()
                reward = 0

                while not curr_state.is_terminal():
                    action_p1, action_p2 = agent.act(curr_state, 0, round_num)
                    opp_action_p1, opp_action_p2 = opp_agent.act(curr_state, 0, round_num)

                    if self.player == P1:
                        action_map['1'], action_map['2'] = action_p1, opp_action_p2

                    else:
                        action_map['1'], action_map['2'] = opp_action_p1, action_p2

                    updated_rewards_map, next_curr_state = self.game.execute_agent_action(action_map)

                    name_to_use = '1' if self.player == P1 else '2'

                    for agent_name, new_reward in updated_rewards_map.items():
                        if agent_name == name_to_use:
                            reward += new_reward

                    curr_state = next_curr_state

                rewards.append(reward)

            reward_avg = np.array(rewards).mean()

            pred = reward_avg if h == 0 else reward_avg + _expected_val(agent, h)

            return pred

        agents = self.experts.values()

        best_agent_name, best_reward = None, -np.inf

        if self.log:
            self.log_message = f'{round_num};{self.priors};{self.agent_probs};'

        for agent in agents:
            expected_reward = _expected_val(deepcopy(agent))

            if self.log:
                self.log_message += f'{expected_reward};'

            if expected_reward > best_reward:
                best_agent_name, best_reward = agent.name, expected_reward

        if self.log:
            self.log_message += f'{best_agent_name}'

        self.expert_to_use = self.experts[best_agent_name]

    def act(self, state, reward, round_num):
        self.prev_reward = reward

        if round_num >= self.next_round_num:
            agents = list(self.experts.values())
            posteriors = []

            for i in range(len(agents)):
                posterior = self.priors[i] * self.probs_prod[i]
                posterior = 0.000001 if posterior == 0 else posterior
                posteriors.append(posterior)
                self.priors[i] = posterior

            if self.priors.min() == self.priors.max():
                self.priors = np.array([1 / len(self.experts)] * len(self.experts))

            else:
                self.priors = (self.priors - self.priors.min()) / (self.priors.max() - self.priors.min())
                self.priors = self.priors / self.priors.sum()

            assert round(self.priors.sum()) == 1

            self._update_expert(round_num)
            self.next_round_num = round_num + 1

        return self.expert_to_use.act(state, reward, round_num)