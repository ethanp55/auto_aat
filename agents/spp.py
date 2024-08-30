from simple_rl.agents.AgentClass import Agent
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP
from agents.alegaatr import AlegAATr
from agents.folk_egal import FolkEgalPunishAgent
from utils.utils import P1
import numpy as np
from copy import deepcopy
import random


class SPP(Agent):
    def __init__(self, name: str, game: MarkovGameMDP, player: int, lmbda=0.99, n_simulations=10, log=False):
        Agent.__init__(self, name=name, actions=[])
        self.name = name
        self.game = deepcopy(game)
        self.player = player
        self.lmbda = lmbda
        self.n_simulations = n_simulations
        self.log = log
        self.log_message = ''
        self.rewards = []
        self.experts = AlegAATr.create_aat_experts(self.game, self.player)
        self.expert_names = list(self.experts.keys())
        self.expert_agents = list(self.experts.values())
        self.minimax_val = self.experts['AlgaaterMinimax'].v_p1[str(self.game.get_init_state())] if self.player == P1 \
            else self.experts['AlgaaterMinimax'].v_p2[str(self.game.get_init_state())]
        self.expert_to_use = random.choice(self.expert_names)
        if self.log:
            self.log_message = f'{self.expert_to_use}'
        self.prev_change_round = 0
        self.our_actions = set()
        self.opp_actions = set()
        self.should_update_expert = False
        self.aspiration = self._get_potential()[0]
        self.last_actions, self.opp_last_actions = None, None

    def _run_simulation(self, agent1, agent2, n_rounds):
        potential_rewards = []

        for _ in range(self.n_simulations):
            round_rewards = []

            prev_reward_1 = 0
            prev_reward_2 = 0

            for r_num in range(n_rounds):
                self.game.reset()
                curr_state = deepcopy(self.game.get_init_state())
                action_map = dict()
                reward1 = 0
                reward2 = 0
                agent_map = {agent1.name: agent1, agent2.name: agent2} if agent1.player == P1 \
                    else {agent2.name: agent2, agent1.name: agent1}

                while not curr_state.is_terminal():
                    for agent_name, agent in agent_map.items():
                        agent_reward = prev_reward_1 if agent.player == P1 else prev_reward_2
                        action_p1, action_p2 = agent.act(curr_state, agent_reward, r_num)
                        action_map[agent_name] = action_p1 if agent.player == P1 else action_p2

                    updated_rewards_map, next_curr_state = self.game.execute_agent_action(action_map)

                    for agent_name, new_reward in updated_rewards_map.items():
                        if agent_map[agent_name].player == P1:
                            reward1 += new_reward

                        else:
                            reward2 += new_reward

                    curr_state = next_curr_state

                prev_reward_1 = reward1
                prev_reward_2 = reward2

                if self.player == P1:
                    round_rewards.append(reward1)

                else:
                    round_rewards.append(reward2)

            potential_rewards.append(np.array(round_rewards).mean())

        return np.array(potential_rewards).mean()

    def _get_potential(self, n_rounds=10):
        potentials = []

        for expert in self.expert_agents:
            if expert.name == 'AlgaaterCoop':
                agent1 = deepcopy(expert)
                agent2 = deepcopy(self.experts['AlgaaterCoop'])
                agent2.player, agent2.name = 1 - self.player, 'AlgaaterCoop1'

                potential = self._run_simulation(agent1, agent2, n_rounds)

                potentials.append(potential)

            elif expert.name == 'AlgaaterCoopPunish':
                agent1 = deepcopy(self.experts['AlgaaterCoop'])
                agent2 = deepcopy(self.experts['AlgaaterCoop'])
                agent2.player, agent2.name = 1 - self.player, 'AlgaaterCoop1'

                potentials.append(self._run_simulation(agent1, agent2, n_rounds))

            elif expert.name == 'AlgaaterBully':
                agent1 = deepcopy(expert)
                agent2 = deepcopy(self.experts['AlgaaterBully'])
                agent2.player, agent2.name = 1 - self.player, 'AlgaaterBully1'

                potential = self._run_simulation(agent1, agent2, n_rounds)

                potentials.append(potential)

            elif expert.name == 'AlgaaterBullyPunish':
                agent1 = deepcopy(self.experts['AlgaaterBully'])
                agent2 = deepcopy(self.experts['AlgaaterBullied'])
                agent2.player = 1 - self.player

                potentials.append(self._run_simulation(agent1, agent2, n_rounds))

            elif expert.name == 'AlgaaterBullied':
                agent1 = deepcopy(expert)
                agent2 = deepcopy(self.experts['AlgaaterBully'])
                agent2.player = 1 - self.player

                potential = self._run_simulation(agent1, agent2, n_rounds)

                potentials.append(potential)

            elif expert.name == 'AlgaaterMinimax':
                potentials.append(self.minimax_val)

            elif expert.name == 'AlgaaterCfr':
                potentials.append(0)

            else:
                raise Exception('Invalid expert name')

        return potentials

    def update_actions_and_rewards(self, our_actions, opp_actions, our_reward):
        our_actions_tup, opp_actions_tup = tuple(sorted(our_actions)), tuple(sorted(opp_actions))
        self.last_actions, self.opp_last_actions = our_actions_tup, opp_actions_tup

        if our_actions_tup in self.our_actions and opp_actions_tup in self.opp_actions:
            self.should_update_expert = True

        else:
            self.our_actions.add(our_actions_tup)
            self.opp_actions.add(opp_actions_tup)

        self.rewards.append(our_reward)

    def act(self, state, reward, round_num):
        if self.should_update_expert and round_num > self.prev_change_round:
            row = len(self.rewards)
            r_bar_t = np.array(self.rewards).mean()

            self.aspiration = (self.lmbda ** row) * self.aspiration + (1 - (self.lmbda ** row)) * r_bar_t

            potentials = self._get_potential()
            filtered_expert_set = []

            for i in range(len(potentials)):
                if potentials[i] >= self.aspiration:
                    filtered_expert_set.append(self.expert_names[i])

            if len(filtered_expert_set) == 0:
                filtered_expert_set.append('AlgaaterCfr')

            f_prob = min(1, (r_bar_t / self.aspiration) ** row)

            use_curr_expert = random.choices([1, 0], weights=[f_prob, 1 - f_prob])[0]

            old_expert = self.expert_to_use

            if not use_curr_expert:
                self.expert_to_use = random.choice(filtered_expert_set)

            self.should_update_expert = False

            self.our_actions.clear()
            self.opp_actions.clear()
            self.our_actions.add(self.last_actions)
            self.opp_actions.add(self.opp_last_actions)

            self.prev_change_round = round_num
            self.rewards = []

            if old_expert != self.expert_to_use and isinstance(self.experts[self.expert_to_use], FolkEgalPunishAgent):
                self.experts[self.expert_to_use].start_round, self.experts[self.expert_to_use].should_attack = \
                    round_num + 1, False

            if self.log:
                self.log_message = f'{round_num};{self.aspiration};{row};{r_bar_t};{f_prob};{use_curr_expert};{filtered_expert_set};{self.expert_to_use}'

        return self.experts[self.expert_to_use].act(state, reward, round_num)

    def reset(self):
        self.rewards = []
        for expert in self.experts.values():
            if isinstance(expert, FolkEgalPunishAgent):
                expert.start_round, expert.should_attack = 0, False

        self.expert_to_use = random.choice(self.expert_names)
        self.prev_change_round = 0
        self.our_actions = set()
        self.opp_actions = set()
        self.should_update_expert = False
        self.aspiration = self._get_potential()[0]
        self.last_actions, self.opp_last_actions = None, None
        self.log_message = ''
        if self.log:
            self.log_message = f'{self.expert_to_use}'