from stag_hare.agents.alegaatr import AlegAATr
from stag_hare.agents.generator import GreedyHareGen, GreedyPlannerHareGen, GreedyPlannerStagGen, TeamAwareGen
from stag_hare.agents.greedy import Greedy
from stag_hare.agents.greedy_prob import GreedyProbabilistic
from stag_hare.agents.modeller import Modeller
from stag_hare.agents.prob_dest import ProbabilisticDestinations
from stag_hare.agents.smalegaatr import SMAlegAATr
from copy import deepcopy
from stag_hare.environment.runner import run
import os
from utils.utils import N_HUNTERS, STAG_NAME


N_EPOCHS = 5
GRID_SIZES = [(9, 9), (12, 12), (15, 15)]
n_training_iterations = N_EPOCHS * len(GRID_SIZES)
progress_percentage_chunk = int(0.05 * n_training_iterations)
curr_iteration = 0
print(n_training_iterations, progress_percentage_chunk)
n_other_hunters = N_HUNTERS - 1

# names = ['AlegAATr', 'AlegAAATr', 'AlegAAATTr', 'SMAlegAATr']
names = ['SMAlegAATr']

# Reset any existing simulation files (opening a file in write mode will truncate it)
for file in os.listdir('../../analysis/stag_hare_results/'):
    name = file.split('_')[0]
    if name in names:
        with open(f'../../analysis/stag_hare_results/{file}', 'w', newline='') as _:
            pass

# Run the training process
for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch + 1}')

    for height, width in GRID_SIZES:
        print(height, width)
        if curr_iteration != 0 and progress_percentage_chunk != 0 and curr_iteration % progress_percentage_chunk == 0:
            print(f'{100 * (curr_iteration / n_training_iterations)}%')

        list_of_other_hunters = []
        list_of_other_hunters.append(([GreedyHareGen(f'GreedyHareGen{i}') for i in range(n_other_hunters)], 'greedyhare'))
        list_of_other_hunters.append(([GreedyPlannerHareGen(f'GreedyPlannerHareGen{i}') for i in range(n_other_hunters)], 'greedyplannerhare'))
        list_of_other_hunters.append(([GreedyPlannerStagGen(f'GreedyPlannerStagGen{i}') for i in range(n_other_hunters)], 'greedyplannerstag'))
        list_of_other_hunters.append(([TeamAwareGen(f'TeamAwareGen{i}') for i in range(n_other_hunters)], 'teamawarestag'))
        list_of_other_hunters.append(([Greedy(f'Greedy{i}', STAG_NAME) for i in range(n_other_hunters)], 'greedystag'))
        list_of_other_hunters.append(([Modeller(f'Modeller{i}') for i in range(n_other_hunters)], 'modellerstag'))
        list_of_other_hunters.append(([GreedyProbabilistic(f'GreedyProbabilistic{i}') for i in range(n_other_hunters)], 'greedyprobhare'))
        list_of_other_hunters.append(([ProbabilisticDestinations(f'ProbabilisticDestinations{i}') for i in range(n_other_hunters)], 'probdesthare'))
        list_of_other_hunters.append((None, 'selfplay'))

        for other_hunters, label in list_of_other_hunters:
            agents_to_test = []
            # agents_to_test.append(AlegAATr(lmbda=0.0, enhanced=True))
            # agents_to_test.append(AlegAATr(name='AlegAAATr', lmbda=0.0, enhanced=True, auto_aat=True))
            # agents_to_test.append(AlegAATr(name='AlegAAATTr', lmbda=0.0, enhanced=True, auto_aat_tuned=True))
            agents_to_test.append(SMAlegAATr(enhanced=True))

            self_play_agents = []
            # self_play_agents.append([AlegAATr(f'AlegAATr{i}', lmbda=0.0, enhanced=True) for i in range(n_other_hunters)])
            # self_play_agents.append([AlegAATr(f'AlegAAATr{i}', lmbda=0.0, enhanced=True, auto_aat=True) for i in range(n_other_hunters)])
            # self_play_agents.append([AlegAATr(f'AlegAAATTr{i}', lmbda=0.0, enhanced=True, auto_aat_tuned=True) for i in range(n_other_hunters)])
            self_play_agents.append([SMAlegAATr(f'SMAlegAATr{i}', enhanced=True) for i in range(n_other_hunters)])

            for i, agent_to_test in enumerate(agents_to_test):
                if label == 'selfplay':
                    assert other_hunters is None
                    assert len(self_play_agents[i]) == n_other_hunters
                    hunters = deepcopy(self_play_agents[i])
                    for hunter in hunters:
                        assert isinstance(hunter, type(agent_to_test))
                else:
                    assert len(other_hunters) == n_other_hunters
                    hunters = deepcopy(other_hunters)
                hunters.append(agent_to_test)
                assert len(hunters) == N_HUNTERS
                sim_label = f'{agent_to_test.name}_{label}_h={height}_w={width}'
                run(hunters, height, width, results_file=f'../../analysis/stag_hare_results/{sim_label}.csv',
                    generator_file=f'../../analysis/stag_hare_generator_usage/{sim_label}.csv',
                    vector_file=f'../../analysis/stag_hare_vectors/{sim_label}.csv')

        curr_iteration += 1
