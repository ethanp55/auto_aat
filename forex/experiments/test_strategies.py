from forex.experiments.metrics_tracker import MetricsTracker
from forex.runner.simulation_runner import SimulationRunner
from forex.strategies.alegaatr import AlegAATr
from forex.strategies.qalegaatr import QAlegAATr
from forex.utils.utils import CURRENCY_PAIRS, TIME_FRAMES, YEARS


def test_strategies() -> None:
    # List of the final results to output
    test_results = []

    # Tracker to keep track of various metrics
    metrics_tracker = MetricsTracker()

    for currency_pair in CURRENCY_PAIRS:
        for time_frame in TIME_FRAMES:
            for year in YEARS[2:]:
                # RUN QALEGAATR (FINE-TUNED AUTO AAT)
                pair_time_frame_year_str = f'{currency_pair}_{time_frame}_{year}'
                print(pair_time_frame_year_str)
                qalegaatr = QAlegAATr()
                result = SimulationRunner.run_simulation(qalegaatr, currency_pair, time_frame, year, False, False,
                                                         metrics_tracker)
                print(result.net_reward)

                # Update the final results
                test_results.append((f'{qalegaatr.name}_{pair_time_frame_year_str}', result))

                print()

                # # RUN ALEGAATR TWICE, ONCE WITH AUTO AAT (NO FINE-TUNING) AND ONCE WITHOUT
                # for auto_aat in [True, False]:
                #     name = 'AlegAAATr' if auto_aat else 'AlegAATr'
                #     alegaatr = AlegAATr(name=name, auto_aat=auto_aat)
                #
                #     result = SimulationRunner.run_simulation(alegaatr, currency_pair, time_frame, year, False, False,
                #                                              metrics_tracker)
                #     print(result.net_reward)
                #
                #     # Update the final results
                #     test_results.append((f'{alegaatr.name}_{pair_time_frame_year_str}', result))
                #
                #     print()

    # Save any metric data in order to perform analysis offline
    # metrics_tracker.save_data(['QAlegAATr', 'AlegAATr', 'AlegAAATr'])
    metrics_tracker.save_data(['QAlegAATr'])

    # Sort the results so that the most profitable results are first
    test_results.sort(key=lambda x: x[1].net_reward, reverse=True)

    # Print the results
    print('\n----------------------------------------------------------')
    print('FINAL TEST RESULTS (ordered from most profitable to least)')
    print('----------------------------------------------------------')

    for name, res in test_results:
        print(name)
        print(res)
        print()

    print('----------------------------------------------------------')
    print('----------------------------------------------------------')
    print('----------------------------------------------------------')


if __name__ == "__main__":
    test_strategies()
