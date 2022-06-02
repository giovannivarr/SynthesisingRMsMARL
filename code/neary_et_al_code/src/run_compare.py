import pickle
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

def plot_two_results(tester_dqprm, tester_strategy):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25_dqprm = list()
    prc_50_dqprm = list()
    prc_75_dqprm = list()
    prc_25_strategy = list()
    prc_50_strategy = list()
    prc_75_strategy = list()
    steps_dqprm = list()
    steps_strategy = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    plot_dict_dqprm = tester_dqprm.results['testing_steps']
    plot_dict_strategy = tester_strategy.results['testing_steps']

    for step in plot_dict_dqprm.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))

        prc_25_dqprm.append(sum(current_25)/len(current_25))
        prc_50_dqprm.append(sum(current_50)/len(current_50))
        prc_75_dqprm.append(sum(current_75)/len(current_75))
        steps_dqprm.append(step)

    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    for step in plot_dict_strategy.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))

        prc_25_strategy.append(sum(current_25)/len(current_25))
        prc_50_strategy.append(sum(current_50)/len(current_50))
        prc_75_strategy.append(sum(current_75)/len(current_75))
        steps_strategy.append(step)

    plt.plot(steps_dqprm, prc_25_dqprm, alpha=0)
    plt.plot(steps_dqprm, prc_50_dqprm, color='red', label='DQPRM_NearyRM')
    plt.plot(steps_dqprm, prc_75_dqprm, alpha=0)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_25_dqprm, color='red', alpha=0.25)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_75_dqprm, color='red', alpha=0.25)

    plt.plot(steps_strategy, prc_25_strategy, alpha=0)
    plt.plot(steps_strategy, prc_50_strategy, color='green', label='DQPRM_strategyRM')
    plt.plot(steps_strategy, prc_75_strategy, alpha=0)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_25_strategy, color='green', alpha=0.25)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_75_strategy, color='green', alpha=0.25)

    plt.grid()
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.legend()
    plt.xscale('log')
    #plt.locator_params(axis='x', nbins=5)

    plt.show()


def plot_three_results(tester_dqprm, tester_strategy, tester_ihrl):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25_dqprm = list()
    prc_50_dqprm = list()
    prc_75_dqprm = list()
    prc_25_strategy = list()
    prc_50_strategy = list()
    prc_75_strategy = list()
    prc_25_ihrl = list()
    prc_50_ihrl = list()
    prc_75_ihrl = list()
    steps_dqprm = list()
    steps_strategy = list()
    steps_ihrl = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    plot_dict_dqprm = tester_dqprm.results['testing_steps']
    plot_dict_strategy = tester_strategy.results['testing_steps']
    plot_dict_ihrl = tester_ihrl.results['testing_steps']

    for step in plot_dict_dqprm.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_dqprm[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_dqprm[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_dqprm[step]),75))
            current_step.append(sum(plot_dict_dqprm[step])/len(plot_dict_dqprm[step]))

        prc_25_dqprm.append(sum(current_25)/len(current_25))
        prc_50_dqprm.append(sum(current_50)/len(current_50))
        prc_75_dqprm.append(sum(current_75)/len(current_75))
        steps_dqprm.append(step)

    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    for step in plot_dict_strategy.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_strategy[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_strategy[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_strategy[step]),75))
            current_step.append(sum(plot_dict_strategy[step])/len(plot_dict_strategy[step]))

        prc_25_strategy.append(sum(current_25)/len(current_25))
        prc_50_strategy.append(sum(current_50)/len(current_50))
        prc_75_strategy.append(sum(current_75)/len(current_75))
        steps_strategy.append(step)

    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    for step in plot_dict_ihrl.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_ihrl[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict_ihrl[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict_ihrl[step]), 75))
            current_step.append(sum(plot_dict_ihrl[step]) / len(plot_dict_ihrl[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_ihrl[step]), 25))
            current_50.append(np.percentile(np.array(plot_dict_ihrl[step]), 50))
            current_75.append(np.percentile(np.array(plot_dict_ihrl[step]), 75))
            current_step.append(sum(plot_dict_ihrl[step]) / len(plot_dict_ihrl[step]))

        prc_25_ihrl.append(sum(current_25) / len(current_25))
        prc_50_ihrl.append(sum(current_50) / len(current_50))
        prc_75_ihrl.append(sum(current_75) / len(current_75))
        steps_ihrl.append(step)

    plt.plot(steps_dqprm, prc_25_dqprm, alpha=0)
    plt.plot(steps_dqprm, prc_50_dqprm, color='red', label='DQPRM')
    plt.plot(steps_dqprm, prc_75_dqprm, alpha=0)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_25_dqprm, color='red', alpha=0.25)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_75_dqprm, color='red', alpha=0.25)

    plt.plot(steps_strategy, prc_25_strategy, alpha=0)
    plt.plot(steps_strategy, prc_50_strategy, color='green', label='h-SeqSIL')
    plt.plot(steps_strategy, prc_75_strategy, alpha=0)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_25_strategy, color='green', alpha=0.25)
    plt.fill_between(steps_strategy, prc_50_strategy, prc_75_strategy, color='green', alpha=0.25)

    plt.plot(steps_ihrl, prc_25_ihrl, alpha=0)
    plt.plot(steps_ihrl, prc_50_ihrl, color='blue', label='h-IL')
    plt.plot(steps_ihrl, prc_75_ihrl, alpha=0)
    plt.fill_between(steps_ihrl, prc_50_ihrl, prc_25_ihrl, color='blue', alpha=0.25)
    plt.fill_between(steps_ihrl, prc_50_ihrl, prc_75_ihrl, color='blue', alpha=0.25)

    plt.grid()
    plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.legend()
    plt.xscale('log')
    #plt.locator_params(axis='x', nbins=5)

    plt.show()


if __name__ == "__main__":

    num_times = 10 # Number of separate trials to run the algorithm for
    experiment = 'buttons_diff_rm'
    #experiment = 'rendezvous_diff_rm'
    #experiment = 'buttons'
    #experiment = 'rendezvous'

    nonmarkovian = True
    #nonmarkovian = False

    if experiment == 'buttons':
        num_agents = 3

        from buttons_config import buttons_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = buttons_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        from experiments.run_strategy_experiment import run_strategy_experiment
        tester_strategy = buttons_config(num_times, num_agents, strategy_rm=True)
        run_strategy_experiment(tester_strategy, num_agents, num_times, show_print=True)

        from experiments.run_ihrl_experiment import run_ihrl_experiment
        tester_ihrl = buttons_config(num_times, num_agents)
        run_ihrl_experiment(tester_ihrl, num_agents, num_times, show_print=True)

        plot_three_results(tester_dqprm, tester_strategy, tester_ihrl)

    elif experiment == 'rendezvous':
        num_agents = 10

        from rendezvous_config import rendezvous_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = rendezvous_config(num_times, num_agents)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        from experiments.run_strategy_experiment import run_strategy_experiment
        tester_strategy = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_strategy_experiment(tester_strategy, num_agents, num_times, show_print=True)

        from experiments.run_ihrl_experiment import run_ihrl_experiment
        tester_ihrl = rendezvous_config(num_times, num_agents)
        run_ihrl_experiment(tester_ihrl, num_agents, num_times, show_print=True)

        plot_three_results(tester_dqprm, tester_strategy, tester_ihrl)

    elif experiment == 'buttons_diff_rm':
        num_agents = 3

        from buttons_config import buttons_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = buttons_config(num_times, num_agents, nonmarkovian=nonmarkovian)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True, nonmarkovian=nonmarkovian)

        tester_strategy = buttons_config(num_times, num_agents, strategy_rm=True, nonmarkovian=nonmarkovian)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True, nonmarkovian=nonmarkovian)

        plot_two_results(tester_dqprm, tester_strategy)

    elif experiment == 'rendezvous_diff_rm':
        num_agents = 10

        from rendezvous_config import rendezvous_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = rendezvous_config(num_times, num_agents)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        tester_strategy = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True)

        plot_two_results(tester_dqprm, tester_strategy)