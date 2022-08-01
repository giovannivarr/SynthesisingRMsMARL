import pickle
from datetime import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_testing_steps_two_results(tester_dqprm, tester_strategy, label1='DQPRM_NearyRM', label2='DQPRM_strategyRM'):
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
    plt.plot(steps_dqprm, prc_50_dqprm, color='red', label=label1)
    plt.plot(steps_dqprm, prc_75_dqprm, alpha=0)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_25_dqprm, color='red', alpha=0.25)
    plt.fill_between(steps_dqprm, prc_50_dqprm, prc_75_dqprm, color='red', alpha=0.25)

    plt.plot(steps_strategy, prc_25_strategy, alpha=0)
    plt.plot(steps_strategy, prc_50_strategy, color='green', label=label2)
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


def plot_counterfactual_experiment(tester_baseline, tester_counterfactual, label1, label2):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    prc_25_baseline = list()
    prc_50_baseline = list()
    prc_75_baseline = list()
    prc_25_counterfactual = list()
    prc_50_counterfactual = list()
    prc_75_counterfactual = list()
    steps_baseline = list()
    steps_counterfactual = list()

    # Buffers for plots
    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    if label1[0] == 'o':
        plot_dict_baseline = tester_baseline.results['testing_reward']
        plot_dict_counterfactual = tester_counterfactual.results['testing_reward']
    else:
        plot_dict_baseline = tester_baseline.results['testing_steps']
        plot_dict_counterfactual = tester_counterfactual.results['testing_steps']

    for step in plot_dict_baseline.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_baseline[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_baseline[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_baseline[step]),75))
            current_step.append(sum(plot_dict_baseline[step])/len(plot_dict_baseline[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_baseline[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_baseline[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_baseline[step]),75))
            current_step.append(sum(plot_dict_baseline[step])/len(plot_dict_baseline[step]))

        prc_25_baseline.append(sum(current_25)/len(current_25))
        prc_50_baseline.append(sum(current_50)/len(current_50))
        prc_75_baseline.append(sum(current_75)/len(current_75))
        steps_baseline.append(step)

    current_step = list()
    current_25 = list()
    current_50 = list()
    current_75 = list()

    for step in plot_dict_counterfactual.keys():
        if len(current_step) < 10:
            current_25.append(np.percentile(np.array(plot_dict_counterfactual[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_counterfactual[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_counterfactual[step]),75))
            current_step.append(sum(plot_dict_counterfactual[step])/len(plot_dict_counterfactual[step]))
        else:
            current_step.pop(0)
            current_25.pop(0)
            current_50.pop(0)
            current_75.pop(0)
            current_25.append(np.percentile(np.array(plot_dict_counterfactual[step]),25))
            current_50.append(np.percentile(np.array(plot_dict_counterfactual[step]),50))
            current_75.append(np.percentile(np.array(plot_dict_counterfactual[step]),75))
            current_step.append(sum(plot_dict_counterfactual[step])/len(plot_dict_counterfactual[step]))

        prc_25_counterfactual.append(sum(current_25)/len(current_25))
        prc_50_counterfactual.append(sum(current_50)/len(current_50))
        prc_75_counterfactual.append(sum(current_75)/len(current_75))
        steps_counterfactual.append(step)

    plt.plot(steps_baseline, prc_25_baseline, alpha=0)
    plt.plot(steps_baseline, prc_50_baseline, color='red', label=label1)
    plt.plot(steps_baseline, prc_75_baseline, alpha=0)
    plt.fill_between(steps_baseline, prc_50_baseline, prc_25_baseline, color='red', alpha=0.25)
    plt.fill_between(steps_baseline, prc_50_baseline, prc_75_baseline, color='red', alpha=0.25)

    plt.plot(steps_counterfactual, prc_25_counterfactual, alpha=0)
    plt.plot(steps_counterfactual, prc_50_counterfactual, color='green', label=label2)
    plt.plot(steps_counterfactual, prc_75_counterfactual, alpha=0)
    plt.fill_between(steps_counterfactual, prc_50_counterfactual, prc_25_counterfactual, color='green', alpha=0.25)
    plt.fill_between(steps_counterfactual, prc_50_counterfactual, prc_75_counterfactual, color='green', alpha=0.25)

    plt.grid()
    if label1[0] == 'o':
        plt.ylabel('Reward Obtained', fontsize=15)
    else:
        plt.ylabel('Testing Steps to Task Completion', fontsize=15)
    plt.xlabel('Training Steps', fontsize=15)
    plt.legend()
    if label1[0] != 'o':
        plt.xscale('log')
    #else:
    #    plt.locator_params(axis='x', nbins=5)

    plt.show()


def plot_testing_steps_three_results(tester_dqprm, tester_strategy, tester_ihrl):
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
    #experiment = 'buttons_diff_rm'
    #experiment = 'rendezvous_diff_rm'
    #experiment = 'buttons'
    #experiment = 'rendezvous'

    #experiment = 'buttons_counterfactual' done
    #experiment = 'rendezvous_counterfactual' done
    experiment = 'officeworld_counterfactual'

    #experiment = 'qrm_buttons_counterfactual' done
    #experiment = 'qrm_rendezvous_counterfactual' done
    #experiment = 'qrm_officeworld_counterfactual' done


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

        plot_testing_steps_three_results(tester_dqprm, tester_strategy, tester_ihrl)

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

        plot_testing_steps_three_results(tester_dqprm, tester_strategy, tester_ihrl)

    elif experiment == 'buttons_diff_rm':
        num_agents = 3

        from buttons_config import buttons_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = buttons_config(num_times, num_agents, nonmarkovian=nonmarkovian)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True, nonmarkovian=nonmarkovian)

        tester_strategy = buttons_config(num_times, num_agents, strategy_rm=True, nonmarkovian=nonmarkovian)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True, nonmarkovian=nonmarkovian)

        plot_testing_steps_two_results(tester_dqprm, tester_strategy)

    elif experiment == 'rendezvous_diff_rm':
        num_agents = 10

        from rendezvous_config import rendezvous_config

        from experiments.dqprm import run_multi_agent_experiment
        tester_dqprm = rendezvous_config(num_times, num_agents)  # Get test object from config script
        run_multi_agent_experiment(tester_dqprm, num_agents, num_times, show_print=True)

        tester_strategy = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_multi_agent_experiment(tester_strategy, num_agents, num_times, show_print=True)

        plot_testing_steps_two_results(tester_dqprm, tester_strategy)

    elif experiment == 'buttons_counterfactual':
        num_agents = 3

        from buttons_config import buttons_config
        from experiments.run_strategy_experiment import run_strategy_experiment

        tester_baseline = buttons_config(num_times, num_agents, strategy_rm=True)  # Get test object from config script
        run_strategy_experiment(tester_baseline, num_agents, num_times, show_print=True)

        tester_counterfactual = buttons_config(num_times, num_agents, strategy_rm=True)  # Get test object from config script
        run_strategy_experiment(tester_counterfactual, num_agents, num_times, counterfactual_experiment=True,
                                show_print=True)

        plot_counterfactual_experiment(tester_baseline, tester_counterfactual, 'b_hrm',
                                       'b_counterfactual_hrm ')

    elif experiment == 'rendezvous_counterfactual':
        num_agents = 10

        from rendezvous_config import rendezvous_config
        from experiments.run_strategy_experiment import run_strategy_experiment

        tester_baseline = rendezvous_config(num_times, num_agents, strategy_rm=True)  # Get test object from config script
        run_strategy_experiment(tester_baseline, num_agents, num_times, show_print=True)

        tester_counterfactual = rendezvous_config(num_times, num_agents,
                                                  strategy_rm=True)  # Get test object from config script
        run_strategy_experiment(tester_counterfactual, num_agents, num_times, counterfactual_experiment=True,
                                show_print=True)

        plot_counterfactual_experiment(tester_baseline, tester_counterfactual, 'r_hrm',
                                       'r_countefactual_hrm')

    elif experiment == 'officeworld_counterfactual':
        num_agents = 2

        from officeworld_config import officeworld_config
        from experiments.run_strategy_experiment import run_strategy_experiment

        tester_baseline = officeworld_config(num_times)  # Get test object from config script
        run_strategy_experiment(tester_baseline, num_agents, num_times, show_print=True)

        tester_counterfactual = officeworld_config(num_times)  # Get test object from config script
        run_strategy_experiment(tester_counterfactual, num_agents, num_times, counterfactual_experiment=True,
                                show_print=True)

        plot_counterfactual_experiment(tester_baseline, tester_counterfactual, 'ow_hrm',
                                       'ow_counterfactual_hrm')

    elif experiment == 'qrm_buttons_counterfactual':
        num_agents = 3

        from buttons_config import buttons_config

        from experiments.dqprm import run_multi_agent_experiment

        tester_qrm = buttons_config(num_times, num_agents, strategy_rm=True,
                                      nonmarkovian=nonmarkovian)  # Get test object from config script
        run_multi_agent_experiment(tester_qrm, num_agents, num_times, show_print=True, counterfactual_training=False,
                                   nonmarkovian=nonmarkovian)

        tester_crm = buttons_config(num_times, num_agents, strategy_rm=True, nonmarkovian=nonmarkovian)
        run_multi_agent_experiment(tester_crm, num_agents, num_times, show_print=True, nonmarkovian=nonmarkovian)

        plot_counterfactual_experiment(tester_qrm, tester_crm, 'b_qrm',
                                       'b_crm')

    elif experiment == 'qrm_rendezvous_counterfactual':
        num_agents = 10

        from rendezvous_config import rendezvous_config
        from experiments.dqprm import run_multi_agent_experiment

        tester_qrm = rendezvous_config(num_times, num_agents, strategy_rm=True)  # Get test object from config script
        run_multi_agent_experiment(tester_qrm, num_agents, num_times, show_print=True, counterfactual_training=False)

        tester_crm = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_multi_agent_experiment(tester_crm, num_agents, num_times, show_print=True)

        plot_counterfactual_experiment(tester_qrm, tester_crm, 'r_qrm', 'r_crm')

    elif experiment == 'qrm_officeworld_counterfactual':
        num_agents = 2

        from officeworld_config import officeworld_config
        from experiments.dqprm import run_multi_agent_experiment

        tester_qrm = officeworld_config(num_times)  # Get test object from config script
        run_multi_agent_experiment(tester_qrm, num_agents, num_times, show_print=True, counterfactual_training=False)

        tester_crm = officeworld_config(num_times)  # Get test object from config script
        run_multi_agent_experiment(tester_crm, num_agents, num_times, show_print=True)

        plot_counterfactual_experiment(tester_qrm, tester_crm, 'ow_qrm', 'ow_crm')