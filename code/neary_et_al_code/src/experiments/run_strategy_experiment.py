import numpy as np
import random, time

from tester.tester import Tester
from datetime import datetime
from Agent.strategy_agent import StrategyAgent
from Environments.coop_buttons.buttons_env import ButtonsEnv
from Environments.coop_buttons.multi_agent_buttons_env import MultiAgentButtonsEnv
from Environments.rendezvous.gridworld_env import GridWorldEnv
from Environments.rendezvous.multi_agent_gridworld_env import MultiAgentGridWorldEnv
from Environments.officeworld.officeworld_env import OfficeWorldEnv
from Environments.officeworld.multi_agent_officeworld_env import MultiAgentOfficeWorldEnv
import matplotlib.pyplot as plt


def get_next_option(agent, task, single_option_training=False, invariant_experiment=False):
    if single_option_training:
        return agent.options_list[0]

    if invariant_experiment and agent.agent_id == 3:
        return 'recharge'

    if task == 'buttons':
        if agent.agent_id == 0:
            if agent.current_option == 'by':
                option = 'g'
            else:
                option = 'by'
        elif agent.agent_id == 1:
            if agent.current_option == 'bg':
                option = 'a2br'
                # option = 'br'
            else:
                option = 'bg'
        else:
            option = 'a3br'
            # option = 'br'
    elif task == 'rendezvous':
        if agent.current_option == 'r{}'.format(agent.agent_id):
            option = 'g{}'.format(agent.agent_id)
        else:
            option = 'r{}'.format(agent.agent_id)
    elif task == 'officeworld':
        if agent.agent_id == 1:
            if agent.current_option == 'c':
                option = 'o'
            else:
                option = 'c'
        else:
            if agent.current_option == 'bb':
                option = 'bob'
            else:
                option = 'bb'

    return option


def get_option_starting_position(option, task, i=None):
    row, col = None, None

    assert task == 'buttons' or task == 'rendezvous' or task == 'officeworld', 'Experiment not implemented yet'

    if task == 'buttons':
        # '''
        if option == 'by':
            row, col = 0, 0
        elif option == 'g':
            row, col = 0, 2
        elif option == 'bg':
            row, col = 0, 5
        elif option == 'a2br':
            row, col = 5, 6
        elif option == 'a3br':
            row, col = 0, 8
        # Generate random starting position depending on the option
        '''
        if option == 'by' or option == 'g':
            row = np.random.randint(10)
            if row < 8:
                col = np.random.randint(3)
            else:
                if option == 'by':
                    col = np.random.randint(5)
                else:
                    col = np.random.randint(10)
        elif option == 'bg':
            row = np.random.randint(2)
            col = np.random.randint(low=4, high=7)
        #elif option == 'br' and i == 1:
        elif option == 'a2br':
            row = np.random.randint(7)
            if row < 5:
                col = np.random.randint(low=4, high=7)
            else:
                col = np.random.randint(low=4, high=10)
        #elif option == 'br' and i == 2:
        elif option == 'a3br':
            row = np.random.randint(2)
            col = np.random.randint(low=8, high=10)
        # '''

    elif task == 'rendezvous':
        if option[0] == 'g':
            row, col = (3, 4)
        elif option == 'recharge':
            row, col = (2, 0)
        else:
            positions = {1: (0, 0), 2: (0, 3), 3: (2, 0), 4: (0, 8),
                         5: (9, 0), 6: (4, 0), 7: (7, 0), 8: (4, 9), 9: (9, 6), 10: (6, 9)}

            row, col = positions[int(option[1:])]

    elif task == 'officeworld':
        if option == 'c':
            row, col = 0, 14
            #row, col = 2, 0
        elif option == 'o':
            row, col = 4, 13
            #row, col = 6, 2

        elif option == 'bb':
            row, col = 0, 2
        elif option == 'bob':
            row, col = 1, 1

    return row, col


def run_strategy_training(epsilon,
                          tester,
                          agent_list,
                          training_agent_list,
                          task,
                          single_option_training,
                          invariant_experiment,
                          show_print=True):
    """
    This code runs one i-hrl rm-based training episode. q-functions, and accumulated reward values of agents
    are updated accordingly. If the appropriate number of steps have elapsed, this function will
    additionally run a test episode.

    Parameters
    ----------
    epsilon : float
        Numerical value in (0,1) representing likelihood of choosing a random action.
    tester : Tester object
        Object containing necessary information for current experiment.
    agent_list : list of Agent objects
        Agent objects to be trained and tested.
    training_agent : StrategyAgent
        Agent used to train the options' policies.
    show_print : bool
        Optional flag indicating whether or not to print output statements to terminal.
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params

    num_agents = len(training_agent_list)

    training_environments = {}
    if task == 'buttons':
        for i, training_agent in enumerate(training_agent_list):
            training_environments[i] = ButtonsEnv(training_agent.rm_file, i + 1, tester.env_settings)
    elif task == 'rendezvous':
        for i, training_agent in enumerate(training_agent_list):
            training_environments[i] = GridWorldEnv(training_agent.rm_file, int(training_agent.agent_id),
                                                    tester.env_settings, invariant_experiment=invariant_experiment)
    elif task == 'officeworld':
        for i, training_agent in enumerate(training_agent_list):
            training_environments[i] = OfficeWorldEnv(training_agent.rm_file, i + 1, tester.env_settings)

    steps_counter = {}

    for i in range(num_agents):
        training_agent_list[i].reset_state()
        training_agent_list[i].reset_option()
        steps_counter[i] = 0

    mc_rewards = dict()
    for i in range(num_agents):
        mc_rewards[i] = []

    # Max number of steps for each training episode
    num_steps = learning_params.max_timesteps_per_task

    for t in range(num_steps):
        tester.add_step()

        for i, training_agent in enumerate(training_agent_list):
            steps_counter[i] += 1
            # option = training_agent.get_options_list()[0]
            if t == 0 or training_agent.option_complete:
                training_agent_list[i].reset_state()
                training_agent_list[i].initialize_reward_machine()

                steps_counter[i] += 1
                training_agent.option_complete = False
                # training_agent.current_option = option

                option = get_next_option(training_agent, tester.experiment, single_option_training,
                                         invariant_experiment)
                training_agent.current_option = option

                if option == 'a2br':
                    training_environments[i].unlock_wall('yellow')
                if option == 'c' or option == 'o':
                    training_environments[i].unlock_wall()

                row, col = get_option_starting_position(option, tester.experiment, i)

                s_new = training_environments[i].get_state_from_description(row, col)

                training_agent.set_state(s_new)
            # print(training_agent.current_option)
            if not training_agent.option_complete:
                s, a = training_agent.get_next_action(epsilon, learning_params)
                r, l, s_new = training_environments[i].hrm_environment_step(s, a,
                                                                            option=training_agent.current_option)
                training_agent.update_agent(s_new, r, a, learning_params)
                #if r == -1 and training_agent.agent_id == 1:
                #    print(training_agent.option_q_dict[training_agent.current_option])
                if training_agent.counterfactual_training:
                    for counterfactual_option in training_agent.option_q_dict.keys():
                        if counterfactual_option == training_agent.current_option:
                            continue
                        counterfactual_r = training_environments[i].counterfactual_hrm_environment_step(s, s_new,
                                                                                         option=counterfactual_option)
                        #print(counterfactual_option, training_agent.current_option)
                        training_agent.update_q_function(s, s_new, a, counterfactual_option,
                                                         counterfactual_r, learning_params)

                if r != 0 or (invariant_experiment and l == ['r,discharged']) or (tester.experiment == 'officeworld' and
                                                                                  'd' in l):
                    training_agent.option_complete = True
                    #if i == 0:
                    #    print('Option ' + training_agent.current_option + ' completed in {} steps'.format(steps_counter[i]) +
                    #      ' with a reward of {}'.format(r))
                    steps_counter[i] = 0

                    if invariant_experiment:
                        training_environments[i].system_recharged = False
                    # if i == 2:
                    #    print('Option ' + training_agent.current_option + ' completed in {} steps'.format(steps_counter[i]))
                    #    steps_counter[i] = 0

        # If enough steps have elapsed, test and save the performance of the agents.
        if testing_params.test and tester.get_current_step() % testing_params.test_freq == 0:
            for agent, training_agent in zip(agent_list, training_agent_list):
                for option in training_agent.option_q_dict.keys():
                    agent.option_q_dict[option] = training_agent.option_q_dict[option][:]

            t_init = time.time()
            step = tester.get_current_step()

            agent_list_copy = []

            # Need to create a copy of the agent for testing. If we pass the agent directly
            # mid-episode to the test function, the test will reset the world-state and reward machine
            # state before the training episode has been completed.
            for j in range(len(agent_list)):
                rm_file = agent_list[j].rm_file
                options = agent_list[j].options_list
                actions = agent_list[j].actions
                s_i = agent_list[j].s_i
                num_states = agent_list[j].num_states
                agent_id = agent_list[j].agent_id
                agent_copy = StrategyAgent(rm_file, options, actions, s_i, num_states, agent_id)
                # Pass only the q-function by reference so that the testing updates the original agent's q-function.
                agent_copy.option_q_dict = agent_list[j].option_q_dict
                # agent_copy.option_q_dict = training_agent.option_q_dict

                agent_list_copy.append(agent_copy)

            # Run a test of the performance of the agents
            testing_reward, trajectory, testing_steps = run_strategy_test(agent_list_copy,
                                                                          tester,
                                                                          learning_params,
                                                                          testing_params,
                                                                          invariant_experiment,
                                                                          show_print=show_print)
            # Save the testing reward
            if 0 not in tester.results.keys():
                tester.results[0] = {}
            if step not in tester.results[0]:
                tester.results[0][step] = []
            tester.results[0][step].append(testing_reward)

            # Save the testing trace
            if 'trajectories' not in tester.results.keys():
                tester.results['trajectories'] = {}
            if step not in tester.results['trajectories']:
                tester.results['trajectories'][step] = []
            tester.results['trajectories'][step].append(trajectory)

            # If task involves an invariant save the coalition's final reward
            if invariant_experiment or tester.experiment == 'officeworld':
                if 'testing_reward' not in tester.results.keys():
                    tester.results['testing_reward'] = {}
                if step not in tester.results['testing_reward']:
                    tester.results['testing_reward'][step] = []
                tester.results['testing_reward'][step].append(testing_reward)
                '''
                if 'completed_testing_steps' not in tester.results.keys():
                    tester.results['completed_testing_steps'] = {}
                if 'failed_testing_steps' not in tester.results.keys():
                    tester.results['failed_testing_steps'] = {}
                if step not in tester.results['completed_testing_steps']:
                    tester.results['completed_testing_steps'][step] = []
                if step not in tester.results['failed_testing_steps']:
                    tester.results['failed_testing_steps'][step] = []

                if testing_reward == 1:
                    tester.results['completed_testing_steps'][step].append(testing_steps)
                    tester.results['failed_testing_steps'][step].append(1000)
                elif testing_reward == -1:
                    tester.results['completed_testing_steps'][step].append(1000)
                    tester.results['failed_testing_steps'][step].append(testing_steps)
                else:
                    tester.results['completed_testing_steps'][step].append(1000)
                    tester.results['failed_testing_steps'][step].append(1000)
                '''
            # Save how many steps it took to complete the task
            else:
                if 'testing_steps' not in tester.results.keys():
                    tester.results['testing_steps'] = {}
                if step not in tester.results['testing_steps']:
                    tester.results['testing_steps'][step] = []
                tester.results['testing_steps'][step].append(testing_steps)

            # Keep track of the steps taken
            if len(tester.steps) == 0 or tester.steps[-1] < step:
                tester.steps.append(step)


def run_strategy_test(agent_list,
                      tester,
                      learning_params,
                      testing_params,
                      invariant_experiment,
                      show_print=True):
    testing_reward = 0
    trajectory = []
    step = 0

    num_agents = len(agent_list)

    if tester.experiment == 'buttons':
        testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings, strategy_rm=True)
    elif tester.experiment == 'rendezvous':
        testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings,
                                             invariant_experiment=invariant_experiment, strategy_rm=True)
    elif tester.experiment == 'officeworld':
        testing_env = MultiAgentOfficeWorldEnv(tester.rm_test_file, tester.env_settings)

    for i in range(num_agents):
        agent_list[i].reset_state()
        agent_list[i].initialize_reward_machine()
        agent_id = i
        agent_list[i].current_option = testing_env.get_next_agent_option(agent_id, agent_list[i].u)
        #if i == 0:
            #print([agent_list[i].option_q_dict[option] for option in agent_list[i].option_q_dict.keys()])

    s_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        s_team[i] = agent_list[i].s
    a_team = np.full(num_agents, -1, dtype=int)
    u_team = np.full(num_agents, -1, dtype=int)
    for i in range(num_agents):
        u_team[i] = agent_list[i].u
    testing_reward = 0

    # Starting interaction with the environment
    for t in range(testing_params.num_steps):
        step += 1

        # Perform a team step
        for i in range(num_agents):
            s, a = agent_list[i].get_next_action(-1.0, learning_params)
            s_team[i] = s
            a_team[i] = a
            u_team[i] = agent_list[i].u
            # if i == 1 or i == 2:
            #    print(agent_list[i].current_option, s, agent_list[i].option_q_dict[agent_list[i].current_option][s])

        r, l, s_team_next = testing_env.environment_step(s_team, a_team)

        testing_reward = testing_reward + r
        projected_l_dict = {}
        for i in range(num_agents):
            # Agent i's projected label is the intersection of l with its local event set
            projected_l_dict[i] = list(set(agent_list[i].local_event_set) & set(l))
            # Check if the event causes a transition from the agent's current RM state
            if not (agent_list[i].is_local_event_available(projected_l_dict[i])):
                projected_l_dict[i] = []

        for i in range(num_agents):
            # Enforce synchronization requirement on shared events
            if projected_l_dict[i]:
                for event in projected_l_dict[i]:
                    for j in range(num_agents):
                        if (event in set(agent_list[j].local_event_set)) and (
                                not (projected_l_dict[j] == projected_l_dict[i])):
                            projected_l_dict[i] = []

            # update the agent's internal representation
            # a = testing_env.get_last_action(i)
            u_next = None
            if projected_l_dict[i]:
                u_next = agent_list[i].rm.get_next_state(agent_list[i].u, projected_l_dict[i][0])
            agent_list[i].update_agent(s_team_next[i], r, a_team[i], learning_params, u_new=u_next,
                                       update_q_function=False)
            agent_id = i  # if tester.experiment == 'buttons' else i + 1
            agent_list[i].current_option = testing_env.get_next_agent_option(agent_id, agent_list[i].u)

        if r != 0 or (invariant_experiment and testing_env.discharged):
            break

    if show_print:
        print('Reward of {} achieved in {} steps. Current step: {} of {}'.format(testing_reward, step,
                                                                                 tester.current_step,
                                                                                 tester.total_steps))

    return testing_reward, trajectory, step


def run_strategy_experiment(tester,
                            num_agents,
                            num_times,
                            invariant_experiment=False,
                            counterfactual_experiment=False,
                            show_print=True):
    """
    Run the entire q-learning with reward machines experiment a number of times specified by num_times.

    Inputs
    ------
    tester : Tester object
        Test object holding true reward machine and all information relating
        to the particular tasks, world, learning parameters, and experimental results.
    num_agents : int
        Number of agents in this experiment.
    num_times : int
        Number of times to run the entire experiment (restarting training from scratch).
    show_print : bool
        Flag indicating whether or not to output text to the terminal.
    """

    learning_params = tester.learning_params

    for t in range(num_times):
        # Resetting default step values
        tester.restart()

        rm_learning_file_list = tester.rm_learning_file_list

        # Verify that the number of local reward machines matches the number of agents in the experiment.
        assertion_string = "Number of specified local reward machines must match specified number of agents."
        assert (len(tester.rm_learning_file_list) == num_agents), assertion_string

        if tester.experiment == 'rendezvous':
            testing_env = MultiAgentGridWorldEnv(tester.rm_test_file, num_agents, tester.env_settings, strategy_rm=True,
                                                 invariant_experiment=invariant_experiment)
            num_states = testing_env.num_states
        elif tester.experiment == 'buttons':
            testing_env = MultiAgentButtonsEnv(tester.rm_test_file, num_agents, tester.env_settings, strategy_rm=True)
            num_states = testing_env.num_states
        elif tester.experiment == 'officeworld':
            testing_env = MultiAgentOfficeWorldEnv(tester.rm_test_file, tester.env_settings)
            num_states = testing_env.num_states

        # Create the list of agents for this experiment
        agent_list = []
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            options_list = testing_env.get_options_list(i)
            s_i = testing_env.get_initial_state(i)
            agent_list.append(StrategyAgent(rm_learning_file_list[i], options_list, actions, s_i, num_states, i))

        num_episodes = 0

        # Task loop
        epsilon = learning_params.initial_epsilon

        training_agent_list = []

        '''
        single_option_training = True
        for option in options:
            rm_file = agent_list[0].rm_file
            s_i = agent_list[0].s_i
            actions = agent_list[0].actions
            if tester.experiment == 'rendezvous':
                agent_id = option[1:]
            else:
                agent_id = options_to_id[option]
            num_states = agent_list[0].num_states
            training_agent_list.append(StrategyAgent(rm_file, [option], actions, s_i, num_states, agent_id))
        '''
        single_option_training = False
        for i in range(num_agents):
            actions = testing_env.get_actions(i)
            options_list = testing_env.get_options_list(i)
            s_i = testing_env.get_initial_state(i)
            agent_id = i if tester.experiment == 'buttons' else i + 1
            training_agent_list.append(StrategyAgent(rm_learning_file_list[i], options_list, actions,
                                                     s_i, num_states, agent_id,
                                                     counterfactual_training=counterfactual_experiment))
            if invariant_experiment and i == 2:
                training_agent_list[i].current_option = options_list[-1]
            elif tester.experiment == 'officeworld':
                training_agent_list[i].current_option = options_list[0]
            else:
                training_agent_list[i].current_option = options_list[-1] if tester.experiment == 'buttons' \
                    else 'g{}'.format(i + 1)

        # '''

        while not tester.stop_learning():
            num_episodes += 1

            # epsilon = epsilon*0.99

            run_strategy_training(epsilon,
                                  tester,
                                  agent_list,
                                  training_agent_list,
                                  tester.experiment,
                                  single_option_training,
                                  invariant_experiment,
                                  show_print=show_print)

        # Backing up the results
        print('Finished iteration ', t)

    tester.agent_list = agent_list

    # avg_timesteps, tests = 0, 0
    # for step in tester.results['testing_steps'].keys():
    #    for test_steps in tester.results['testing_steps'][step]:
    #        avg_timesteps += test_steps
    #        tests += 1

    plot_strategy_results(tester, num_agents, invariant_experiment)


def plot_strategy_results(tester, num_agents, invariant_experiment):
    """
    Plot the results stored in tester.results for each of the agents.
    """

    if invariant_experiment:
        avg_reward = list()

        steps, rewards_plot_dict = [], tester.results['testing_reward']

        for step in rewards_plot_dict.keys():
            avg_reward.append(sum(rewards_plot_dict[step]) / len(rewards_plot_dict[step]))
            steps.append(step)

        plt.plot(steps, avg_reward, color='green')

        plt.grid()
        plt.ylabel('Average Reward', fontsize=15)
        plt.xlabel('Training Steps', fontsize=15)
        plt.locator_params(axis='x', nbins=5)
    elif tester.experiment == 'officeworld':
        prc_25 = list()
        prc_50 = list()
        prc_75 = list()

        # Buffers for plots
        current_step = list()
        current_25 = list()
        current_50 = list()
        current_75 = list()
        steps = list()

        plot_dict = tester.results['testing_reward']

        for step in plot_dict.keys():
            if len(current_step) < 10:
                current_25.append(np.percentile(np.array(plot_dict[step]), 25))
                current_50.append(np.percentile(np.array(plot_dict[step]), 50))
                current_75.append(np.percentile(np.array(plot_dict[step]), 75))
                current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))
            else:
                current_step.pop(0)
                current_25.pop(0)
                current_50.pop(0)
                current_75.pop(0)
                current_25.append(np.percentile(np.array(plot_dict[step]), 25))
                current_50.append(np.percentile(np.array(plot_dict[step]), 50))
                current_75.append(np.percentile(np.array(plot_dict[step]), 75))
                current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))

            prc_25.append(sum(current_25) / len(current_25))
            prc_50.append(sum(current_50) / len(current_50))
            prc_75.append(sum(current_75) / len(current_75))
            steps.append(step)

        plt.plot(steps, prc_25, alpha=0)
        plt.plot(steps, prc_50, color='red')
        plt.plot(steps, prc_75, alpha=0)
        plt.grid()
        plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
        plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
        plt.ylabel('Reward Obtained', fontsize=15)
        plt.xlabel('Training Steps', fontsize=15)
        plt.locator_params(axis='x', nbins=5)
    else:
        prc_25 = list()
        prc_50 = list()
        prc_75 = list()

        # Buffers for plots
        current_step = list()
        current_25 = list()
        current_50 = list()
        current_75 = list()
        steps = list()

        plot_dict = tester.results['testing_steps']

        for step in plot_dict.keys():
            if len(current_step) < 10:
                current_25.append(np.percentile(np.array(plot_dict[step]), 25))
                current_50.append(np.percentile(np.array(plot_dict[step]), 50))
                current_75.append(np.percentile(np.array(plot_dict[step]), 75))
                current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))
            else:
                current_step.pop(0)
                current_25.pop(0)
                current_50.pop(0)
                current_75.pop(0)
                current_25.append(np.percentile(np.array(plot_dict[step]), 25))
                current_50.append(np.percentile(np.array(plot_dict[step]), 50))
                current_75.append(np.percentile(np.array(plot_dict[step]), 75))
                current_step.append(sum(plot_dict[step]) / len(plot_dict[step]))

            prc_25.append(sum(current_25) / len(current_25))
            prc_50.append(sum(current_50) / len(current_50))
            prc_75.append(sum(current_75) / len(current_75))
            steps.append(step)

        plt.plot(steps, prc_25, alpha=0)
        plt.plot(steps, prc_50, color='red')
        plt.plot(steps, prc_75, alpha=0)
        plt.grid()
        plt.fill_between(steps, prc_50, prc_25, color='red', alpha=0.25)
        plt.fill_between(steps, prc_50, prc_75, color='red', alpha=0.25)
        plt.ylabel('Testing Steps to Task Completion', fontsize=15)
        plt.xlabel('Training Steps', fontsize=15)
        plt.locator_params(axis='x', nbins=5)

    axes = plt.gca()
    if not (invariant_experiment or tester.experiment == 'officeworld'):
        axes.set_ylim([-27.5, 1047.5])
        plt.legend()
        plt.xscale('log')

    plt.show()
