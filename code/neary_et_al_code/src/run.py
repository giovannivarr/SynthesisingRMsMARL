import pickle
from datetime import datetime
import os

if __name__ == "__main__":

    num_times = 10 # Number of separate trials to run the algorithm for

    num_agents = 10 # This will be automatically set to 3 for buttons experiment (max 10)

    #experiment = 'rendezvous'
    #experiment = 'centralized_rendezvous'
    #experiment = 'ihrl_rendezvous'
    #experiment = 'strategy_rendezvous'
    #experiment = 'strategy_invariant_rendezvous'
    #experiment = 'iql_rendezvous'

    #experiment = 'buttons'
    #experiment = 'ihrl_buttons'
    #experiment = 'strategy_buttons'
    #experiment = 'counterfactual_buttons'
    #experiment = 'iql_buttons'

    #experiment = 'officeworld'
    experiment = 'strategy_officeworld'

    if experiment == 'rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.dqprm import run_multi_agent_experiment
        tester = rendezvous_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times)

    if experiment == 'centralized_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.run_centralized_coordination_experiment import run_centralized_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_centralized_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'ihrl_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.run_ihrl_experiment import run_ihrl_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'strategy_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.run_strategy_experiment import run_strategy_experiment
        tester = rendezvous_config(num_times, num_agents, strategy_rm=True)
        run_strategy_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'strategy_invariant_rendezvous':
        num_agents = 3
        from rendezvous_config import rendezvous_config
        from experiments.run_strategy_experiment import run_strategy_experiment
        tester = rendezvous_config(num_times, num_agents, strategy_rm=True, invariant_experiment=True)
        run_strategy_experiment(tester, num_agents, num_times, show_print=True, invariant_experiment=True)

    if experiment == 'iql_rendezvous':
        from rendezvous_config import rendezvous_config
        from experiments.iql import run_iql_experiment
        tester = rendezvous_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'buttons':
        from buttons_config import buttons_config
        from experiments.dqprm import run_multi_agent_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents) # Get test object from config script
        run_multi_agent_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'ihrl_buttons':
        from buttons_config import buttons_config
        from experiments.run_ihrl_experiment import run_ihrl_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_ihrl_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'strategy_buttons':
        from buttons_config import buttons_config
        from experiments.run_strategy_experiment import run_strategy_experiment
        num_agents = 3  # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents, strategy_rm=True)
        run_strategy_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'counterfactual_buttons':
        from buttons_config import buttons_config
        from experiments.run_strategy_experiment import run_strategy_experiment
        num_agents = 3  # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents, strategy_rm=True)
        run_strategy_experiment(tester, num_agents, num_times, show_print=True, counterfactual_experiment=True)

    if experiment == 'iql_buttons':
        from buttons_config import buttons_config
        from experiments.iql import run_iql_experiment
        num_agents = 3 # Num agents must be 3 for this example
        tester = buttons_config(num_times, num_agents)
        run_iql_experiment(tester, num_agents, num_times, show_print=True)

    if experiment == 'officeworld':
        from officeworld_config import officeworld_config
        from experiments.dqprm import run_multi_agent_experiment
        tester = officeworld_config(num_times)
        run_multi_agent_experiment(tester, 2, num_times)

    if experiment == 'strategy_officeworld':
        from officeworld_config import officeworld_config
        from experiments.run_strategy_experiment import run_strategy_experiment
        #tester = officeworld_config(1)
        tester = officeworld_config(num_times)
        #run_strategy_experiment(tester, 2, 1)
        run_strategy_experiment(tester, 2, num_times, counterfactual_experiment=True)


    # Save the results
    parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    data_path = os.path.join(parentDir, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    experiment_data_path = os.path.join(data_path, experiment)

    if not os.path.isdir(experiment_data_path):
        os.mkdir(experiment_data_path)

    now = datetime.now()
    save_file_str = r'\{}_'.format(now.strftime("%Y-%m-%d_%H-%M-%S"))
    save_file_str = save_file_str + experiment + '.p'
    save_file = open(experiment_data_path + save_file_str, "wb")
    pickle.dump(tester, save_file)
