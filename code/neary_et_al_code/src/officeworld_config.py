from tester.tester import Tester
from tester.tester_params import TestingParameters
from tester.learning_params import LearningParameters
import os

def officeworld_config(num_times):
    """
    Function setting the experiment parameters and environment.

    Output
    ------
    Tester : tester object
        Object containing the information necessary to run this experiment.
    """
    base_file_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    local_rm_files = [os.path.join(base_file_path, 'experiments', 'officeworld',
                                               'officeworld_rm_agent_{}.txt'.format(i+1)) for i in range(2)]

    joint_rm_file = os.path.join(base_file_path, 'experiments', 'officeworld',
                                 'team_officeworld_rm.txt')

    step_unit = 5000 # use 5000 for hardest officeworld with walls, otherwise use 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit  # 0.1*step_unitr
    # experiment where options are pretrained:
    # testing_params.test_freq = 0.1*step_unit
    testing_params.num_steps = step_unit

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.99  # 0.9
    learning_params.alpha = 0.8 # 0.8
    learning_params.T = 50
    learning_params.initial_epsilon = 0.0  # Set epsilon to zero to turn off epsilon-greedy exploration (only using boltzmann)
    learning_params.max_timesteps_per_task = testing_params.num_steps

    tester = Tester(learning_params, testing_params)
    tester.step_unit = step_unit
    tester.total_steps = 250 * step_unit  # 25 * step_unit (strategy experiment of around 35 seconds per iteration), 150 * step_unit (Neary RM experiment of aroung 35 seconds per iteration)
    # experiment where options are pretrained:
    # tester.total_steps = 25*step_unit
    tester.min_steps = 1

    tester.num_times = num_times
    tester.num_agents = 2

    tester.rm_test_file = joint_rm_file
    tester.rm_learning_file_list = local_rm_files

    # Set the environment settings for the experiment
    env_settings = dict()

    env_settings['Nr'] = 11
    env_settings['Nc'] = 15
    env_settings['initial_states'] = [14, 2]
    env_settings['blue_button'] = (1, 1)
    env_settings['orange_button'] = (6, 1)
    env_settings['walls'] = [(0, 3), (0, 7), (0, 11),
                             (2, 3), (2, 7), (2, 11),
                             (3, 0), (3, 2), (3, 3), (3, 4), (3, 6), (3, 7), (3, 8), (3, 10), (3, 11), (3, 12), (3, 14),
                             (4, 3), (4, 7), (4, 11),
                             (6, 3), (6, 7), (6, 11),
                             (7, 0), (7, 2), (7, 3), (7, 4), (7, 6), (7, 7), (7, 8), (7, 10), (7, 11), (7, 12), (7, 14),
                             (8, 3), (8, 7), (8, 11),
                             (10, 3), (10, 7), (10, 11)]
    #env_settings['walls'] = []
    env_settings['decorations'] = [(1, 5), (1, 9), (1, 13),
                                   (5, 1), (5, 9), (5, 13),
                                   (9, 1), (9, 5), (9, 9), (9, 13)]
    env_settings['blueorange_tiles'] = [(3, 13), (5, 11), (7, 13)]
    #env_settings['blueorange_tiles'] = [(3, 11), (3, 12), (3, 13), (3, 14),
    #                                    (4, 11),
    #                                    (5, 11),
    #                                    (6, 11),
    #                                    (7, 11), (7, 12), (7, 13), (7, 14)]
    env_settings['office'] = (5, 5)
    env_settings['coffee'] = (4,13)
    '''
    env_settings['Nr'] = 7
    env_settings['Nc'] = 7
    env_settings['initial_states'] = [14, 2]
    env_settings['blue_button'] = (1, 1)
    env_settings['orange_button'] = (1, 6)
    env_settings['walls'] = [(0, 3),
                             (2, 3),
                             (3, 0), (3, 2), (3, 3), (3, 4), (3, 6),
                             (4, 3),
                             (6, 3)]
    env_settings['decorations'] = [(1, 5),
                                   (5, 1)]
    env_settings['blueorange_tiles'] = [(3, 1), (5, 3)]
    env_settings['office'] = (5, 5)
    env_settings['coffee'] = (6, 2)'''

    env_settings['p'] = 0.98

    tester.env_settings = env_settings

    tester.experiment = 'officeworld'

    return tester
