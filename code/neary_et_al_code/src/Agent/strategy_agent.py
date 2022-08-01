from reward_machines.sparse_reward_machine import SparseRewardMachine
import numpy as np
import random


class StrategyAgent:
    """
    Class meant to represent an independent hierarchical agent augmented with an individual RM.
    The agent maintains a representation of its own q-function and accumulated reward
    which are updated across training episodes.
    """

    def __init__(self, rm_file, options_list, actions, s_i, num_states, agent_id, counterfactual_training=True):
        """
        Initialize agent object.

        Parameters
        ----------
        rm_file : str
            File path pointing to the reward machine this agent is meant to use for learning.
        options_list : list
            list of strings describing the different options available to the agent
        s_i : int
            Index of initial state.
        num_states : int
            Number of states in the environment
        actions : list
            List of actions available to the agent.
        agent_id : int
            Index of this agent.
        """
        self.rm_file = rm_file
        self.options_list = options_list
        self.agent_id = agent_id
        self.s_i = s_i
        self.s = s_i
        #self.last_s = self.s
        self.actions = actions
        self.num_states = num_states
        self.counterfactual_training = counterfactual_training

        self.rm = SparseRewardMachine(self.rm_file)
        self.u_i = self.rm.get_initial_state()
        self.u = self.rm.get_initial_state()
        self.local_event_set = self.rm.get_events()

        #self.options = np.arange(len(self.options_list))
        #self.meta_q = np.zeros([num_meta_states, len(self.options_list)])

        #self.meta_q_dict = dict()               # reward machine q-values for each state and option
        #for state in self.rm.get_states():
        #    self.meta_q_dict[state] = np.zeros((self.num_states, len(self.options)))

        self.option_q_dict = dict()             # options q-values for each state and action
        for option in options_list:
            self.option_q_dict[option] = np.zeros((self.num_states, len(self.actions)))

        self.current_option = ''
        self.option_start_state = -1
        self.option_complete = False

    def initialize_reward_machine(self):
        """
        Reset the state of the reward machine to the initial state and reset task status.
        """
        self.u = self.rm.get_initial_state()
        self.is_task_complete = 0

    def reset_state(self):
        """
        Reset the agent to the initial state of the environment and the initial state of her RM.
        """
        self.s = self.s_i
        self.u = self.u_i

    def reset_option(self):
        """
        Reset the agent to have no currently active option.
        """
        self.current_option = ''
        self.option_start_state = -1
        self.option_complete = False

    def get_options_list(self):
        return self.options_list[:]

    def set_state(self, s_new):
        self.s = s_new

    def is_local_event_available(self, label):
        if label: # Only try accessing the first event in label if it exists
            event = label[0]
            return self.rm.is_event_available(self.u, event)
        else:
            return False

    def get_next_action(self, epsilon, learning_params):
        """
        Return the current state and the next action selected by the agent.

        Outputs
        -------
        a : int
            Selected next action for this agent.
        """
        T = learning_params.T
        option = self.current_option
        # If agent has to wait return "None" action
        if option == 'w{}'.format(int(self.agent_id) + 1):
            return self.s, 4

        q = self.option_q_dict[option]

        if random.random() < epsilon:
            a = random.choice(self.actions)
            a_selected = a
        else:
            pr_sum = np.sum(np.exp(q[self.s, :] * T))
            pr = np.exp(q[self.s, :] * T) / pr_sum  # pr[a] is probability of taking action a

            # If any q-values are so large that the softmax function returns infinity,
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                #print('BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.')
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            pr_select = np.zeros([len(self.actions) + 1, 1])
            pr_select[0] = 0
            for i in range(len(self.actions)):
                pr_select[i + 1] = pr_select[i] + pr[i]

            randn = random.random()
            for a in self.actions:
                if randn >= pr_select[a] and randn <= pr_select[a + 1]:
                    a_selected = a
                    break
            '''

            best_actions = np.where(self.q[self.s, self.u, :] == np.max(self.q[self.s, self.u, :]))[0]
            a_selected = random.choice(best_actions)'''

        a = a_selected

        return self.s, a

    def update_agent(self, s_new, r, a, learning_params, u_new=None, update_q_function=True):
        """
        Update the agent's state, q-function, and reward machine after
        interacting with the environment.

        Parameters
        ----------
        s_new : int
            Index of the agent's next state.
        r : int
            Current reward given by the environment to the agent.
        a : int
            Action the agent took from the last state.
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        u_new : int
            Next state of the agent's RM if we are to update it.
        """
        if update_q_function:
            self.update_q_function(self.s, s_new, a, self.current_option, r, learning_params)

        # Update current state and RM state if given in input
        self.s = s_new

        if u_new:
            self.u = u_new

    def update_q_function(self, s, s_new, a, option, reward, learning_params):
        """
        Update the q function using the action, states, and reward value.

        Parameters
        ----------
        s : array
            Indeces of the agents' previous state
        s_new : array
            Indeces of the agents' updated state
        a : int
            Index of low-level action taken
        option : string
            String describing the option whose q function is being updated.
        reward : float
            Intrinsic reward. Should be 1 if option was completed in moving from s to s_new, 0 otherwise.
        a : int
            Action the agent took from state s
        learning_params : LearningParameters object
            Object storing parameters to be used in learning.
        """
        alpha = learning_params.alpha
        gamma = learning_params.gamma

        # Bellman update
        self.option_q_dict[option][s, a] = (1 - alpha) * self.option_q_dict[option][s, a] + alpha * (
                    reward + gamma * np.amax(self.option_q_dict[option][s_new]))