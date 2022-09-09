import sys
from itertools import product
from typing import Dict, List, Tuple


def get_prop_symbols(actions: Dict[str, List[str]], env_vars: List[str]) -> List[str]:
    """
    Returns the list of propositional symbols of the ispl specification.

    :param actions: Dictionary in which the keys are the actions' names and the values lists of strings containing the
    three lines of the specification of the corresponding action.
    :return: List of strings, where each string is a propositional symbol.
    """

    prop_symbols = []
    for conditions in actions.values():
        prop_symbols += conditions[0].split(',') + conditions[1].split(',')

    env_vars_list = []
    for var in env_vars:
        env_vars_list += [var.split(':')[0]]

    prop_symbols += env_vars_list

    prop_symbols = [p for p in prop_symbols if p]
    for i in range(len(prop_symbols)):
        p = prop_symbols[i]
        if p[0] == '~':
            prop_symbols[i] = p[1:]

    prop_symbols = list(set(prop_symbols))

    return prop_symbols


def get_valid_nonvalid_action_tuples(agents_actions: List[List[str]], actions: Dict[str, List[str]],
                                     postconditions_dict: Dict[str, List[str]]) \
        -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """
    Returns the list of valid action tuples and non-valid action tuples.

    :param agents_actions: List of lists of strings, where the i-th list is the list of actions that can be performed by
    agent i.
    :param actions: Dictionary in which the keys are the actions' names and the values lists of strings containing the
    three lines of the specification of the corresponding action.
    :param postconditions_dict: Dictionary in which the keys are the actions' names and the values list of strings
    containing the postconditions of the corresponding action.
    :return: A pair of lists of tuples of strings, where the first list is the list of valid action tuples and the
    second list the list of non-valid action tuples.
    """

    agents_needed = {a: int(actions[a][2]) for a in actions.keys()}

    actions_tuples = list(product(*agents_actions))

    valid_action_tuples = []

    for actions_tuple in actions_tuples:
        single_actions = set(actions_tuple)
        tuple_postconditions = []
        flag = True
        # For each action tuple check that every action is performed by the exact number of agents needed for it
        for a in single_actions:
            tuple_postconditions += postconditions_dict[a]
            # If the exact number is 0 then the action can be performed by a variable number of agents
            if agents_needed[a] == 0:
                continue
            elif actions_tuple.count(a) != agents_needed[a]:
                flag = False
                break

        if not flag:
            continue

        # For each action tuple check whether the set of its postconditions does not contain contradictory ones: if it
        # does the tuple is not valid
        for literal in tuple_postconditions:
            if literal in tuple_postconditions and '~' + literal in tuple_postconditions:
                flag = False
                break

        if flag:
            valid_action_tuples += [actions_tuple]

    return valid_action_tuples


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Two file paths should be given in input"

    input_file, output_file = open(sys.argv[1], 'r'), open(sys.argv[2], 'w')

    spec, out_ispl = input_file.read(), ''
    sections = spec.split('=====')

    # Dictionary of actions: for each action (key) we save preconditions, postconditions and agents needed to perform it
    actions = {}
    for a in sections[0].split('::')[1:]:
        action = a.split('\n')[:-1]
        actions[action[0]] = action[1:]

    # Now we obtain all propositional symbols that appear in the specification of the ispl
    env_vars = []
    if sections[3].split('\n')[3]:
        env_vars = sections[3].split('\n')[3].split(',')
    prop_symbols = get_prop_symbols(actions, env_vars)

    # For each propositional symbol we create a variable for the environment (and add an 'error' variable)
    out_ispl += 'Semantics=SingleAssignment;\n\nAgent Environment\n\tVars:\n'
    for prop in prop_symbols:
        out_ispl += '\t\t' + prop + ': boolean;\n'

    out_ispl += '\tend Vars\n\tActions = { };\n\tProtocol:\n\tend Protocol\n'

    # Create a dictionary where the keys are actions and the values the list of corresponding postconditions
    postconditions_dict = {}
    for a in actions.keys():
        postconditions_dict[a] = actions[a][1].split(',')

    # Create a list of lists where the i-th list is the list of actions that can be performed by agent i
    agents = sections[1].split('\n')[1:-1]
    agents_actions = [a.split('::')[0].split(',') for a in agents]
    agents_actions = [a for a in agents_actions]

    agents_observables = {i: [] for i in range(len(agents))}
    for i, a in enumerate(agents):
        if len(a.split('::')) > 1:
            agents_observables[i] = a.split('::')[1].split(',')

    valid_action_tuples = get_valid_nonvalid_action_tuples(agents_actions, actions, postconditions_dict)

    # Environment evolutions sections
    out_ispl += '\tEvolution:\n'
    prop_dynamics = {prop: {'true': [], 'false': []} for prop in prop_symbols}
    constant_actions, constant_tuples = [], []
    for action, postconditions in postconditions_dict.items():
        if postconditions == ['']:
            constant_actions += [action]
        else:
            for postcondition in postconditions:
                new_value = 'false' if postcondition[0] == '~' else 'true'
                if postcondition[0] == '~':
                    prop_dynamics[postcondition[1:]]['false'] += [t for t in valid_action_tuples if action in t]
                else:
                    prop_dynamics[postcondition]['true'] += [t for t in valid_action_tuples if action in t]

    # Create the list of tuples that do not have any effect on the environment, if there are any such actions
    if constant_actions:
        constant_tuples = [action_tuple for action_tuple in valid_action_tuples if
                           all(action in constant_actions for action in action_tuple)]

    prop_dynamics_str = ''
    for prop, prop_dict in prop_dynamics.items():
        for new_value, tuple_list in prop_dict.items():
            # If this new_value for the propositional variables cannot be brought about, skip to the next
            if not (tuple_list or constant_tuples):
                continue

            if not tuple_list and constant_tuples:
                prop_dynamics_str += '\t\t' + prop + '=' + new_value + '\tif ( ' + prop + ' = ' + new_value + ' and ' + \
                                    ' and '.join(['Agent' + str(i) + '.Action = ' + action for i, action
                                    in enumerate(constant_tuples[0])]) + ' )'
                for actions_tuple in constant_tuples[1:]:
                    prop_dynamics_str += ' or ( ' + prop + ' = ' + new_value + ' and ' + \
                                         ' and '.join(['Agent' + str(i) + '.Action = ' + action for i, action
                                                       in enumerate(actions_tuple)]) + ' )'
                prop_dynamics_str = prop_dynamics_str + ';\n'
                continue

            if tuple_list:
                prop_dynamics_str += '\t\t' + prop + '=' + new_value + '\tif ( ' + ' and '.join(['Agent' + str(i) +
                                                                '.Action = ' + action for i, action
                                                                in enumerate(tuple_list[0])]) + ' )'
                for actions_tuple in tuple_list[1:]:
                    prop_dynamics_str += ' or ( ' + ' and '.join(['Agent' + str(i) + '.Action = ' + action for i, action
                                                                  in enumerate(actions_tuple)]) + ' )'
            if constant_tuples:
                for actions_tuple in constant_tuples:
                    prop_dynamics_str += ' or ( ' + prop + ' = ' + new_value + ' and ' + \
                                         ' and '.join(['Agent' + str(i) + '.Action = ' + action for i, action
                                         in enumerate(actions_tuple)]) + ' )'

            prop_dynamics_str = prop_dynamics_str + ';\n'

    if env_vars != ['']:
        for var in env_vars:
            prop = var.split(':')[0]
            old_val, new_val = var.split(':')[1].split('>')

            prop_dynamics_str += '\t\t' + prop + '=' + new_val + '\tif ( ' + prop + ' = ' + old_val

            for action, postconditions in postconditions_dict.items():
                if (new_val == 'true' and '~' + prop in postconditions) \
                        or (new_val == 'false' and prop in postconditions):
                    for agent, agent_actions in enumerate(agents_actions):
                        if action in agent_actions:
                            prop_dynamics_str += ' and !( Agent{}.Action = '.format(agent) + action + ' )'

            prop_dynamics_str += ' );\n'



    out_ispl += prop_dynamics_str

    out_ispl += '\tend Evolution\nend Agent\n\n'

    # Agent section of the ispl
    agents_ispl_section = ''
    for agent, agent_actions in enumerate(agents_actions):
        agents_ispl_section += 'Agent Agent' + str(agent)
        agent_observable_variables = set()
        if agents_observables[agent]:
            agent_observable_variables = agents_observables[agent]
        else:
            for action in agent_actions:
                if action in constant_actions:
                    continue
                # This adds both pre- and postconditions as observables for each agent
                action_variables = [var for var in actions[action][0].split(',') + actions[action][1].split(',') if var]
                # This adds only preconditions
                for i, var in enumerate(action_variables):
                    if var[0] == '~':
                        action_variables[i] = var[1:]
                for variable in action_variables:
                    agent_observable_variables.add(variable)
        agents_ispl_section += '\n\tLobsvars = { ' + ', '.join(
            agent_observable_variables) + ' };\n\tVars:\n\t\tagent_dummy: boolean;\n\tend Vars\n\tActions = { ' + \
                               ', '.join(agent_actions) + ' };\n\tProtocol:\n'



        for action in agent_actions:
            if action in constant_actions:
                agents_ispl_section += '\t\tagent_dummy = true or agent_dummy = false:\t{ ' + action + ' };\n'
                continue
            action_preconditions_str, action_preconditions = '\t\t', []
            action_variables = [var for var in actions[action][0].split(',') if var]
            for var in action_variables:
                if var[0] == '~':
                    assert var[1:] in agent_observable_variables, 'Agent {} should be able to observe all variables ' \
                                                              'needed in the preconditions of ' \
                                                              'their actions'.format(agent)
                    action_preconditions += ['Environment.' + var[1:] + ' = false']
                else:
                    assert var in agent_observable_variables, 'Agent {} should be able to observe all variables ' \
                                                              'needed in the preconditions of ' \
                                                              'their actions'.format(agent)
                    action_preconditions += ['Environment.' + var + ' = true']
            if not action_preconditions:
                action_preconditions_str += 'agent_dummy = true or agent_dummy = false:\t'
            else:
                action_preconditions_str += ' and '.join(action_preconditions) + ':\t'
            agents_ispl_section += action_preconditions_str + '{ ' + action + ' };\n'

        agents_ispl_section += '\tend Protocol\n\tEvolution:\n\t\tagent_dummy = true\tif agent_dummy = true;' \
                               '\n\t\tagent_dummy = false\tif agent_dummy = false;\n\tend Evolution\nend Agent\n\n'

    out_ispl += agents_ispl_section

    # Evaluation section of the ispl
    formula = sections[2][1:-1]
    out_ispl += 'Evaluation\n'
    goals_dict = {}

    for element in formula.split(' '):
        if element not in ['U', 'X', 'F', 'G', 'and', 'or', '!', '(', ')', '<gA>']:
            literal = element[1:] if element[0] == '!' else element
            out_ispl += '\t{} if Environment.{} = true;\n'.format(literal, literal)

    out_ispl += 'end Evaluation\n'

    # Wrapping up the ispl by adding the initial states, the agents group and the goal formula to verify
    prop_initial_values = {prop: 'false' for prop in prop_symbols}
    group = range(len(agents))
    if len(sections) == 4:
        if sections[3].split('\n')[1].split(',')[0]:
            group = sections[3].split('\n')[1].split(',')

        for prop in sections[3].split('\n')[2].split(','):
            if not prop:
                continue
            if prop[0] == '~':
                prop_initial_values[prop[1:]] = 'false'
            else:
                prop_initial_values[prop] = 'true'

    out_ispl += '\nInitStates\n\t'
    out_ispl += ' and '.join(['Environment.' + prop + ' = ' + prop_initial_values[prop] for prop in prop_symbols]) \
                + ' and\n\t'
    out_ispl += ' and\n\t'.join(['Agent' + str(i) + '.agent_dummy = false' for i in range(len(agents_actions))]) + ';'
    out_ispl += '\nend InitStates\n'

    out_ispl += '\nGroups\n\tgA = { ' + ', '.join(
        ['Agent' + str(i) for i in group]) + ' };\nend Groups\n'


    out_ispl += '\nFormulae\n\t<gA> ' + formula + ' ;\nend Formulae'

    output_file.write(out_ispl)
    input_file.close()
    output_file.close()
