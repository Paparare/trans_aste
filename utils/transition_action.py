from copy import deepcopy

action2id = {'shift': 0, 'lr': 1, 'rr': 2, 
            'lr_n': 3, 'rr_n': 4, 'merge': 5}

def is_span_covered(s, span_set):
    for span in span_set:
        if span[0] <= s[0] and span[1] >= s[1]:
            return True
    return False

def get_action(s_pair, pairs):
    action = None

    pairs = set([(pair[0], pair[1]) for pair in pairs])

    span_set = set()
    for pair in pairs:
        span_set.add(pair[0])
        span_set.add(pair[1])

    first_element = s_pair[1]
    second_element = s_pair[0]

    is_first_element_covered = is_span_covered(first_element, span_set)
    is_second_element_covered = is_span_covered(second_element, span_set)
    is_first_element_in_span_set = first_element in span_set
    is_second_element_in_span_set = second_element in span_set

    if is_first_element_covered and is_second_element_covered:
        if is_second_element_in_span_set and not is_first_element_in_span_set:
            action = 'shift'
        elif is_second_element_in_span_set and is_first_element_in_span_set:
            if (first_element, second_element) in pairs:
                action = 'lr'
            elif (second_element, first_element) in pairs:
                action = 'rr'
            else:
                action = 'lr_n'
        elif not is_second_element_in_span_set and is_first_element_in_span_set:
            action = 'lr_n'
        elif not is_second_element_in_span_set and not is_first_element_in_span_set:
            if second_element[1] + 1 == first_element[0]:
                if is_span_covered((second_element[0], first_element[1]), span_set):
                    action = 'merge'
                else:
                    action = 'lr_n'
            else:
                action = 'lr_n'
    elif not is_first_element_covered and is_second_element_covered:
        action = 'rr_n'
    elif is_first_element_covered and not is_second_element_covered:
        action = 'lr_n'
    elif not is_first_element_covered and not is_second_element_covered:
        action = 'lr_n'
    
    return action


def get_action_sequence(words, ao_pairs):
    # first two elements as key, last element as value
    pair2sent = {(pair[0], pair[1]): pair[2] for pair in ao_pairs}


    stack = []
    target_pairs = ao_pairs
    buffer = list(range(0, len(words)))
    stack.append([0, 0]), stack.append([1, 1])
    buffer.remove(0), buffer.remove(1)
    actions = ['shift', 'shift']
    pairs = []
    states = []

    while len(buffer) > 0:
        if len(stack) < 2:
            first_ele = buffer.pop(0)
            stack.append([first_ele, first_ele])
            actions.append('shift')
        s_pair = (tuple(stack[-2]), tuple(stack[-1]))
        
        action = get_action(s_pair, target_pairs)
        actions.append(action)
        if action == 'rr':
            pair = (stack[-2], stack[-1])
            pair = (tuple(pair[0]), tuple(pair[1]))
            sent_label = pair2sent[pair]
        elif action == 'lr':
            pair = (stack[-1], stack[-2])
            pair = (tuple(pair[0]), tuple(pair[1]))
            sent_label = pair2sent[pair]
        else:
            sent_label = 3
        states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions), sent_label))

        pair = None
        if action == 'lr_n':
            stack.pop(-2)
        elif action == 'rr_n':
            stack.pop(-1)
        elif action == 'merge':
            end = stack.pop(-1)
            start = stack.pop(-1)
            stack.append([start[0], end[1]])
            # assert len(stack) == 1
        elif action == 'rr':
            pair = (stack[-2], stack[-1])
            stack.pop(-1)
        elif action == 'lr':
            pair = (stack[-1], stack[-2])
            stack.pop(-1)
        elif action == 'shift':
            first_ele = buffer.pop(0)
            stack.append([first_ele, first_ele])
        
        if pair is not None:
            pairs.append(pair)

    # assert len(stack) <= 2
    
    while len(stack) > 1:
        s_pair = (tuple(stack[-2]), tuple(stack[-1]))
        action = get_action(s_pair, target_pairs)
        actions.append(action)
        if action == 'rr':
            pair = (stack[-2], stack[-1])
            pair = (tuple(pair[0]), tuple(pair[1]))
            sent_label = pair2sent[pair]
        elif action == 'lr':
            pair = (stack[-1], stack[-2])
            pair = (tuple(pair[0]), tuple(pair[1]))
            sent_label = pair2sent[pair]
        else:
            sent_label = 3
        states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions), sent_label))

        pair = None
        if action == 'lr_n':
            stack.pop(-2)
        elif action == 'rr_n':
            stack.pop(-1)
        elif action == 'merge':
            end = stack.pop(-1)
            start = stack.pop(-1)
            stack.append([start[0], end[1]])
            # assert len(stack) == 1
        elif action == 'rr':
            pair = (stack[-2], stack[-1])
            stack.pop(-1)
        elif action == 'lr':
            pair = (stack[-1], stack[-2])
            stack.pop(-2)
        elif action == 'shift':
            stack.pop(-1)
        
        if pair is not None:
            pairs.append(pair)

    # transform actions in states to ids
    for i in range(len(states)):
        states[i] = (states[i][0], states[i][1], [action2id[action] for action in states[i][2]], states[i][3])

    return states, pairs

