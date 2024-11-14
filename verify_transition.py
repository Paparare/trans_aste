from copy import deepcopy
import os
from transformers import BertTokenizer
# with open('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt', 'r') as f:
#     lines = f.readlines()
# line: The gourmet food is delicious but the service is poor####[([1, 2], [4], 'POS'), ([7], [9], 'NEG')]



# for line in lines:
#     line = line.strip()
#     sentence, triplets = line.split('####')
#     triplets = eval(triplets)

def is_span_covered(s, span_set):
    for span in span_set:
        if span[0] <= s[0] and span[1] >= s[1]:
            return True
    return False

def get_action(s_pair, pairs):
    action = None

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
        states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions)))

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
        states.append((deepcopy(stack), deepcopy(buffer), deepcopy(actions)))

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

    return states, pairs


# sentence = "The gourmet food is delicious but the service is poor"
# triplets = [([1, 2], [4], 'POS'), ([7], [9], 'NEG')]

# sentence = "The good gourmet food ."
# triplets = [([2, 3], [1], 'POS')]

# sentence = "The gourmet food is good ."
# triplets = [([1, 2], [4], 'POS')]
# pair_tuple_set = set()
# for triplet in triplets:
#     if len(triplet[0]) == 1: 
#         aspect = triplet[0] * 2
#     else:
#         aspect = triplet[0]
#     if len(triplet[1]) == 1: 
#         opinion = triplet[1] * 2
#     else:
#         opinion = triplet[1]
#     pair_tuple_set.add((tuple(aspect), tuple(opinion)))
# actions, pairs = get_action_sequence(sentence.split(), pair_tuple_set)

def eval_metric(truth_pairs_list, extracted_pairs_list):
    tp = 0
    fp = 0
    fn = 0
    for truth_pairs, extracted_pairs in zip(truth_pairs_list, extracted_pairs_list):
        tp += len(truth_pairs & extracted_pairs)
        fp += len(extracted_pairs - truth_pairs)
        fn += len(truth_pairs - extracted_pairs)
    precision = tp / (tp + fp + 1e-8)  
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


def cal_coverage_degree(dataset_path, dataset_name):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(dataset_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    truth_pairs_list = []
    extracted_pairs_list = []
    for line in lines:
        sentence, triplets = line.split('####')
        triplets = eval(triplets)
        pair_tuple_set = set()
        for triplet in triplets:
            if len(triplet[0]) == 1: 
                aspect = triplet[0] * 2
            else:
                aspect = triplet[0]
            if len(triplet[1]) == 1: 
                opinion = triplet[1] * 2
            else:
                opinion = triplet[1]

            if len(aspect) > 2:
                aspect = (aspect[0], aspect[-1])
            if len(opinion) > 2:
                opinion = (triplet[1][0], triplet[1][-1])
            pair_tuple_set.add((tuple(aspect), tuple(opinion)))

        sent_text = sentence
        truth_pairs = pair_tuple_set
        orig_pos2bert_pos = {}
        sent_tokens = sent_text.split(' ')
        sent_tokens_for_bert = []
        for orig_pos, token in enumerate(sent_tokens):
            bert_tokens = tokenizer.tokenize(token)
            cur_len = len(sent_tokens_for_bert)

            orig_pos2bert_pos[orig_pos] = (cur_len, cur_len+len(bert_tokens)-1)
            sent_tokens_for_bert += bert_tokens
        
        truth_pairs_for_bert = []
        for pair in truth_pairs:
            source_span = pair[0]
            target_span = pair[1]
            source_span_for_bert = (orig_pos2bert_pos[source_span[0]][0], orig_pos2bert_pos[source_span[1]][1])
            target_span_for_bert = (orig_pos2bert_pos[target_span[0]][0], orig_pos2bert_pos[target_span[1]][1])
            truth_pairs_for_bert.append((source_span_for_bert, target_span_for_bert))
        truth_pairs_for_bert_set = set(truth_pairs_for_bert)

        actions, pairs = get_action_sequence(sent_tokens_for_bert, truth_pairs_for_bert_set)
        # actions, pairs = get_action_sequence(sentence.split(), pair_tuple_set)
        pairs = [(tuple(pair[0]), tuple(pair[1])) for pair in pairs]
        truth_pairs_list.append(truth_pairs_for_bert_set)
        extracted_pairs_list.append(set(pairs))

    pre, rec, f1 = eval_metric(truth_pairs_list, extracted_pairs_list)
    print("Dataset: ", dataset_name)
    print("precision: ", pre)
    print("recall: ", rec)
    print("f1: ", f1)
    print()

cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/train_triplets.txt', '14lap_train')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/train_triplets.txt', '14res_train')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/train_triplets.txt', '15res_train')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/train_triplets.txt', '16res_train')

cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/dev_triplets.txt', '14lap_dev')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/dev_triplets.txt', '14res_dev')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/dev_triplets.txt', '15res_dev')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/dev_triplets.txt', '16res_dev')

cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/test_triplets.txt', '14lap_test')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/test_triplets.txt', '14res_test')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/test_triplets.txt', '15res_test')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/test_triplets.txt', '16res_test')

cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/total_triplets.txt', '14lap_total')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/total_triplets.txt', '14res_total')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/total_triplets.txt', '15res_total')
cal_coverage_degree('SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/total_triplets.txt', '16res_total')


# import os
# main_path = 'SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res'
# total_dataset = []
# for sp in ['train', 'dev', 'test']:
#     with open(os.path.join(main_path, sp + '_triplets.txt'), 'r') as f:
#         lines = [line.strip() for line in f.readlines()]
#     total_dataset.extend(lines)

# with open(os.path.join(main_path, 'total_triplets.txt'), 'w') as f:
#     for line in total_dataset:
#         f.write(line + '\n')







