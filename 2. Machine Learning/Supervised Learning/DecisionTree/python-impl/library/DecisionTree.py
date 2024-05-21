import math
import random
import time
import pprint
from collections import defaultdict, deque

class Node:
    def __init__(self, is_leaf=None, distribution=None, node_id=None, attribute=None, threshold=None, gain=None):
        self.is_leaf = is_leaf
        self.inited = True if is_leaf is not None else False
        self.attribute = attribute
        self.node_id = node_id
        self.gain = gain
        self.threshold = threshold
        self.distribution = distribution if distribution is not None else {}
        self.left = None
        self.right = None

    def extend(self):
        if self.left is None:
            self.left = Node()
        if self.right is None:
            self.right = Node()


class DecisionTree:
    def __init__(self):
        self.forest = []
        self.pruning_threshold = 0

    def set_forest_size(self, tree_count):
        self.forest = [None] * tree_count

    def set_pruning_threshold(self, threshold):
        self.pruning_threshold = threshold

    def get_pruning_threshold(self):
        return self.pruning_threshold

    def information_gain(self, examples, attribute, threshold):
        num_examples = len(examples)
        left_child = [x for x in examples if x[attribute] < threshold]
        right_child = [x for x in examples if x[attribute] >= threshold]

        num_left = len(left_child)
        num_right = len(right_child)

        def frequency_count(vec):
            freq = defaultdict(int)
            for x in vec:
                freq[int(x[-1])] += 1
            return freq

        main_node_freq = frequency_count(examples)
        left_node_freq = frequency_count(left_child)
        right_node_freq = frequency_count(right_child)

        def entropy(freq, total):
            return sum(-count / total * math.log2(count / total) for count in freq.values() if count > 0)

        h = entropy(main_node_freq, num_examples)
        hl = entropy(left_node_freq, num_left) if num_left > 0 else 0
        hr = entropy(right_node_freq, num_right) if num_right > 0 else 0
        pl = num_left / num_examples
        pr = num_right / num_examples

        return h - (hl * pl + hr * pr)

    def distribution(self, examples):
        freq_count = defaultdict(int)
        for x in examples:
            freq_count[int(x[-1])] += 1

        total = len(examples)
        return {k: v / total for k, v in freq_count.items()}

    def min_max(self, attribute, examples):
        vals = [x[attribute] for x in examples]
        return min(vals), max(vals)

    def get_best_outcome(self, attribute, examples):
            best_attribute, best_gain, best_threshold = -1, -1, -1
            min_val, max_val = self.min_max(attribute, examples)
            for k in range(1, 51):
                threshold = min_val + k * (max_val - min_val) / 51
                gain = self.information_gain(examples, attribute, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = attribute
                    best_threshold = threshold

            return best_attribute, best_threshold, best_gain

    def choose_attribute(self, examples, attributes, is_optimized):
        best_attribute = -1
        best_gain = -1
        best_threshold = -1
        if is_optimized:
            for attribute in attributes:
                _attribute, _threshold, _gain = self.get_best_outcome(attribute, examples)
                if _gain > best_attribute:
                    best_attribute, best_threshold, best_gain = _attribute, _threshold, _gain
        else:
            attribute = random.choice(attributes)
            best_attribute, best_threshold, best_gain = self.get_best_outcome(attribute, examples)
        

        return best_attribute, best_threshold, best_gain

    def all_same_class(self, examples):
        classes = set([int(x[-1]) for x in examples])
        return len(classes) == 1

    def build_decision_tree(self, curr: Node, examples, attributes, pruning_threshold, is_optimized, node_id):
        
        if self.all_same_class(examples) or len(examples) < pruning_threshold:
            curr = Node(True, self.distribution(examples), node_id)
            return curr

        best_attribute, best_threshold, best_gain = self.choose_attribute(examples, attributes, is_optimized)

        curr = Node(False, None, node_id, best_attribute, best_threshold, best_gain)
        curr.extend()

        left_child_elem = [x for x in examples if x[best_attribute] < best_threshold]
        right_child_elem = [x for x in examples if x[best_attribute] >= best_threshold]

        if len(left_child_elem) > 0:
            curr.left = self.build_decision_tree(curr.left, left_child_elem, attributes, pruning_threshold, is_optimized, node_id * 2)
        if len(right_child_elem) > 0:
            curr.right = self.build_decision_tree(curr.right, right_child_elem, attributes, pruning_threshold, is_optimized, node_id * 2 + 1)

        return curr

    def train(self, data_file, option):
        examples, attrs = self.prepare_data(data_file)

        root_id = 1
        if option == "optimized":
            self.set_forest_size(1)
            self.forest[0] = self.build_decision_tree(None, examples, attrs, self.get_pruning_threshold(), True, root_id)
        elif option == "randomized":
            self.set_forest_size(1)
            self.forest[0] = self.build_decision_tree(None, examples, attrs, self.get_pruning_threshold(), False, root_id)
        elif option == "forest3":
            self.set_forest_size(3)
            for i in range(3):
                self.forest[i] = self.build_decision_tree(None, examples, attrs, self.get_pruning_threshold(), False, root_id)
        elif option == "forest15":
            self.set_forest_size(15)
            for i in range(15):
                self.forest[i] = self.build_decision_tree(None, examples, attrs, self.get_pruning_threshold(), False, root_id)

    def get_predicted_distribution(self, curr, test_obj):
        if curr.is_leaf:
            return curr.distribution
        if curr.left and curr.left.inited and test_obj[curr.attribute] < curr.threshold:
            return self.get_predicted_distribution(curr.left, test_obj)
        if curr.right and curr.right.inited:
            return self.get_predicted_distribution(curr.right, test_obj)

    def classify_object(self, test_obj, obj_id, out):
        random.seed(int(time.time())) 

        final_distribution = defaultdict(float)
        distributions = [self.get_predicted_distribution(tree, test_obj) for tree in self.forest]

        num_distributions = len(distributions)
        for distribution in distributions:
            for k, v in distribution.items():
                final_distribution[k] += v / num_distributions

        max_probability = max(final_distribution.values())
        predicted_class = max(final_distribution, key=final_distribution.get)

        tied_elements = [k for k, v in final_distribution.items() if v == max_probability]
        
        accuracy = 0

        if len(tied_elements) == 1:
            accuracy = 1.0 if predicted_class == int(test_obj[-1]) else 0.0
        else:
            accuracy = 1.0 / len(tied_elements) if int(test_obj[-1]) in tied_elements else 0.0
            predicted_class = random.choice(tied_elements)

        self.print_test_result(out, obj_id, predicted_class, int(test_obj[-1]), accuracy)

        return accuracy, accuracy > 0.0

    def test_data_prediction(self, data_file, output_file):
        test_data, _ = self.prepare_data(data_file)

        classification_accuracy = 0.0
        correct_prediction = 0

        with open(output_file, 'w') as out:
            for obj_id, test_obj in enumerate(test_data):
                acc, corr_pred = self.classify_object(test_obj, obj_id, out)
                classification_accuracy += acc
                correct_prediction += corr_pred

            classification_accuracy /= len(test_data)
            out.write(f"classification accuracy={classification_accuracy:.4f}\n")
            print(f"classification accuracy={classification_accuracy:.4f}")

            incorrect_prediction = len(test_data) - correct_prediction
            out.write(f"Correct Predictions = {correct_prediction}, Incorrect Predictions = {incorrect_prediction}\n")
            print(f"Correct Predictions = {correct_prediction}, Incorrect Predictions = {incorrect_prediction}")

    def prepare_data(self, data_file):
        examples = []
        with open(data_file, 'r') as in_file:
            for line in in_file:
                row_values = list(map(float, line.strip().split()))
                examples.append(row_values)

            attrs = range(0, len(examples[0]) - 1)
        return examples, attrs

    def print_test_result(self, out, obj_id, predicted, true, accuracy):
        result_str = (f"ID={obj_id}, predicted={predicted}, true={true}, "
                      f"accuracy={accuracy}\n")
        out.write(result_str)
        print(result_str)

    def print_formatted(self, out, tree_id, curr, is_leaf):
        node_id = curr.node_id
        attribute = -1 if is_leaf else curr.attribute
        threshold = -1.0 if is_leaf else curr.threshold
        gain = 0.0 if is_leaf else curr.gain

        result_str = (f"tree={tree_id:2d}, node={node_id:90s}, feature={attribute:2d}, "
                      f"thr={threshold:6.2f}, gain={gain:24.20f}\n")
        out.write(result_str)
        print(result_str)

    def print_decision_tree(self, output_file):
        with open(output_file, 'w') as out:
            for tree_id, curr_tree in enumerate(self.forest):
                q = deque([curr_tree])

                while q:
                    curr = q.popleft()
                    self.print_formatted(out, tree_id, curr, curr.is_leaf)

                    if curr.left and curr.left.inited:
                        q.append(curr.left)
                    if curr.right and curr.right.inited:
                        q.append(curr.right)


# Example usage:
# tree = DecisionTree()
# tree.set_pruning_threshold(5)
# tree.train('../../data/pendigits_training.txt', 'randomized')
# tree.test_data_prediction('../../data/pendigits_test.txt', 'output.txt')
# tree.print_decision_tree('tree_output.txt')
