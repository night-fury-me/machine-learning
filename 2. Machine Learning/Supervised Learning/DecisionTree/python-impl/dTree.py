import sys
from library.DecisionTree import DecisionTree

def solve_task(number_of_arguments, arguments):
    if number_of_arguments != 5:
        print("Incorrect number of arguments, please recheck!")
        return

    training_file = arguments[1]
    test_data_file = arguments[2]
    training_type = arguments[3]
    pruning_thr = arguments[4]

    try:
        pruning_threshold = int(pruning_thr)
    except ValueError:
        print("Pruning threshold must be an integer!")
        return

    valid_training_types = ["optimized", "randomized", "forest3", "forest15"]
    if training_type not in valid_training_types:
        print("Training Type is not valid!")
        return

    d_tree = DecisionTree()
    d_tree.set_pruning_threshold(pruning_threshold)
    d_tree.train(training_file, training_type)
    #d_tree.print_forest("Forest.txt")
    d_tree.test_data_prediction(test_data_file, "testResult.txt")

if __name__ == "__main__":
    solve_task(len(sys.argv), sys.argv)
