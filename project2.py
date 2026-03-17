import pandas as pd
import numpy as np
import time

def forward_selection(data):
    # col 0 is the label
    num_features = data.shape[1] - 1
    curr = [] #empty set
    best_accuracy = -1
    best_subset = []
    
    print("Beginning search.")
    # outer loop for interating through the features
    for i in range(1, num_features + 1):
        added_feature = None
        best = -1

        #inner loop to test and add all other features
        for j in range(1, num_features + 1):
        # check if feature is in the set
            if j not in curr:
            
                test_feature = curr + [j]
                accuracy = leave_one_out(data, test_feature)
                print(f"Using feature(s) {set(test_feature)} accuracy is {accuracy*100:.1f}%")
                
                if accuracy > best:
                    best = accuracy
                    added_feature = j
        if added_feature != None:
            curr.append(added_feature)
        if best < best_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        elif best > best_accuracy:
            best_accuracy = best
            best_subset = curr.copy()

        print(f"Feature set {set(curr)} was best, accuracy is {best*100:.1f}%\n")

    return best_subset, best_accuracy

def backward_elimination(data):
    # col 0 is the label
    num_features = data.shape[1] - 1
    curr = list(range(1,num_features+1)) #start with all features
    # best subset initialized with all features
    best_subset = curr.copy()
    best_accuracy = leave_one_out(data, curr)

    print("Beginning search.\n")
    #remove features til only one is left
    while len(curr) > 1:
        remove_feature = None
        best = -1

        #removing each feature in current set
        for feature in curr:
            test_feature = curr.copy()
            test_feature.remove(feature)
            accuracy = leave_one_out(data, test_feature)

            print(f"Using feature(s) {set(test_feature)} accuracy is {accuracy*100:.1f}%")
            
            if accuracy > best: # track if feature removal gives a higher accuracy
                best = accuracy
                remove_feature = feature

        if remove_feature != None:
            curr.remove(remove_feature)

        if best < best_accuracy:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")

        #if this subset is better than the best so far, store it
        if best > best_accuracy:
            best_accuracy = best
            best_subset = curr.copy()

        print(f"Feature set {set(curr)} was best, accuracy is {best*100:.1f}%\n")

    return best_subset, best_accuracy

#leave one out cross-validation
# using nearest neighbor classifier 
def leave_one_out(data, subset):
    # correct used to keep track of features correctly identified
    correct = 0
    size = len(data)

    labels = data[:,0]
    features = data[:, subset]

    #iterate through each feature within the dataset
    for i in range(size):
        classify_object = features[i]
        label = labels[i]

        nearest_dist = float('inf')
        nearest_label = None

        #compares against ever other feature
        #to find which is closest
        for j in range(size):
            if j != i:
                # euclidean distance
                dist = np.sum((classify_object - features[j])**2)

                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_label = labels[j]
        if label == nearest_label: #found nearest neighbor
            correct +=1
    return correct/size # return accuracy percentage

def main():
    print("Welcome to Motunrayo Lawrence's Feature Selection Algorithm!")

    file = input (
        "Type in the name of the file to test:\n"
    )

    df = pd.read_csv(file, sep=r'\s+', header=None, engine='python')
    data = df.to_numpy()

    num_features = data.shape[1] - 1
    num_instances = data.shape[0]

    print(f"\nThis dataset has {num_features} features (not including the class attribute), with {num_instances} instances.")
    start_time = time.time() # start timer

    all_features = list(range(1, num_features + 1))
    accuracy = leave_one_out(data, all_features)

    print(f'Running nearest neighbor with all {num_features} features, using "leaving-one-out" evaluation, I get an accuracy of {accuracy*100:.1f}%\n')

    algorithm = input(
        "Type in the number of the algorithm you want to run:\n"
        "(1) Forward Section\n"
        "(2) Backward Elimination\n"
    )

    if (algorithm == "1"):
        result, final_accuracy = forward_selection(data)

    elif(algorithm == "2"):
        result, final_accuracy = backward_elimination(data)

    end_time = time.time() #end timer
    elapsed = end_time - start_time

    print(f"Finished search!! The best feature subset is {set(result)}, which has an accuracy of {final_accuracy*100:.1f}%")
    print(f"Total computation time: {elapsed:.2f} seconds")

main()
