import pandas as pd
import numpy as np

def forward_selection(data):
    # col 0 is the label
    num_features = data.shape[1] - 1
    curr = [] #empty set

    # outer loop for the levels of the search tree
    for i in range(1, num_features + 1):
        print(f"On level {i} of the search tree")
        added_feature = None
        best = -1

        #inner loop to test each feature
        for j in range(1, num_features + 1):
        # check if feature is in the set
            if j not in curr:
                print(f"--Considering adding feature {j}")
            
                test_feature = curr + [j]
                accuracy = leave_one_out(data, test_feature)
                
                if accuracy > best:
                    best = accuracy
                    added_feature = j
        if added_feature != None:
            curr.append(added_feature)
            print(f"On level {i} added feature {added_feature} to current set(Accuracy: {best:.2%})\n")

    return curr

def backward_elimination(data):
    # col 0 is the label
    num_features = data.shape[1] - 1
    curr = list(range(1,num_features+1)) #start with all variables
    best_subset = []

    for i in range(curr):
        remove_feature = None
        best = -1

        for j in range(curr):
            test_feature = curr - [j]
            accuracy = leave_one_out(data, test_feature)
            if accuracy > best:
                best = accuracy
                remove_feature = j
        if remove_feature != None:
            best_subset.append(i)
    return best_subset



#leave one out cross-validation
# using nearest neighbor classifier 
def leave_one_out(data, subset):
    # correct used to keep track of objects correctly identified
    correct = 0
    size = len(data)

    labels = data[:,0]
    features = data[:, subset]

    #iterate through each object within the dataset
    for i in range(size):
        classify_object = features[i]
        label = labels[i]

        nearest_dist = float('inf')
        nearest_label = None

        #compares against ever other object
        #to find which is closest
        for j in range(size):
            if j != i:
                # euclidean distance
                dist = np.sqrt(np.sum((classify_object - features[j])**2))

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

    df = pd.read_csv(file, sep=None, engine='python')
    data = df.to_numpy()

    algorithm = input(
        "Type in the number of the algorithm you want to run:\n"
        "(1) Forward Section\n"
        "(2) Backward Elimination\n"
    )

    if (algorithm == "1"):
        forward_selection(data)

    elif(algorithm == 2):
        backward_elimination(data)

main()
