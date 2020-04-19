from nbc import nbc
from nbc import message


# to use naive bayes classifier 
# needs to create an object of nbc class
run = nbc()

path = input("\nProvide dataset file path: ")

# First Read dataset
run.readDataset(path)
print("")

# check if data is properly loaded
# run.printDataset()

# shuffle dataset before applying prediction
run.shuffleDataset()

ans = input("Do you want to view shuffled dataset (y/n): ")
if ans == 'y':
    print(" '''''' Check Shuffled dataset ''''' ")
    run.printDataset()
    print(" +++++++++++++++ END +++++++++++++++++ \n")

# now lets split dataset into training and testing
# the parameters passing to this methods are:
# training_percentage and testing_percentage
run.splitDataset(90, 10)

# provide inputs for your model
print("\n+++++++++++ Provide inputs for your model ++++++++++++")
inputs = run.getInputs()

sms = run.predictClass(inputs)
# it will give you the predicted class
# using given inputs

for i in range(0, len(sms.cls)):
    print("Class '",sms.cls[i],"' have " ,sms.prob[i]," probability")
print("Hence,")
print("Predicted Class is '",sms.preclass,"' with highest probability ", sms.preprob)
print("")


