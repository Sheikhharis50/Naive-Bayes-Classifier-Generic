from nbc import nbc
from nbc import message
import sys, os
from time import sleep

# to use naive bayes classifier 
# needs to create an object of nbc class
run = nbc()

#path = input("\nProvide dataset file path: ")

# First Read dataset
run.readDataset('data/iris.csv')
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

while True:
    print('++++++++++++ MENU ++++++++++++++')
    print('1) \tGive Custom inputs')
    print('2) \tUse testing dataset as input')
    print('0) \tExit')
    print('++++++++++++++++++++++++++++++++')
    choice = int(input('Enter your choice: '))

    if choice == 0: sys.exit()

    elif choice == 1:

        # provide inputs for your model
        print("\n+++++++++++ Provide inputs for your model ++++++++++++")
        inputs = run.getInputs()
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

        # sms = run.predictClass(inputs)
        sms = run.predictClass(inputs)
        # it will give you the predicted class
        # using given inputs

        for i in range(0, len(sms.cls)):
            print("Class '",sms.cls[i],"' have " ,sms.prob[i]," probability")
            print("Hence,")
            print("Predicted Class is '",sms.preclass,"' with highest probability ", sms.preprob)
            print("")

    elif choice == 2:

        predictions = run.getPredictions()
        acc = run.getAccuracy(predictions)
        print("Accuracy: {0}%\n".format(acc))

    else:
        print("\nWrong Choice, Enter again\n")
        continue

    input("Press Enter to continue...")

    try:
        os.system('clear')
    except:
        os.system('cls')


