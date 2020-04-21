import csv
import sys
import random

class LIST:
    def __init__(self):
        self.lst = []
        super().__init__()
    def addToList(self, val):
        if val in self.lst:
            return
        # Sorted Insertion List
        # size = len(self.lst)
        # for i in range(0, size):
        #     prev = i
        #     if self.lst[i] > val:
        #         self.lst.insert(prev, val)
        #         return
        self.lst.append(val)
        return
    def getLength(self):
        return len(self.lst)


class message(object):
    def __init__(self):
        self.cls = []
        self.prob = []
        self.preclass = ''
        self.preprob = 0.0
        super().__init__()

    def create(self, cls, prob):
        self.cls.append(cls)
        self.prob.append(prob)

    def predClass(self, c, p):
        self.preclass = c
        self.preprob = p


class nbc:

    def __init__(self):
        """
        setting Class Attrributes
        """

        self.__dataset = []
        self.__dataset_len = 0
        self.__classes = []
        self.__classes_len = 0
        self.__filepath = ''
        self.__testing = []
        self.__training = []
        self.__datasetVars = LIST()

        super().__init__()

    def __getUniqueValues(self):
        """
        this function will call when dataset is readed
        and its extracting all the integer which occurs in dataset
        """

        for data in self.__dataset:
            for x in data[0:len(data)-1]:
                self.__datasetVars.addToList(x)

    def readDataset(self, path):
        """
        Read data from csv file
        and extract classes from it
        """

        self.__filepath = path
        # open file eg: "data/iris.csv"
        try:
            lines = csv.reader(open(self.__filepath, 'r+'))
        except IOError:
            print("Could not open file! Path Not Found!")
            sys.exit()

        f = open('data/classes.csv', 'w+')
        self.__dataset = list(lines)
        for i in range(len(self.__dataset)):
            attr = []
            for x in self.__dataset[i][0:len(self.__dataset[i])-1]:
                if len(x) > 0:
                    attr.append(x)
                else:
                    attr.append('?')
            classs = self.__dataset[i][len(self.__dataset[i])-1]
            if classs not in self.__classes:
                self.__classes.append(classs)
                f.write(classs+'\n')

            self.__dataset[i] = attr
            self.__dataset[i].append(classs)
            del attr
        
        # save new length lists
        self.__dataset_len = len(self.__dataset)
        self.__classes_len = len(self.__classes)

        # extract all the distinct values from dataset
        # which is used in it
        self.__getUniqueValues()
        print("\ndata is loaded successfully ...")

    def printDataset(self):
        """
        it will print the dataset with it's classes
        """

        print(f'Your dataset from path "{self.__filepath}" is shown below: ')
        print(*self.__dataset, sep='\n')
        print("")
        print('Your dataset classes are shown below: ')
        print(*self.__classes, sep='\n')
        print("")

    def getInputs(self):
        """ 
        it will get inputs according to the number 
        attrributes in dataset
        """

        inputs = []
        for i in range(len(self.__dataset[0][0:len(self.__dataset[0])-1])):
            inputs.append(input(f"Enter attribute {i+1} value: \t"))
        print("")
        return inputs

    def shuffleDataset(self):
        random.shuffle(self.__dataset)
        print("data is shuffled ....\n")

    def splitDataset(self, training_perc, testing_perc):
        """
        it will split the dataset into
        training set and testing set
        according to paramenters setting
        """

        train_count = (training_perc/100)*self.__dataset_len
        test_count = (testing_perc/100)*self.__dataset_len

        print(f"\nTraining Samples: {train_count}/{self.__dataset_len}\n")
        print(f"Testing Samples: {test_count}/{self.__dataset_len}\n")


        self.__training = self.__dataset[0:int(train_count)]
        self.__testing = self.__dataset[int(train_count):self.__dataset_len]

        # delete classes and add training classes in it
        del self.__classes
        self.__classes = []
        for x in self.__training:
            if x[len(x)-1] not in self.__classes:
                self.__classes.append(x[len(x)-1])
        self.__classes_len = len(self.__classes)

    def getOccurances(self, cls=None, input_val=None, attr_index=None):

        if input_val == None and attr_index == None:
            return len([x for x in self.__training if cls in x[-1]])
        
        if cls == None:
            return len([x for x in self.__training if x[attr_index] == input_val])
        
        return len([x for x in self.__training if cls in x[-1] and x[attr_index] == input_val])

    def findLikelihood(self, cls, inputs):
        result, index = 1.0, 0
        
        # Get total occurance of a Class
        # P(cls)
        total = self.getOccurances(cls)


        # print(f"Given Class '{cls}' have occurences {total}")

        # Finding P(attr=inputs | output = cls )
        for index, inputt in enumerate(inputs):

            # P(cls | inputt) 
            likliehood = self.getOccurances(cls, inputt, index)
            # P(input1)
            prior = self.getOccurances(input_val=inputt, attr_index=index)
            # (P(cls | inputt) * P(inputt))/P(cls)
            ans = (likliehood*prior) / total

            # if prior become 0
            if ans == 0:

                # (a + m*p)/(b + m) <= m-estimation formula
                # where p=1/t and t is total number of values
                p = (1/self.__datasetVars.getLength())
                ans = (0 + (len(self.__training)*p))/(total+len(self.__training))
            result *= ans

        result *= (total/len(self.__training))
        return result

    def predictClass(self, inputs):
        """
        it will provide the prediction of inputed features 
        """
        
        predictedClass = ''
        sms = message()
        max = 0.0
        liklihoods = []

        # below loop will sending classes, inputs and dataset
        for cls in self.__classes:
            liklihoods.append(self.findLikelihood(cls, inputs))
        
        for clsIn in range(self.__classes_len):
            result = liklihoods[clsIn]/sum(liklihoods)
            sms.create(self.__classes[clsIn], result*100)
            if result > max:
                max = result
                predictedClass = self.__classes[clsIn]

        sms.predClass(predictedClass, max*100)
        return sms

    def getPredictions(self):
        """
        it will provide the predictions of testing data
        """

        predictions = []

        for index, inputt in enumerate(self.__testing):
            predictions.append(self.predictClass(inputt[:len(inputt)-1]).preclass)
            print(f"For Input {index+1} prediction is: {predictions[index]}")

        return predictions

    def getAccuracy(self, predictions):
        """
        it will calculate the accuracy on the basis of testing dataset
        """

        correct = 0

        for i, cls in enumerate(self.__testing):
            if cls[-1] == predictions[i]:
                correct += 1
                print(f"Accuracy until now: {(correct/float(i)*100.0)}")
        
        return (correct/float(len(self.__testing)))*100.0
