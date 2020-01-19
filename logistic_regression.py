import scipy.io
import numpy as np

def sigmoid(features, weights):
	scores = np.dot(features, weights)
	return 1 / (1 + np.exp(-scores))


def findGradientAscent(features, target, predictions):
	error_value = target - predictions
	return np.dot(features.T, error_value)


def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
	if add_intercept:
		intercept = np.ones((features.shape[0], 1))
		features = np.hstack((intercept, features))

	weights = np.zeros(features.shape[1])

	for step in range(num_steps):
		predictions = sigmoid(features, weights)
		# Update weights with gradient
		gradientAscent = findGradientAscent(features, target, predictions)
		weights += learning_rate * gradientAscent

	return weights



#Starting main
#loading dataset
Numpyfile= scipy.io.loadmat('data/mnist_data.mat')

#separating training and testing data and labels
trainX = Numpyfile['trX']
trainY = Numpyfile['trY']
testX = Numpyfile['tsX']
testY = Numpyfile['tsY']

trainFeatures = []
#trainLabels = []
for i in range(len(trainX)):
	npImage = np.asarray(trainX[i])
	trainFeatures.append([npImage.mean(), npImage.std()])
	#trainLabels.append([trainY[i]])
trainFeatures = np.asarray(trainFeatures)
#trainLabels = np.asarray(trainLabels)
testFeatures = []
for image in testX:
	npImage = np.asarray(image)
	testFeatures.append([npImage.mean(), npImage.std()])
testFeatures = np.asarray(testFeatures)

#train Logistic regression
lgr_weights = logistic_regression(trainFeatures, np.asarray(trainY[0]), 400, 0.001, False)

#test logistic regression
predictions = sigmoid(testFeatures, lgr_weights)

lgr_correct7 = 0
lgr_correct8 = 0
total7 = 0
total8 = 0
for i in range(len(predictions)):
	if predictions[i] < 0.5:
		predictedClass = 0
	else:
		predictedClass = 1
	if predictedClass == testY[0][i]:
		if predictedClass == 0:
			lgr_correct7 += 1
		else:
			lgr_correct8 += 1
	if testY[0][i] == 0:
		total7 += 1
	else:
		total8 += 1

lgr_accuracy7 = lgr_correct7/total7
lgr_accuracy8 = lgr_correct8/total8
lgr_accuracy = (lgr_correct7 + lgr_correct8)/(total7 + total8)
print("Accuracy of Digit 7 in Logistic Regression: " + str(round(lgr_accuracy7 * 100, 2)) + "%")
print("Accuracy of Digit 8 in Logistic Regression: " + str(round(lgr_accuracy8 * 100, 2)) + "%")
print("Overall Accuracy in Logistic Regression: " + str(round(lgr_accuracy * 100, 2)) + "%")




