from Bio import SeqIO
import numpy as np

from sklearn.linear_model import LogisticRegression


def oneHot(seq):
	'''
	returns a one-hot encoding np array of the sequence, flattened.
	'''
	encodings = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
	alphaD = {"A": 0, "C": 1, "G": 2, "T":3}
	return encodings[[alphaD[x] for x in seq], :].flatten()

def ordinal(seq):
	'''
	returns an ordinal encoding np array of the sequence.
	'''
	D = {"A":0.25, "C":0.5, "G":0.75, "T":1}
	return np.array([D[x] for x in seq])

def loadData():
	'''
	returns a list of true positive strings and a list of other strings
	to pull true negatives from
	'''
	trueSeqs = []
	for line in open("rap1-lieb-positives.txt", "r"):
		trueSeqs.append(line.rstrip())


	fasta_sequences = SeqIO.parse(open("yeast-upstream-1k-negative.fa"),'fasta')
	fullSeqs = [str(x.seq) for x in fasta_sequences if len(str(x.seq)) == 1000]
	return trueSeqs, fullSeqs



# sample the same number as the number of true sequences
def getFalseSeqs(trueSeqs, n, fullSeqs): 
	'''
	generates a set of false sequences.

	input: trueSeqs, a list of true sequence strings, and fullSeqs, a
	list of null sequence strings, *each the same length as 
	fullSeqs[0]*.

	output: a list with n elements, each a len(trueSeqs)[0] substring
	of a random fullSeqs string that is not in trueSeqs.
	'''
	seqLen = len(trueSeqs[0]) # should always be 17.
	# first choose which sequences
	theseSeqs = np.random.choice(fullSeqs, n, replace=False)
	# then choose where the substrings will start
	indices = np.random.randint(0, len(fullSeqs[0])-seqLen, n)
	for seq, i in zip(theseSeqs, indices):
		if len(seq[i:i+seqLen]) < seqLen:
			print(i, seq)
	falseSeqs = [seq[i:i+seqLen] for seq,i in zip(theseSeqs, indices)]
	# ensure we don't pick a trueSeq on accident
	for seq in falseSeqs:
		if seq in trueSeqs:
			return(getFalseSeqs(trueSeqs, n, fullSeqs))
	return falseSeqs


# Okay, we're ready to rock.

def removeHoldout(trueSeqs, holdoutSize=len(trueSeqs)//10 + 1):
	'''
	pulls out holdoutSize samples and returns the rest, and 
	then the holdout.
	'''
	trainSize = len(trueSeqs) - holdoutSize # for us, 123
	np.random.shuffle(trueSeqs)
	return np.split(trueSeqs, [holdoutSize])


trueSeqs, fullSeqs = loadData()

trueHoldoutSeqs, trueTrainSeqs = removeHoldout(trueSeqs)


def bagLR(trueTrainSeqsSubset, encodingF=oneHot, regularization='l2', regStrength=1):
	# we do bagging on the true sequences to train the model. custom-bake
	# it so we can use (potentially) new false sequences each time.
	# In each iteration, we get a bootstrap sample of our true positive
	# data and a new sample of true negative data to train a logistic
	# regression model.
	trainSize = len(trueTrainSeqsSubset)
	coeffs = []
	intercepts = []
	classes = []
	for i in range(100):
		# get a bootstrap sample of the true positives
		thisTrainSeqs = np.random.choice(trueTrainSeqsSubset, size=trainSize, replace=True)
		falseSeqs = getFalseSeqs(trueSeqs, trainSize, fullSeqs)
		sequences = np.append(thisTrainSeqs, falseSeqs)
		labels = np.append(np.ones(trainSize), np.zeros(trainSize))
		thisLr = LogisticRegression(penalty=regularization, C=regStrength)
		thisLr.fit([encodingF(x) for x in sequences], labels)
		coeffs.append(thisLr.coef_)
		intercepts.append(thisLr.intercept_)
		if len(classes) == 0:
			classes = thisLr.classes_
	# then we average coefficients between models.
	avgLr = LogisticRegression(penalty=regularization, C=regStrength)
	avgLr.coef_ = np.average(coeffs, axis=0)
	avgLr.intercept_ = np.average(intercepts)
	avgLr.classes_ = classes
	return avgLr, coeffs, intercepts, classes

# I do something like k-fold stratified cross-validation here.
# Again, I want to make sure I'm always using fresh false data since
# there's so much. 
def determineAccuracyInCrossValidation(k=8, encodingF=oneHot, regularization='l2', regStrength=1)
	scores = []
	scores_holdout = []
	seqs_test_holdout = np.append(trueHoldoutSeqs, getFalseSeqs(trueSeqs, len(trueHoldoutSeqs), fullSeqs))
	labels_test_holdout = np.append(np.ones(len(trueHoldoutSeqs)), np.zeros(len(trueHoldoutSeqs)))
	for test_i in np.array_split(np.random.permutation(range(len(trueTrainSeqs))), k):
		trueTrainSeqsSubset = np.delete(trueTrainSeqs, test_i)
		seqs_train = np.append(trueTrainSeqsSubset, getFalseSeqs(trueSeqs, len(trueTrainSeqsSubset), fullSeqs))
		labels_train = np.append(np.ones(len(trueTrainSeqsSubset)), np.zeros(len(trueTrainSeqsSubset)))
		trueTestSeqsSubset = trueTrainSeqs[test_i]
		seqs_test = np.append(trueTestSeqsSubset, getFalseSeqs(trueSeqs, len(trueTestSeqsSubset), fullSeqs))
		labels_test = np.append(np.ones(len(trueTestSeqsSubset)), np.zeros(len(trueTestSeqsSubset)))
		lr, coeffs, intercepts, classes = bagLR(trueTrainSeqsSubset, encodingF=oneHot, regularization='l2', regStrength=1)
		scores.append(lr.score([encodingF(x) for x in seqs_test], labels_test))
		scores_holdout.append(lr.score([encodingF(x) for x in seqs_test_holdout], labels_test_holdout))

def plotAccuracy(scores, scores_holdout):
	fig, ax = plt.subplots()
	r1 = ax.bar(np.arange(len(scores)), scores, width=1/3, color="royalblue")
	r2 = ax.bar(np.arange(len(scores)) + 1/3, scores_holdout, width=1/3, color="seagreen")
	ax.legend((r1[0], r2[0]), ('x-val k holdout', '1/10 blind holdout'), loc=8)
	plt.xlabel("cross-validation iteration")
	plt.ylabel("accuracy")
	plt.show()


def tryLessIterations():
	# see if 100 iterations is enough:
	scores = []
	lr, coeffs, intercepts, classes = bagLR(trueTrainSeqs)
	seqs_test_holdout = np.append(trueHoldoutSeqs, getFalseSeqs(trueSeqs, len(trueHoldoutSeqs), fullSeqs))
	labels_test_holdout = np.append(np.ones(len(trueHoldoutSeqs)), np.zeros(len(trueHoldoutSeqs)))
	for i in range(1, 101):
		avgLr = LogisticRegression()
		avgLr.coef_ = np.average(coeffs[:i], axis=0)
		avgLr.intercept_ = np.average(intercepts[:i])
		avgLr.classes_ = classes
		scores.append(sum(abs(avgLr.predict_proba([oneHot(x) for x in seqs_test_holdout])[:,1] - labels_test_holdout)))

	import matplotlib.pyplot as plt
	plt.plot(scores)
	plt.xlabel("number of bootstraps aggregated")
	plt.ylabel("sum of errors")
	plt.show()


# predict on unlabelled data. For this, I train on all our data.
lr, coeffs, intercepts, classes = bagLR(trueSeqs)
unkSeqs = []
for line in open("rap1-lieb-test.txt", "r"):
	unkSeqs.append(line.rstrip())
unkProbs = lr.predict_proba([oneHot(x) for x in unkSeqs])[:,1]
with open("ap1-lieb-predictions.txt", "w") as f:
	for seq, prob in zip(unkSeqs, unkProbs): 
		f.write("\t".join(map(str, [seq, prob])) + "\n")


