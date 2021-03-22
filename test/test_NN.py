import pytest
import numpy as np
from scripts import NN

def test_encoder():

	# test that each letter is being properly encoded
	assert (NN.encode("A") == [1,0,0,0]).all()
	assert (NN.encode("C") == [0,1,0,0]).all()
	assert (NN.encode("G") == [0,0,1,0]).all()
	assert (NN.encode("T") == [0,0,0,1]).all()

	# test that the encoder is giving back the right number of values
	assert (len(NN.encode("acgt")) == 16)

	# test that a sequence returns the right encoded values
	assert ((NN.encode("acgt") == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).all())

def test_read_fasta():

	# read in fasta files
	pos = NN.read_fasta("./data/rap1-lieb-positives.txt")
	neg = NN.read_fasta("./data/yeast-upstream-1k-negative.fa")

	# testing the first and last sequences were read in correctly
	assert pos[0] == 'ACATCCGTGCACCTCCG'
	assert pos[-1] == 'ACACCCATACACCAAAC'

	# testing that the number of negative sequences were read in correctly
	# I would do the same thing as the the positive samples but the sequences are too long
	assert len(neg.keys()) == 3147

	# testing that the sequences read in are the correct length
	assert len(list(neg.keys())[0]) == 1000
	assert len(list(neg.keys())[-1]) == 1000

	# testing that the sequences were mapped to the correct fasta header
	assert neg[list(neg.keys())[0]] == ">YAL003W 5' untranslated region, chrI 141172 - 142171, 1000 bp"
	assert neg[list(neg.keys())[-1]] == ">YPR202W 5' untranslated region, chrXVI 942027 - 943026, 1000 bp"
	
def test_init_model():
	# create neural net 
	nn = NN.NeuralNetwork(sizes=[3,2,1])

	# test that weights are the correct sizes
	assert nn.weights[0].shape == (2,3)
	assert nn.weights[1].shape == (1,2)

	# test that bias are the correct sizes
	assert len(nn.bias) == 3

	# create neural net 
	nn2 = NN.NeuralNetwork(sizes=[1,2,3])

	# test that weights are the correct sizes
	assert nn2.weights[0].shape == (2,1)
	assert nn2.weights[1].shape == (3,2)

	# test that bias are the correct sizes
	assert len(nn2.bias) == 3

	# create 4-layer neural net 
	nn3 = NN.NeuralNetwork(sizes=[1,2,3,4])

	# test that weights are the correct sizes
	assert nn3.weights[0].shape == (2,1)
	assert nn3.weights[1].shape == (3,2)
	assert nn3.weights[2].shape == (4,3)

	# test that bias are the correct sizes
	assert len(nn3.bias) == 4

def test_fit():
	# create neural net 
	# use an autoencoder here 
	nn = NN.NeuralNetwork(sizes=[3,1,3], seed=0)

	# test that weights are the correct sizes
	assert (nn.weights[0].shape == (1,3))
	assert (nn.weights[1].shape == (3,1))

	# test that bias are the correct sizes
	assert len(nn.bias) == 3

	# create fake dataset
	test_x = np.array([[1,0,0], [0,1,0], [0,0,1]])
	test_y = np.array([[1,0,0], [0,1,0], [0,0,1]])

	# store the current weights & bias to compare to after fitting
	w = [a for a in nn.weights[0][0]]
	b = [a for a in nn.bias]

	# fit neural net to text_x, text_y
	nn.fit(test_x, test_y)

	# check that layers are being imported correctly
	assert (nn.layers[0] == test_x).all()
	assert (nn.layers[1].shape == (1,3))
	assert (nn.layers[-1].shape == (3,3))
	
	# check that the weights have been updated
	assert !(nn.weights[0] == np.array([w])).all()
	assert !(nn.bias == np.array([b])).all()

	# check that the 

def test_predict():
	# create neural net 
	nn = NN.NeuralNetwork(sizes=[2,1,1], seed=0, alpha=1)

	# create fake dataset of various sample sizes
	test_x = np.array([[1,0], [0,1]])
	test_x2 = np.array([[1,0], [0,1], [0,1]])
	test_x3 = np.array([[1,0], [0,1], [0,1], [0,1]])

	# check that the predictions are the right shape
	assert (nn.predict(test_x).shape == (2,1))
	assert (nn.predict(test_x2).shape == (3,1))
	assert (nn.predict(test_x3).shape == (4,1))


