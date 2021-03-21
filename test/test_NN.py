from scripts import NN


def test_encoder():

	# test that each letter is being properly encoded
	assert (encode("A") == [1,0,0,0]).all()
	assert (encode("C") == [0,1,0,0]).all()
	assert (encode("G") == [0,0,1,0]).all()
	assert (encode("T") == [0,0,0,1]).all()

	# test that the encoder is giving back the right number of values
	assert len(encode("acgt")) == 16

	# test that a sequence returns the right encoded values
    assert (encode("acgt") == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]).all()

def test_one_d_output():
    assert True

def test_one_d_output():
    assert True
