import pickle 

def load_dict(filename):
	with open(filename, 'rb') as f:
		di = pickle.load(f)
	return di

def load_expl(train_file, test_file):
	expl_train = []
	expl_test = []
	prediction_train = []
	prediction_test = []
	if train_file is not None:
		train_dict = load_dict(train_file)
		for i in range(len(train_dict)):
			# imshow_expl(train_dict[i]['expl'])
			expl_train.append(train_dict[i]['expl'])
			prediction_train.append(train_dict[i]['prediction'])
	if test_file is not None:
		test_dict = load_dict(test_file)
		for i in range(len(test_dict)):
			expl_test.append(test_dict[i]['expl'])
			prediction_test.append(test_dict[i]['prediction'])
	return expl_train, expl_test, prediction_train, prediction_test