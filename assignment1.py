import sys
import argparse
import math

UNKNOWN_THRESHOLD = 3

UNKNOWN_TOKEN = '<UNK>'
START_TOKEN = '<START>'
STOP_TOKEN = '<STOP>'

STOP_JOIN = ' ' + STOP_TOKEN + ' '
UNDEF_PERPLEXITY = 0 

TRAIN = "1b_benchmark.train.tokens"
DEVELOP = "1b_benchmark.dev.tokens"
TEST = "1b_benchmark.test.tokens"

OPTIMIZED_UNI = 0.1
OPTIMIZED_BI = 0.4
OPTIMIZED_TRI = 0.5

class Dataset():
	"""A Dataset which stores perplexity related information relating to an NLP dataset""" 
	
	def __init__(self, document, vocab, bigram_dict = None, trigram_dict = None):
		""" Dataset constructor. 

		Parameters:
			document: the document (list of sentences).
			vocab: vocabulary by which to evaluate the perplexities of this dataset, including <STOP> but not <START> tokens.
			bigram_dict: (optional) bigram probability dictionary built using train dataset.
			trigram_dict: (optional) trigram probability dictionary built using train dataset.  
		"""

		self.document = document
		self.document_count = ' '.join(document).replace(START_TOKEN + ' ', '').count(' ') + 1
		self.vocab = vocab
		self.bigram_dict = bigram_dict
		self.trigram_dict = trigram_dict


	def get_perplexity(self, sentence_probabilities):
		""" Gets perplexity given a list of lists of word probabilities, wherein
		the inner list represents the word probabilities of each sentence and the outer
		list is a list of sentence probabilities. Uses math rule: log of product == sum of logs.

		Parameters:
			sentence_probabilities: list of lists of word probabilities.

		Returns:
			Perplexity score of list of sentence probabilities.
		"""

		summation = 0
		for word_probabilities in sentence_probabilities:
			for word_probability in word_probabilities:
				if word_probability == UNDEF_PERPLEXITY:
					return UNDEF_PERPLEXITY
				summation -= math.log(word_probability, 2)

		perplexity = 2 ** (summation / self.document_count)
		return perplexity


	def get_perplexity_unigram(self):
		""" Gets list of list of unigram probabilities, wherein the inner list represents the 
		unigram probabilities of each sentence and the outer list is a list of sentence probabilities.
		Uses vocab for unigram probabilities.

		Returns:
			List of lists of unigram probabilities.
		"""

		probabilities = []

		for sentence in self.document:
			words = sentence.replace(START_TOKEN + ' ', '').split(' ')
			sentence_probabilities = []
			for word in words:
				sentence_probabilities.append(self.vocab.get(word))
			probabilities.append(sentence_probabilities)

		return probabilities


	def get_perplexity_bigram(self):
		""" Gets list of list of bigram probabilities, wherein the inner list represents the 
		bigram probabilities of each sentence and the outer list is a list of sentence probabilities.
		Uses bigram_dict for bigram probabilities. If a bigram is not found in bigram_dict, the probability
		is assigned to UNDEF_PROBABILITY (0) and calculating perplexity gives 0.

		Returns:
			-1 if bigram_dict is not defined.
			List of lists of bigram probabilities.
		"""

		if self.bigram_dict == None:
			return -1
		probabilities = []

		for sentence in self.document:
			words = sentence.split(' ')
			words_len = len(words)
			sentence_probabilities = []

			for word_i in range(words_len):
				if word_i < words_len - 1:
					bigram = (words[word_i], words[word_i + 1])
					if bigram in self.bigram_dict:
						sentence_probabilities.append(self.bigram_dict.get(bigram))
					else:
						sentence_probabilities.append(UNDEF_PERPLEXITY)
			probabilities.append(sentence_probabilities)

		return probabilities


	def get_perplexity_trigram(self):
		""" Gets list of list of trigram probabilities, wherein the inner list represents the 
		trigram probabilities of each sentence and the outer list is a list of sentence probabilities.
		Uses trigram_dict for trigram probabilities. If a trigram is not found in trigram_dict, the probability
		is assigned to UNDEF_PROBABILITY (0) and calculating perplexity gives 0.

		Returns:
			-1 if trigram_dict is not defined.
			List of lists of trigram probabilities.
		"""

		if self.trigram_dict == None:
			return -1
		probabilities = []

		for sentence in self.document:
			words = sentence.split(' ')
			words_len = len(words)
			sentence_probabilities = []

			first_trigram = (words[0], words[1])
			if first_trigram in self.trigram_dict:
				sentence_probabilities.append(self.trigram_dict.get(first_trigram))
			else:
				sentence_probabilities.append(UNDEF_PERPLEXITY)
				undef_perplexity = True

			for word_i in range(words_len):
				if word_i < words_len - 2:
					trigram = (words[word_i], words[word_i + 1], words[word_i + 2])
					if trigram in self.trigram_dict:
						sentence_probabilities.append(self.trigram_dict.get(trigram))
					else:
						sentence_probabilities.append(UNDEF_PERPLEXITY)
			probabilities.append(sentence_probabilities)

		return probabilities


	def get_perplexity_trigram_smoothed(self, weight_uni, weight_bi, weight_tri):
		""" Gets list of list of smoothed trigram probabilities, wherein the inner list represents the 
		trigram probabilities of each sentence and the outer list is a list of sentence probabilities.
		Uses functions get_perplexity_unigram, get_perplexity_bigram, and get_perplexity_trigram to get list
		of list of each respective ngram probabilities and linearly weights each using given weights to find
		a list of list of smoothed probabilities.

		Parameters:
			weight_uni: weight assigned to unigram language model.
			weight_bi: weight assigned to bigram language model.
			weight_tri: weight assigned to trigram language model.		

		Returns:
			List of lists of trigram smoothed probabilities.
		"""

		unigram_word_probabilities = self.get_perplexity_unigram()
		bigram_word_probabilities = self.get_perplexity_bigram()
		trigram_word_probabilities = self.get_perplexity_trigram()

		trigram_smoothed_probabilities = []
		for r in range(len(unigram_word_probabilities)):
			sentence_probabilities = []
			for c in range(len(unigram_word_probabilities[r])):
				if bigram_word_probabilities[r][c] != 0.0:
					smoothed_probability = unigram_word_probabilities[r][c] * weight_uni + \
										   bigram_word_probabilities[r][c] * weight_bi + \
										   trigram_word_probabilities[r][c] * weight_tri
				elif weight_uni != 0.0:
					smoothed_probability = unigram_word_probabilities[r][c]
				else:
					smoothed_probability = 0
				sentence_probabilities.append(smoothed_probability)
			trigram_smoothed_probabilities.append(sentence_probabilities)
		return trigram_smoothed_probabilities


def read_file(file, vocab = None):
	""" Reads file as list of sentences and modifies it with correct formatting.
	
	Parameters:
		file: file path to read.
		vocab: vocabulary by which to enforce on the read file.
	
	Disclaimers:
		Inserts <UNK> tokens to words not found in vocab if vocab is defined.

	Returns:
		list of sentences read from the file, starting and ending with <START> and <STOP> tokens respectively.
	"""

	with open(file, encoding='utf8') as fp:
		lines = fp.read().splitlines()
	if vocab is not None:
		doc = STOP_JOIN.join(lines)
		words = [word if word in vocab else UNKNOWN_TOKEN for word in doc.split(' ')]
		doc = ' '.join(words)
		lines = doc.split(STOP_JOIN)
	lines = [START_TOKEN + ' ' + line + ' ' + STOP_TOKEN for line in lines]
	return lines


def get_vocab(document, unknown_threshold = UNKNOWN_THRESHOLD):
	""" Builds vocab given document, which is a normalized dictionary mapping
	tokens to probability of each unique token in the dataset, replacing uncommon
	tokens with UNKNOWN_TOKEN based on given unknown_threshold. Used for uni-, bi-,
	and tri-gram language models.

	Parameters:
		document: list of sentences.
		unknown_threshold: (defaults to UNKNOWN_THRESHOLD) threshold by which to replace uncommon tokens.
	
	Returns:
		vocab, dict mapping tokens -> probability of that token in dataset.
	"""

	vocab = dict()

	sentences_count = len(document)
	document = ' '.join(document)

	no_start_document = document.replace('<START> ', '')
	train_size = no_start_document.count(' ') + 1

	for word in no_start_document.split(' '):
		vocab[word] = vocab.get(word, 0) + 1

	unknown_count = 0
	unknown_tokens = set()
	for token, count in vocab.items():
		if count < unknown_threshold:
			unknown_tokens.add(token)
			unknown_count += count
		else:
			vocab[token] = count / train_size

	for unknown_token in unknown_tokens:
		del vocab[unknown_token]
	
	vocab[STOP_TOKEN] = sentences_count / train_size
	vocab[UNKNOWN_TOKEN] = unknown_count / train_size

	return vocab, train_size


def get_bigram_dict(document, vocab, train_size):
	""" Builds bigram dictionary mapping bigrams to their probability in given train 
	document. Used by getting trigram_dict as well as evaluating bi- and tri-gram perplexities.

	Parameters:
		document: list of sentences with <START>, <STOP>, and <UNK> tokens.
		vocab: vocabulary of train dataset. 
		train_size: number of tokens in train dataset.

	Returns:
		bigram_dict mapping bigrams (x, y) -> probability of each (x, y) given x
	"""

	bigram_dict = dict()
	for line in document:
		words = line.split(' ')
		words_len = len(words)
		for word_i in range(words_len):
			if word_i < words_len - 1:
				bigram = (words[word_i], words[word_i + 1])
				bigram_dict[bigram] = bigram_dict.get(bigram, 0) + 1

	for bigram, count in bigram_dict.items():
		if bigram[0] != START_TOKEN:
			bigram_dict[bigram] = count / (vocab[bigram[0]] * train_size)
		else:
			bigram_dict[bigram] = count / (vocab[STOP_TOKEN] * train_size)
	return bigram_dict


def get_trigram_dict(document, bigram_dict, vocab, train_size):
	""" Builds trigram dictionary mapping trigrams to their probability in given train 
	document. Used in evaluating trigram and smoothed perplexities.

	Parameters:
		document: list of sentences with <START>, <STOP>, and <UNK> tokens.
		bigram_dict: bigram dictionary of train dataset.
		vocab: vocabulary of train dataset. 
		train_size: number of tokens in train dataset.

	Returns:
		trigram_dict mapping trigrams (x, y, z) -> probability of each (x, y, z) given (x, y)
	"""

	trigram_dict = dict()
	for line in document:
		words = line.split(' ')
		words_len = len(words)
		first_trigram = (words[0], words[1])
		trigram_dict[first_trigram] = trigram_dict.get(first_trigram, 0) + 1
		for word_i in range(words_len):
			if word_i < words_len - 2:
				trigram = (words[word_i], words[word_i + 1], words[word_i + 2])
				trigram_dict[trigram] = trigram_dict.get(trigram, 0) + 1

	for trigram, count in trigram_dict.items():
		if len(trigram) == 3 and trigram[0] != START_TOKEN:
			trigram_dict[trigram] = count / (bigram_dict.get((trigram[0], trigram[1])) * vocab.get(trigram[0], 1) * train_size)
		elif len(trigram) == 2: 
			trigram_dict[trigram] = count / (vocab[STOP_TOKEN] * train_size)
		else:
			trigram_dict[trigram] = count / (bigram_dict.get((trigram[0], trigram[1])) * vocab.get(STOP_TOKEN, 1) * train_size)
	return trigram_dict


def parse_args(params):
	parser = argparse.ArgumentParser(description='Evaluating perplexities of uni-, bi-, tri-gram, and smoothed language models with train, dev, and test datasets')
	parser.add_argument('-n', '--train', default=TRAIN, type=str, help='Path to train dataset.')
	parser.add_argument('-v', '--dev', default=DEVELOP, type=str, help='Path to dev dataset.')
	parser.add_argument('-s', '--test', default=TEST, type=str, help='Path to test dataset.')
	parser.add_argument('-u', '--uniweight', default=OPTIMIZED_UNI, type=float, help='Optimized unigram weight to be applied in smoothing test dataset.')
	parser.add_argument('-b', '--biweight', default=OPTIMIZED_BI, type=float, help='Optimized bigram weight to be applied in smoothing test dataset.')
	parser.add_argument('-t', '--triweight', default=OPTIMIZED_TRI, type=float, help='Optimized trigram weight to be applied in smoothing test dataset.')
	parser.add_argument('-k', '--unk-threshold', default=UNKNOWN_THRESHOLD, type=int, help='Threshold by which to categorize uncommon tokens to <UNK> tokens.')

	args = parser.parse_args(params)
	weight_sum = args.uniweight + args.biweight + args.triweight
	if weight_sum != 1.0:
		raise Exception("Weights for unigram ({}), bigram ({}), trigram ({}) must sum to 1, not {}".format(args.uniweight, args.biweight, args.triweight, weight_sum))
	return args


def main(args):
	""" Builds vocabulary as well as bi- and tri-gram dictionaries off of TRAIN dataset
	and evaluates uni-, bi-, tri-gram perplexities on train, dev, and test datasets.
	Also evaluates smoothed trigram perplexities using various sets of hyperparameters.
	"""

	train_vocab = read_file(args.train)
	vocab, train_size = get_vocab(train_vocab, args.unk_threshold)

	train_doc = read_file(args.train, vocab)
	dev_doc = read_file(args.dev, vocab)
	test_doc = read_file(args.test, vocab)
	docs = [(' - train - ', train_doc), (' - dev - ', dev_doc), (' - test - ', test_doc)]

	bigram_dict = get_bigram_dict(train_doc, vocab, train_size)
	trigram_dict = get_trigram_dict(train_doc, bigram_dict, vocab, train_size)

	for doc in docs:
		doc_dataset = Dataset(doc[1], vocab, bigram_dict, trigram_dict)
		print(doc[0])
		print('unigram - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(1, 0, 0))))
		print('bigram - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(0, 1, 0))))
		print('trigram - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(0, 0, 1))))

		opt_smoothed = doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(args.uniweight, args.biweight, args.triweight))
		print("optimized/parameter smoothed {}, {}, {} - {}".format(args.uniweight, args.biweight, args.triweight, str(opt_smoothed)))
		
		if doc[0] == ' - test - ':
			continue
		print("default smoothed .1, .3, .6 - " + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(.1, .3, .6))))
		print('smoothed .33, .33, .34 - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(.33, .33, .34))))
		print('smoothed .2, .4, .4 - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(.2, .4, .4))))
		print('smoothed .2, .3, .5 - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(.2, .3, .5))))
		print('smoothed .1, .2, .7 - ' + str(doc_dataset.get_perplexity(doc_dataset.get_perplexity_trigram_smoothed(.1, .2, .7))))
		print()


if __name__== "__main__":
	args = parse_args(sys.argv[1:])
	main(args)
