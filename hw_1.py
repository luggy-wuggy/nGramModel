import tokenize 
import math

# make dictionary with KEY: 


class SpecialWord:
	startWord = ""
	stopWord  = ""
	unkWord   = ""

	def __init__(self):
		self.startWord = "<START>"
		self.stopWord  = "<STOP>"
		self.unkWord   = "<UNK>"

	def start(self):
		return self.startWord

	def stop(self):
		return self.stopWord

	def unk(self):
		return self.unkWord

file = "1b_benchmark.train.tokens"
develop = "1b_benchmark.dev.tokens"
test = "1b_benchmark.test.tokens"


## ADD <START> and <STOP> to the beginning and end of each line (aka sentence)
## File writing
with open(file) as fp:
	lines = fp.read().splitlines()
with open(file, "w") as fp:
	for line in lines:
		print("<START> " + line + " <STOP>", file=fp)

## File reading
fileRead = open(file, "r")
lines = fileRead.readlines()

dic = dict()   ## Dictionary of all the words
vocab = {"<UNK>" : 0} ## Vocabulary (dictionary) filtered with words >= count of 3
countWord = 0  ## Count of tokens in the whole dataset minus <START>

## CREATION OF DIC
for line in lines:
	wds = line.split()
	countWord = countWord + len(wds) - 1  #Not including START symbol
	for w in wds:
		if w != "<START>":
			dic[w] = dic.get(w,0) + 1

## CREATION OF VOCAB
for (key, value) in dic.items():
	if value >= 3:
		vocab[key] = value
	else:
		vocab["<UNK>"] = vocab.get("<UNK>") + value


uni = []

## UNI model list
## probability of each sentence in data set
perplexity = 1
totalWords = 0

for line in lines:
	wds = line.split()
	probSentence = 1
	wordCount = len(wds)
	numerator = 1 
	for w in wds:
		numerator *= (vocab.get(w, vocab.get("<UNK>")))

	totalWords += wordCount
	probSentence = -1 * ((math.log(numerator,2)) - (wordCount * math.log(countWord,2)))
	uni.append(probSentence)

summ = 0
for prob in uni:
	summ += prob

perplex = 2 ** (summ/totalWords)

print(perplex)


'''
bi = []
tri = []

## BI and TRI model list
for x in range(len(uni)):
	if x < (len(uni) - 1):
		pair = (uni[x], uni[x+1])
		bi.append(pair)

	if x < (len(uni) - 2):
		triple = (uni[x], uni[x+1], uni[x+2])
		tri.append(triple)

print(bi)
'''

