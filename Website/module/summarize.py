# importing libraries 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 

# Input text - to summarize 
text = """Developing assisting system of handicapped persons become a challenging ask in research projects. Recently, a
variety of tools are designed to help visually impaired or blind people object as a visual substitution system. The majority
of these tools are based on the conversion of input information into auditory or tactile sensory information. Furthermore,
object recognition and text retrieval are exploited in the visual substitution systems. Text detection and recognition
provides the description of the surrounding environments, so that the blind person can readily recognize the scene. In this
work, we aim to introduce a method for detecting and recognizing text in indoor scene. The process consists on the
detection of the regions of interest that should contain the text using the connected component. Then, the text detection is
provided by employing the images correlation. This component of an assistive blind person should be simple, so that the
users are able to obtain the most informative feedback within the shortest time"""

# Tokenizing the text 
stopWords = set(stopwords.words("english")) 
words = word_tokenize(text) 

# Creating a frequency table to keep the 
# score of each word 

freqTable = dict() 
for word in words: 
	word = word.lower() 
	if word in stopWords: 
		continue
	if word in freqTable: 
		freqTable[word] += 1
	else: 
		freqTable[word] = 1

# Creating a dictionary to keep the score 
# of each sentence 
sentences = sent_tokenize(text) 
sentenceValue = dict() 

for sentence in sentences: 
	for word, freq in freqTable.items(): 
		if word in sentence.lower(): 
			if sentence in sentenceValue: 
				sentenceValue[sentence] += freq 
			else: 
				sentenceValue[sentence] = freq 



sumValues = 0
for sentence in sentenceValue: 
	sumValues += sentenceValue[sentence] 

# Average value of a sentence from the original text 

average = int(sumValues / len(sentenceValue)) 

# Storing sentences into our summary. 
summary = '' 
for sentence in sentences: 
	if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)): 
		summary += " " + sentence 
print(summary) 
