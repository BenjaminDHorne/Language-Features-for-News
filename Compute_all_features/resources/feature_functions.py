import nltk
from nltk import tokenize
from nltk.util import ngrams
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from readability import Readability
import collections
from nltk.stem.porter import *
from nltk import word_tokenize
import string
import pickle

### This File contains functions for each type of feature. Use Compute_All_Features.py to run.

DIRNAME = os.path.dirname(__file__)

class Functions:

	def fix(self, text):
		try:
			text = text.decode("ascii", "ignore")
		except:
			t=[unicodedata.normalize('NFKD', unicode(q)).encode('ascii','ignore') for q in text]
			text=''.join(t).strip()
		return text

	def load_happiness_index_lexicon(self, filepath="./resources/"):
		word_to_happiness = {}
		with open(os.path.join(filepath, "happiness_index.txt")) as lex:
			lex.readline()
			for line in lex:
				line = line.strip().split("\t")
				word_to_happiness[line[0]] = line[2]
		return word_to_happiness


	def happiness_index_feats(self, text):
		happiness_scores = []
		happiness = self.load_happiness_index_lexicon()
		tokens = word_tokenize(text)
		tokens = [t.lower() for t in tokens]
		with open("./resources/stopwords.txt") as stopdata:
			stopwords = [w.strip() for w in stopdata]
		stopwords = set(stopwords)
		for token in tokens:
			if token not in stopwords:
				if token in happiness.keys():
					happiness_scores.append(float(happiness[token]))
				else:
					happiness_scores.append(5)
		if len(happiness_scores) == 0:
			return 0
		h = float(sum(happiness_scores)) / len(happiness_scores)
		return h


	def load_moral_foundations_lexicon(self, filepath="./resources/"):
		code_to_foundation = {}
		foundation_to_lex = {}
		with open(os.path.join(filepath, "moral foundations dictionary.dic")) as lex:
			header_token = self.fix(lex.readline())
			for line in lex:
				line = self.fix(line)
				if line == header_token:
					break
				code_foundation = line.strip().split()
				code_to_foundation[code_foundation[0]] = code_foundation[1]
				foundation_to_lex[code_foundation[1]] = []
			for line in lex:
				try:
					word_code = line.strip().split()
					stem = word_code[0].replace("*", "")
					codes = word_code[1:]
					for x in xrange(len(codes)):
						foundation_to_lex[code_to_foundation[codes[x]]].append(stem)
				except:
					continue
		return foundation_to_lex


	def moral_foundation_feats(self, text):
		foundation_counts = {}
		foundation_lex_dictionary = self.load_moral_foundations_lexicon()
		tokens = word_tokenize(text)
		stemmer = PorterStemmer()
		stemed_tokens = [stemmer.stem(t) for t in tokens]
		for key in foundation_lex_dictionary.keys():
			foundation_counts[key] = float(sum([stemed_tokens.count(i) for i in foundation_lex_dictionary[key]])) / len(
				stemed_tokens)
		return foundation_counts["HarmVirtue"], foundation_counts["HarmVice"], foundation_counts["FairnessVirtue"], \
			   foundation_counts["FairnessVice"], foundation_counts["IngroupVirtue"], foundation_counts["IngroupVice"], \
			   foundation_counts["AuthorityVirtue"], foundation_counts["AuthorityVice"], foundation_counts["PurityVirtue"], \
			   foundation_counts["PurityVice"], foundation_counts["MoralityGeneral"]


	def load_acl13_lexicons(self, filepath="./resources/"):
		with open(os.path.join(filepath, "bias-lexicon.txt")) as lex:
			bias = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "assertives.txt")) as lex:
			assertives = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "factives.txt")) as lex:
			factives = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "hedges.txt")) as lex:
			hedges = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "implicatives.txt")) as lex:
			implicatives = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "report_verbs.txt")) as lex:
			report_verbs = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "negative-words.txt")) as lex:
			negative = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "positive-words.txt")) as lex:
			positive = set([self.fix(l.strip()) for l in lex])
		with open(os.path.join(filepath, "subjclueslen.txt")) as lex:
			wneg = set([])
			wpos = set([])
			wneu = set([])
			sneg = set([])
			spos = set([])
			sneu = set([])
			for line in lex:
				line = self.fix(line).split()
				if line[0] == "type=weaksubj":
					if line[-1] == "priorpolarity=negative":
						wneg.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=positive":
						wpos.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=neutral":
						wneu.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=both":
						wneg.add(line[2].split("=")[1])
						wpos.add(line[2].split("=")[1])
				elif line[0] == "type=strongsubj":
					if line[-1] == "priorpolarity=negative":
						sneg.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=positive":
						spos.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=neutral":
						sneu.add(line[2].split("=")[1])
					elif line[-1] == "priorpolarity=both":
						spos.add(line[2].split("=")[1])
						sneg.add(line[2].split("=")[1])
		return bias, assertives, factives, hedges, implicatives, report_verbs, positive, negative, wneg, wpos, wneu, sneg, spos, sneu


	def bias_lexicon_feats(self, text):
		bias, assertives, factives, hedges, implicatives, report_verbs, positive_op, negative_op, wneg, wpos, wneu, sneg, spos, sneu = self.load_acl13_lexicons()
		tokens = word_tokenize(text)
		bigrams = [" ".join(bg) for bg in ngrams(tokens, 2)]
		trigrams = [" ".join(tg) for tg in ngrams(tokens, 3)]
		bias_count = float(sum([tokens.count(b) for b in bias])) / len(tokens)
		assertives_count = float(sum([tokens.count(a) for a in assertives])) / len(tokens)
		factives_count = float(sum([tokens.count(f) for f in factives])) / len(tokens)
		hedges_count = sum([tokens.count(h) for h in hedges]) + sum([bigrams.count(h) for h in hedges]) + sum(
			[trigrams.count(h) for h in hedges])
		hedges_count = float(hedges_count) / len(tokens)
		implicatives_count = float(sum([tokens.count(i) for i in implicatives])) / len(tokens)
		report_verbs_count = float(sum([tokens.count(r) for r in report_verbs])) / len(tokens)
		positive_op_count = float(sum([tokens.count(p) for p in positive_op])) / len(tokens)
		negative_op_count = float(sum([tokens.count(n) for n in negative_op])) / len(tokens)
		wneg_count = float(sum([tokens.count(n) for n in wneg])) / len(tokens)
		wpos_count = float(sum([tokens.count(n) for n in wpos])) / len(tokens)
		wneu_count = float(sum([tokens.count(n) for n in wneu])) / len(tokens)
		sneg_count = float(sum([tokens.count(n) for n in sneg])) / len(tokens)
		spos_count = float(sum([tokens.count(n) for n in spos])) / len(tokens)
		sneu_count = float(sum([tokens.count(n) for n in sneu])) / len(tokens)
		return bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count


	def ttr(self, text):
		words = text.split()
		dif_words = len(set(words))
		tot_words = len(words)
		if tot_words == 0:
			return 0
		return (float(dif_words) / tot_words)


	def POS_features(self, fn, text, outpath):
		fname = os.path.join(outpath, fn.split(".")[0] + "_tagged.txt")

		pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT",
					"POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WP$", "WRB", "VB", "VBD", "VBG",
					"VBN", "VBP", "VBZ", "WDT", "WP"]
		sents = tokenize.sent_tokenize(text)
		counts_norm = []
		allwords = []
		sents = tokenize.sent_tokenize(text)

		with open(fname, "w") as out:
			for sent in sents:
				words = sent.strip(".").split()
				tags = nltk.pos_tag(words)
				strtags = ["/".join((wt[0], wt[1])) for wt in tags]
				out.write(" ".join(strtags) + " ")

		with open(fname, "r") as fl:
			line = fl.readline()  # each file is one line

		wordandtag = line.strip().split()
		try:
			tags = [wt.split("/")[1] for wt in wordandtag]
		except:
			print wordandtag
		counts = collections.Counter(tags)

		for pt in pos_tags:
			try:
				counts_norm.append(float(counts[pt]) / len(tags))
			except:
				counts_norm.append(0)

		return counts_norm


	def vadersent(self, text):
		analyzer = SentimentIntensityAnalyzer()
		vs = analyzer.polarity_scores(text)
		return vs['neg'], vs['neu'], vs['pos']


	def readability(self, text):
		rd = Readability(text)
		fkg_score = rd.FleschKincaidGradeLevel()
		SMOG = rd.SMOGIndex()
		return fkg_score, SMOG


	def wordlen_and_stop(self, text):
		with open("./resources/stopwords.txt") as data:
			stopwords = [w.strip() for w in data]
		set(stopwords)
		words = word_tokenize(text)
		WC = len(words)
		stopwords_in_text = [s for s in words if s in stopwords]
		percent_sws = float(len(stopwords_in_text)) / len(words)
		lengths = [len(w) for w in words if w not in stopwords]
		if len(lengths) == 0:
			word_len_avg = 3
		else:
			word_len_avg = float(sum(lengths)) / len(lengths)
		return percent_sws, word_len_avg, WC


	def stuff_LIWC_leftout(self, pid, text):
		puncs = set(string.punctuation)
		tokens = word_tokenize(text)
		quotes = tokens.count("\"") + tokens.count('``') + tokens.count("''")
		Exclaim = tokens.count("!")
		AllPunc = 0
		for p in puncs:
			AllPunc += tokens.count(p)
		words_upper = 0
		for w in tokens:
			if w.isupper():
				words_upper += 1
		try:
			allcaps = float(words_upper) / len(tokens)
		except:
			print pid
		return (float(quotes) / len(tokens)) * 100, (float(Exclaim) / len(tokens)) * 100, (
		float(AllPunc) / len(tokens)) * 100, allcaps


	def subjectivity(self, text):
		loaded_model = pickle.load(open(os.path.join(DIRNAME, '', 'NB_Subj_Model.sav'), 'rb'))
		count_vect = pickle.load(open(os.path.join(DIRNAME, '', 'count_vect.sav'), 'rb'))
		tfidf_transformer = pickle.load(open(os.path.join(DIRNAME, '', 'tfidf_transformer.sav'), 'rb'))
		X_new_counts = count_vect.transform([text])
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)
		result = loaded_model.predict_proba(X_new_tfidf)
		prob_obj = result[0][0]
		prob_subj = result[0][1]
		return prob_obj, prob_subj


	def load_LIWC_dictionaries(self, filepath="./resources/"):
		cat_dict = {}
		stem_dict = {}
		counts_dict = {}
		with open(os.path.join(filepath, "LIWC2007_English100131.dic")) as raw:
			raw.readline()
			for line in raw:
				if line.strip() == "%":
					break
				line = line.strip().split()
				cat_dict[line[0]] = line[1]
				counts_dict[line[0]] = 0
			for line in raw:
				line = line.strip().split()
				stem_dict[line[0]] = [l.replace("*", "") for l in line[1:]]
		return cat_dict, stem_dict, counts_dict


	def LIWC(self, text, cat_dict, stem_dict, counts_dict):
		for key in counts_dict:
			counts_dict[key] = 0
		tokens = word_tokenize(text)
		WC = len(tokens)
		stemmer = PorterStemmer()
		stemed_tokens = [stemmer.stem(t) for t in tokens]

		# count and percentage
		for stem in stem_dict:
			count = stemed_tokens.count(stem.replace("*", ""))
			if count > 0:
				for cat in stem_dict[stem]:
					counts_dict[cat] += count
		counts_norm = [float(counts_dict[cat]) / WC * 100 for cat in counts_dict]
		cats = [cat_dict[cat] for cat in cat_dict]
		return counts_norm, cats