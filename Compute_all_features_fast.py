import nltk
from nltk import tokenize
from nltk.util import ngrams
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from resources.readability import Readability
import collections
from nltk.stem.porter import *
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import pickle
from multiprocessing import Process, Manager, freeze_support

DIRNAME = os.path.dirname(__file__)

stopwords = set(stopwords.words('english'))

#Feature Functions <-----------------------------------------------------------------------
def load_happiness_index_lexicon(filepath="./resources/"):
    word_to_happiness = {}
    with open(os.path.join(filepath, "happiness_index.txt")) as lex:
        lex.readline()
        for line in lex:
            line = line.strip().split("\t")
            word_to_happiness[line[0]] = line[2]
    return word_to_happiness


def happiness_index_feats(happiness, text):
    happiness_scores = []
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens]
    for token in tokens:
        if token not in stopwords:
            if token in happiness.keys():
                happiness_scores.append(float(happiness[token]))
            else:
                happiness_scores.append(5)
    if len(happiness_scores) == 0:
        return 0
    h = float(sum(happiness_scores)) / len(happiness_scores)
    feature_dictionary['Happiness'] = h


def load_moral_foundations_lexicon(filepath="./resources/"):
    code_to_foundation = {}
    foundation_to_lex = {}
    with open(os.path.join(filepath, "moral foundations dictionary.dic")) as lex:
        header_token = fix(lex.readline())
        for line in lex:
            line = fix(line)
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


def moral_foundation_feats(foundation_lex_dictionary, text):
    foundation_counts = {}
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    stemed_tokens = [stemmer.stem(t) for t in tokens]
    for key in foundation_lex_dictionary.keys():
        foundation_counts[key] = float(sum([stemed_tokens.count(i) for i in foundation_lex_dictionary[key]])) / len(
            stemed_tokens)

    feature_dictionary["HarmVirtue"] = foundation_counts["HarmVirtue"]
    feature_dictionary["HarmVice"] = foundation_counts["HarmVice"]
    feature_dictionary["FairnessVirtue"] = foundation_counts["FairnessVirtue"]
    feature_dictionary["FairnessVice"] = foundation_counts["FairnessVice"]
    feature_dictionary["IngroupVirtue"] = foundation_counts["IngroupVirtue"]
    feature_dictionary["IngroupVice"] = foundation_counts["IngroupVice"]
    feature_dictionary["AuthorityVirtue"] = foundation_counts["AuthorityVirtue"]
    feature_dictionary["AuthorityVice"] = foundation_counts["AuthorityVice"]
    feature_dictionary["PurityVirtue"] = foundation_counts["PurityVirtue"]
    feature_dictionary["PurityVice"] = foundation_counts["PurityVice"]
    feature_dictionary["MoralityGeneral"] = foundation_counts["MoralityGeneral"]

def load_acl13_lexicons(filepath="./resources/"):
    with open(os.path.join(filepath, "bias-lexicon.txt")) as lex:
        bias = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "assertives.txt")) as lex:
        assertives = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "factives.txt")) as lex:
        factives = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "hedges.txt")) as lex:
        hedges = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "implicatives.txt")) as lex:
        implicatives = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "report_verbs.txt")) as lex:
        report_verbs = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "negative-words.txt")) as lex:
        negative = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "positive-words.txt")) as lex:
        positive = set([fix(l.strip()) for l in lex])
    with open(os.path.join(filepath, "subjclueslen.txt")) as lex:
        wneg = set([])
        wpos = set([])
        wneu = set([])
        sneg = set([])
        spos = set([])
        sneu = set([])
        for line in lex:
            line = fix(line).split()
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


def bias_lexicon_feats(bias, assertives, factives, hedges, implicatives, report_verbs, positive_op, negative_op, wneg, wpos, wneu, sneg, spos, sneu, text):
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

    feature_dictionary["bias_count"] = bias_count
    feature_dictionary["assertives_count"] = assertives_count
    feature_dictionary["factives_count"] = factives_count
    feature_dictionary["hedges_count"] = hedges_count
    feature_dictionary["implicatives_count"] = implicatives_count
    feature_dictionary["report_verbs_count"] = report_verbs_count
    feature_dictionary["positive_op_count"] = positive_op_count
    feature_dictionary["negative_op_count"] = negative_op_count
    feature_dictionary["wneg_count"] = wneg_count
    feature_dictionary["wpos_count"] = wpos_count
    feature_dictionary["wneu_count"] = wneu_count
    feature_dictionary["sneg_count"] = sneg_count
    feature_dictionary["spos_count"] = spos_count
    feature_dictionary["sneu_count"] = sneu_count


def ttr(text):
    words = text.split()
    dif_words = len(set(words))
    tot_words = len(words)
    if tot_words == 0:
        feature_dictionary['ttr'] = 0
    feature_dictionary['TTR'] = str(float(dif_words) / tot_words)


def POS_features(fn, text, outpath):
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
            feature_dictionary[pt] = str(float(counts[pt]) / len(tags))
        except:
            feature_dictionary[pt] = str(0)

def vadersent(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    feature_dictionary['vad_neg'] = vs['neg']
    feature_dictionary['vad_neu'] = vs['neu']
    feature_dictionary['vad_pos'] = vs['pos']

def readability(text):
    rd = Readability(text)
    fkg_score = rd.FleschKincaidGradeLevel()
    SMOG = rd.SMOGIndex()
    feature_dictionary['FKE'] = fkg_score
    feature_dictionary['SMOG'] = SMOG

def wordlen_and_stop(text):
    words = word_tokenize(text)
    WC = len(words)
    stopwords_in_text = [s for s in words if s in stopwords]
    percent_sws = float(len(stopwords_in_text)) / len(words)
    lengths = [len(w) for w in words if w not in stopwords]

    if len(lengths) == 0:
        word_len_avg = 3
    else:
        word_len_avg = float(sum(lengths)) / len(lengths)

    feature_dictionary['stop'] = percent_sws
    feature_dictionary['wordlen'] = word_len_avg
    feature_dictionary['WC'] = WC


def stuff_LIWC_leftout(pid, text):
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
        allcaps = 0

    feature_dictionary['quotes'] = (float(quotes) / len(tokens)) * 100
    feature_dictionary['Exclaim'] = (float(Exclaim) / len(tokens)) * 100
    feature_dictionary['AllPunc'] = float(AllPunc) / len(tokens) * 100
    feature_dictionary['allcaps'] = allcaps


def subjectivity(loaded_model, count_vect, tfidf_transformer, text):
    X_new_counts = count_vect.transform([text])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    result = loaded_model.predict_proba(X_new_tfidf)
    prob_obj = result[0][0]
    prob_subj = result[0][1]

    feature_dictionary["NB_pobj"] = prob_obj
    feature_dictionary["NB_psubj"] = prob_subj


def load_LIWC_dictionaries(filepath="./resources/"):
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


def LIWC(text, cat_dict, stem_dict, counts_dict):
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

    i = 0
    for cat in cats:
        feature_dictionary[cat] = counts_norm[i]
        i+=1

# Other Functions <-----------------------------------------------------------------------------------------------

def fix(text):
    try:
        text = text.decode("ascii", "ignore")
    except:
        t = [unicodedata.normalize('NFKD', unicode(q)).encode('ascii', 'ignore') for q in text]
        text = ''.join(t).strip()
    return text

def whatsbeendon(filename):
    pids = []
    try:
        with open(filename) as data:
            pids = [line.strip().split(",")[0] for line in data]
        return set(pids)
    except:
        return set(pids)

def make_str(seq):
    strseq = [str(s) for s in seq]
    return strseq

def runInParallel(*funs):
  proc = []
  for fun in funs:
    p = Process(target=fun)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()
            
#main <-----------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    freeze_support()
    outfile = "test_new_feat_code.csv" # keep name for now
    outpath = "./"
    done = whatsbeendon(outfile)
    text_file_start_path = "D:/RPI Research/News Producer Networks/Data/Plain Text Data/New Content/"

    #load stuff

    #happiness_lex = load_happiness_index_lexicon()
    foundation_lex_dictionary = load_moral_foundations_lexicon()
    bias_lex, assertives_lex, factive_lexs, hedges_lex, implicatives_lex, report_verbs_lex, positive_op_lex, negative_op_lex, wneg_lex, wpos_lex, wneu_lex, sneg_lex, spos_lex, sneu_lex = load_acl13_lexicons()
    loaded_model = pickle.load(open(os.path.join('./resources/', 'NB_Subj_Model.sav'), 'rb'))
    count_vect = pickle.load(open(os.path.join('./resources/', 'count_vect.sav'), 'rb'))
    tfidf_transformer = pickle.load(open(os.path.join('./resources/', 'tfidf_transformer.sav'), 'rb'))
    cat_dict, stem_dict, counts_dict = load_LIWC_dictionaries()
    liwc_cats = [cat_dict[cat] for cat in cat_dict]
    pos_tags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP"]

    feature_header_str = "HarmVirtue,HarmVice,FairnessVirtue,FairnessVice,IngroupVirtue,IngroupVice,AuthorityVirtue,AuthorityVice,PurityVirtue,PurityVice,MoralityGeneral,bias_count,assertives_count,factives_count,hedges_count,implicatives_count,report_verbs_count,positive_op_count,negative_op_count,wneg_count,wpos_count,wneu_count,sneg_count,spos_count,sneu_count,TTR,vad_neg,vad_neu,vad_pos,FKE,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps," + ",".join(pos_tags) + "," + ",".join(liwc_cats)
    feature_list = feature_header_str.split(",")

    if len(done) == 0:
        with open(os.path.join(outpath, outfile), "a") as out:
            other = "pid,source,date,"
            string_of_seq = other + ",".join(feature_list)
            out.write(string_of_seq+"\n")

    with Manager() as manager:
        for dirName, subdirList, fileList in os.walk(text_file_start_path):
            #print dirName, subdirList, fileList
            path = dirName+"/"
            for fn in fileList:
                feature_dictionary = manager.dict()
                source = dirName.split("\\")[-1]
                date = fn.split("--")[1]
                pid = fn.split(".")[0]

                if pid in done:
                    continue
                else:
                    print "working on", pid

                #try:
                with open(path+fn) as textdata:
                    text_content = [line.strip() for line in textdata]
                    text = " ".join(text_content)
                    text = fix(text)

                if len(text) <= 3:
                    print "Too little text in document, Skipping"
                    continue
                    #raise ValueError("No Text")

                pos_features_path = "./temp/"

                runInParallel(stuff_LIWC_leftout(pid, text), ttr(text), POS_features("input", text, pos_features_path),
                              LIWC(text, cat_dict, stem_dict, counts_dict), vadersent(text), readability(text),
                              wordlen_and_stop(text), subjectivity(loaded_model, count_vect, tfidf_transformer, text),
                              bias_lexicon_feats(bias_lex, assertives_lex, factive_lexs, hedges_lex, implicatives_lex, report_verbs_lex, positive_op_lex, negative_op_lex, wneg_lex, wpos_lex, wneu_lex, sneg_lex, spos_lex, sneu_lex, text),
                              moral_foundation_feats(foundation_lex_dictionary, text))

                outseq = [pid, source, date]
                [outseq.append(feature_dictionary[feat]) for feat in feature_list]

                with open(os.path.join(outpath, outfile), "a") as out:
                    print "writing out"
                    outstr = make_str(outseq)
                    feat_str = ",".join(outstr)
                    out.write(feat_str + "\n")
                # except:
                #     print "Bad File, Skipping"
                #     continue
    

