import nltk
from nltk import tokenize
from nltk.util import ngrams
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from resources.readability import Readability
import collections
from nltk.stem.porter import *
from nltk import word_tokenize
import string
import pickle
from resources.feature_functions import Functions
import sqlite3

def writeout(feat_table):
    featdb = "features.db"
    conn = sqlite3.connect(featdb)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS features (id int, HarmVirtue real, HarmVice real, FairnessVirtue real, FairnessVice real, IngroupVirtue real, IngroupVice real, AuthorityVirtue real, AuthorityVice real, \
     PurityVirtue real, PurityVice real, MoralityGeneral real, bias_count real, assertives_count real, factives_count real, hedges_count real, implicatives_count real, report_verbs_count real, positive_op_count real, \
     negative_op_count real, wneg_count real, wpos_count real, wneu_count real, sneg_count real, spos_count real, sneu_count real, TTR,vad_neg real,vad_neu real,vad_pos real,FKE real,SMOG real,stop real,wordlen real, \
     WC real,NB_pobj real,NB_psubj real,quotes real,Exclaim real,AllPunc real,allcaps real, CC real,CD real,DT real,EX real,FW real,IN_pos real,JJ real,JJR real,JJS real,LS real,MD real,NN real,NNS real,NNP real,NNPS real,PDT real,POS real,PRP real,PRP$ real,RB real,RBR real,RBS real, \
     RP real,SYM real,TO_pos real,UH real,WP$ real,WRB real,VB real,VBD real,VBG real,VBN real,VBP real,VBZ real,WDT real,WP real,ingest real,cause real,insight real,cogmech real,sad real,inhib real,certain real,tentat real,discrep real,space real,time real,excl real,incl real,relativ real,motion real,quant real,number real, swear real, funct real, ppron real,\
     pronoun real,we real,i real,shehe real,you real,ipron real,they real,death real,	bio real,body real,	hear real,feel real,percept real, see real,	filler real,health real,sexual real,social real,family real,friend real,humans real,affect real,posemo real,negemo real,anx real,anger real,assent real,nonfl real, \
     verb real,	article real,past real,	auxverb real,future real, present real,	preps real,	adverb real,negate real,conj real,home real,leisure real,achieve real,work real,relig real,money real)''')
    for row in feat_table:
        c.execute('''INSERT INTO features VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,\
        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', tuple(row))
    conn.commit()


def make_str(seq):
    strseq = [str(s) for s in seq]
    return strseq

#main <-----------------------------------------------------------------------------------------------------------
Functions = Functions()
database_path = ""
cat_dict, stem_dict, counts_dict = Functions.load_LIWC_dictionaries()

table = []
chunk_size = 50
count = 0

db = "../Data/database.db"
conn = sqlite3.connect(db)
c = conn.cursor()
c.execute('''SELECT id,col1 FROM database''')
results = c.fetchall()

for tup in results:
    tid = tup[0]
    text = tup[1]

    #use if running with Python2:
    text = Functions.fix(text)

    pos_features_path = "./temp/"

    quotes, Exclaim, AllPunc, allcaps = Functions.stuff_LIWC_leftout(tid, text)
    lex_div = Functions.ttr(text)
    counts_norm = Functions.POS_features("input", text, pos_features_path)
    counts_norm = [str(c) for c in counts_norm]
    counts_norm_liwc, liwc_cats = Functions.LIWC(text, cat_dict, stem_dict, counts_dict)
    counts_norm_liwc = [str(c) for c in counts_norm_liwc]
    vadneg, vadneu, vadpos = Functions.vadersent(text)
    fke, SMOG = Functions.readability(text)
    stop, wordlen, WC = Functions.wordlen_and_stop(text)
    NB_pobj, NB_psubj = Functions.subjectivity(text)
    bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count = Functions.bias_lexicon_feats(text)
    HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral = Functions.moral_foundation_feats(text)
    #happiness = Functions.happiness_index_feats(text)

    seq = [tid, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count,\
    hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,\
    NB_psubj,quotes,Exclaim,AllPunc,allcaps] + counts_norm + counts_norm_liwc

    table.append(seq)
    count += 1

    if count == chunk_size:
        writeout(table)
        table = []
        count = 0
