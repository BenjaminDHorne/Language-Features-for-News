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

DIRNAME = os.path.dirname(__file__)

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

#main <-----------------------------------------------------------------------------------------------------------
Functions = Functions()
outfile = "NELA_Apr2017toSept2019-contentfeats.csv.csv"
outpath = "./"
done = whatsbeendon(outfile)
text_file_start_path = "../../NELADataCollection/DataTextOnly/Content/"

cat_dict, stem_dict, counts_dict = Functions.load_LIWC_dictionaries()
liwc_cats = [cat_dict[cat] for cat in cat_dict]
pos_tags = ["CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","WP$","WRB","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP"]

if len(done) == 0:
    with open(os.path.join(outpath, outfile), "a") as out:
        seq = ("pid, source, date, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, TTR,vad_neg,vad_neu,vad_pos,FKE,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps",",".join(pos_tags),",".join(liwc_cats))
        out.write(",".join(seq)+"\n")

for dirName, subdirList, fileList in os.walk(text_file_start_path):
    #print dirName, subdirList, fileList
    path = dirName+"/"
    c=0
    for fn in fileList:
        c+=1
        cat = dirName.split("/")[-1]#source
        date = "NA"#fn.split("--")[1]
        pid = fn.split(".")[0]

        if pid in done:
            continue
        else:
            print "working on", pid

        with open(path+fn) as textdata:
            text_content = [line.strip() for line in textdata]
            text = " ".join(text_content)
            text = Functions.fix(text)


        if len(text) == 0:
            continue
            #raise ValueError("No Text")
        try:
            pos_features_path = "./temp/"

            quotes, Exclaim, AllPunc, allcaps = Functions.stuff_LIWC_leftout(pid, text)
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

            with open(os.path.join(outpath, outfile), "a") as out:
                print "writing out"
                seq = (pid, cat, date, HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice, AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral, bias_count, assertives_count, factives_count, hedges_count, implicatives_count, report_verbs_count, positive_op_count, negative_op_count, wneg_count, wpos_count, wneu_count, sneg_count, spos_count, sneu_count, lex_div,vadneg,vadneu,vadpos,fke,SMOG,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc,allcaps, ",".join(counts_norm), ",".join(counts_norm_liwc))
                seq = make_str(seq)
                feat_str = ",".join(seq)
                out.write(feat_str + "\n")
        except:
            continue
