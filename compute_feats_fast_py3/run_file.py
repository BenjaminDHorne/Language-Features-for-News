import csv
import os
import pickle
import warnings
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from time import sleep, time

from tqdm import tqdm

from compute_feats_fast_py3.run_functions import load_moral_foundations_lexicon, load_LIWC_dictionaries, fix, \
    stuff_LIWC_leftout, ttr, POS_features, vadersent, LIWC, wordlen_and_stop, bias_lexicon_feats, \
    subjectivity, moral_foundation_feats, make_str, load_acl13_lexicons, readability, utility_data_path, base_path
from compute_feats_fast_py3.src.tokenizing import NewTokenizer


class ArticleProcessor(Process):
    def __init__(self, worker_nr, input_queue: Queue, output_queue: Queue):
        super().__init__()
        self.worker_nr = worker_nr
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.daemon = True

    def __str__(self):
        return f"Worker({self.worker_nr})"

    def __repr__(self):
        return str(self)

    def run(self):
        ####
        # Prepare everything

        # happiness_lex = load_happiness_index_lexicon()
        foundation_lex_dictionary = load_moral_foundations_lexicon()
        (bias_lex, assertives_lex, factive_lexs, hedges_lex, implicatives_lex, report_verbs_lex, positive_op_lex,
         negative_op_lex, wneg_lex, wpos_lex, wneu_lex, sneg_lex, spos_lex, sneu_lex) = load_acl13_lexicons()
        bias_lex = set(bias_lex)
        assertives_lex = set(assertives_lex)
        factive_lexs = set(factive_lexs)
        hedges_lex = set(hedges_lex)
        implicatives_lex = set(implicatives_lex)
        report_verbs_lex = set(report_verbs_lex)
        positive_op_lex = set(positive_op_lex)
        negative_op_lex = set(negative_op_lex)
        wneg_lex = set(wneg_lex)
        wpos_lex = set(wpos_lex)
        wneu_lex = set(wneu_lex)
        sneg_lex = set(sneg_lex)
        spos_lex = set(spos_lex)
        sneu_lex = set(sneu_lex)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                loaded_model = pickle.load(open(os.path.join(utility_data_path, 'NB_Subj_Model.sav'), 'rb'))
            except UnicodeDecodeError:
                with open(os.path.join(utility_data_path, 'NB_Subj_Model.sav'), 'rb') as file:
                    loaded_model = pickle.load(file, encoding='latin1')

            try:
                count_vect = pickle.load(open(os.path.join(utility_data_path, 'count_vect.sav'), 'rb'))
            except UnicodeDecodeError:
                with open(os.path.join(utility_data_path, 'count_vect.sav'), 'rb') as file:
                    count_vect = pickle.load(file, encoding='latin1')

            try:
                tfidf_transformer = pickle.load(open(os.path.join(utility_data_path, 'tfidf_transformer.sav'), 'rb'))
            except UnicodeDecodeError:
                with open(os.path.join(utility_data_path, 'tfidf_transformer.sav'), 'rb') as file:
                    tfidf_transformer = pickle.load(file, encoding='latin1')

        cat_dict, stem_dict, counts_dict = load_LIWC_dictionaries()
        liwc_cats = [cat_dict[cat] for cat in cat_dict]
        pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS",
                    "PDT",
                    "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "WP$", "WRB", "VB", "VBD", "VBG",
                    "VBN", "VBP", "VBZ", "WDT", "WP"]

        feature_header_str = "HarmVirtue,HarmVice,FairnessVirtue,FairnessVice,IngroupVirtue,IngroupVice," \
                             "AuthorityVirtue," \
                             "AuthorityVice,PurityVirtue,PurityVice,MoralityGeneral,bias_count,assertives_count," \
                             "factives_count,hedges_count,implicatives_count,report_verbs_count,positive_op_count," \
                             "negative_op_count,wneg_count,wpos_count,wneu_count,sneg_count,spos_count,sneu_count," \
                             "TTR," \
                             "vad_neg,vad_neu,vad_pos,stop,wordlen,WC,NB_pobj,NB_psubj,quotes,Exclaim,AllPunc," \
                             "allcaps,FKE,SMOG," + ",".join(pos_tags) + "," + ",".join(liwc_cats)  #
        feature_list = feature_header_str.split(",")

        ####
        # Run everything
        while True:
            # Get job
            job = self.input_queue.get()

            # Look for poison
            if job is None:
                break

            # Split job
            job_nr, dir_name, file_name = job

            # Prepare
            path = dir_name + "/"
            feature_dictionary = dict()
            source = dir_name.split("\\")[-1]
            date = file_name.split("--")[1]
            pid = file_name.split(".")[0]

            # try:
            with open(path + file_name) as textdata:
                text_content = [line.strip() for line in textdata]
                text = " ".join(text_content)

            text = fix(text)

            if len(text) <= 3:
                continue

            pos_features_path = "./temp/"
            tokenizer = NewTokenizer()

            # Run all processes
            stuff_LIWC_leftout(pid=pid, text=text, feature_dictionary=feature_dictionary)

            ttr(text=text, feature_dictionary=feature_dictionary)

            POS_features(fn="input", text=text, outpath=pos_features_path, feature_dictionary=feature_dictionary)

            LIWC(text=text, cat_dict=cat_dict, stem_dict=stem_dict, counts_dict=counts_dict,
                 feature_dictionary=feature_dictionary)

            vadersent(text=text, feature_dictionary=feature_dictionary)  # readability(text),

            wordlen_and_stop(text=text, feature_dictionary=feature_dictionary)

            subjectivity(loaded_model=loaded_model, count_vect=count_vect, tfidf_transformer=tfidf_transformer,
                         text=text, feature_dictionary=feature_dictionary)

            bias_lexicon_feats(bias=bias_lex, assertives=assertives_lex, factives=factive_lexs, hedges=hedges_lex,
                               implicatives=implicatives_lex, report_verbs=report_verbs_lex,
                               positive_op=positive_op_lex, negative_op=negative_op_lex, wneg=wneg_lex,
                               wpos=wpos_lex,
                               wneu=wneu_lex, sneg=sneg_lex, spos=spos_lex, sneu=sneu_lex, text=text,
                               feature_dictionary=feature_dictionary,
                               tokenizer=tokenizer)

            moral_foundation_feats(foundation_lex_dictionary=foundation_lex_dictionary, text=text,
                                   feature_dictionary=feature_dictionary)

            readability(text=text, feature_dictionary=feature_dictionary)

            # Prepare name
            outseq = [pid, source, date] + [feature_dictionary[feat] for feat in feature_list]
            outstr = make_str(outseq)

            # Send result
            self.output_queue.put(outstr)


def run_save(csv_writer, data):
    for row in data:
        csv_writer.writerow(row)


def from_queue_2_file(a_queue: Queue, a_bar: tqdm, csv_writer, save_interval=20):

    # Initialize
    queue_data = []
    c_nr = 0

    # Run as long as there is anything in the queue
    while True:
        # Try and break on empty
        try:

            # Check for saving
            if c_nr >= save_interval:
                run_save(csv_writer=csv_writer, data=queue_data)
                queue_data = []
                c_nr = 0

            # Get some more data
            queue_data.append(a_queue.get(block=True, timeout=1.))
            c_nr += 1

            # Update bar
            a_bar.update()

        except Empty:
            break

    # Save remaining data
    run_save(csv_writer=csv_writer, data=queue_data)


if __name__ == "__main__":
    text_file_start_path = Path(base_path, "..", "data/").resolve()
    n_workers = 4

    # Make data queue
    data_queue = Queue()
    file_nr = 0
    for directory_name, _, file_list in os.walk(text_file_start_path):
        for fn in file_list:
            data_queue.put((file_nr, directory_name, fn))
            file_nr += 1

    # Insert poison
    for _ in range(n_workers):
        data_queue.put(None)

    start_time = time()

    # Results queue
    results_queue = Queue()

    # Make workers
    workers = []
    for w_nr in range(n_workers):
        worker = ArticleProcessor(
            worker_nr=w_nr,
            input_queue=data_queue,
            output_queue=results_queue,
        )
        worker.start()
        workers.append(worker)

    # Get results
    output_dir = Path("output_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(total=file_nr, desc="Processing files", unit="file") as bar:
        with Path(output_dir, "processed_data.csv").open("w") as save_file:
            writer = csv.writer(save_file, delimiter=",")
            while any([worker.is_alive() for worker in workers]):
                from_queue_2_file(a_queue=results_queue, a_bar=bar, csv_writer=writer)
            sleep(3)
            from_queue_2_file(a_queue=results_queue, a_bar=bar, csv_writer=writer)

    end_time = time()

    total_time = end_time - start_time
    print("\n"*3 + f"Total time: {total_time:.2f}s")
    print(f"Mean time per file: {total_time / file_nr:.4f}s")
