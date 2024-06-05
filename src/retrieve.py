# import package ...
import sys
import gzip
import json
import math

"""
What do you think you might need to store and where would it make sense to store it?
- number of terms
- number of documents
- term counts
- document counts
"""

"""
helper functions to write to files
"""


def write_file_json(output_file: str, json_obj: json):
    with open(output_file, "wt", encoding="utf8") as outfile:
        for obj in json_obj["corpus"]:
            outfile.write("{obj}\n".format(obj=obj))
    return 0


def write_terms_postings(output_file: str, term_dict: [str, list[(int, int)]]):
    with open(output_file, "wt", encoding="utf8") as outfile:
        for term, postings in term_dict.items():
            outfile.write("{term}\n".format(term=term))
            for docid, pos in postings:
                outfile.write("{docid} - {pos}\n".format(docid=docid, pos=pos))
    return 0


def write_docids_lens(
    output_file: str, docid_dict: dict[str, int], doclens: dict[int, int]
):
    with open(output_file, "wt", encoding="utf8") as outfile:
        for str_id, docid in docid_dict.items():
            outfile.write(
                "{str_id}: {docid} {lens}\n".format(
                    str_id=str_id, docid=docid, lens=doclens[docid]
                )
            )
        # outfile.write(
        #     "max: \t{max}\nmin: \t{min}".format(
        #         max=max(doclens.values()), min=min(doclens.values())
        #     )
        # )
        outfile.write("total: \t {total}\n".format(total=doclens["total"]))
    return 0


def write_freqs_num_docs(output_file: str, freqs_num_docs: dict[str, int]):
    with open(output_file, "wt", encoding="utf8") as outfile:
        outfile.write("term\tmost freq\ttotal freq\tnum docs\n")
        for term, att in freqs_num_docs.items():
            outfile.write(
                "{term}\t{most_freq}\t{total_freq}\t{numdoc}\n".format(
                    term=term,
                    most_freq=att["most_freq"],
                    total_freq=att["total_freq"],
                    numdoc=att["num_doc"],
                )
            )
    return 0


def write_result(
    output_file: str, proc_ret_docs: list[(str, int)], qr_name: str
):
    with open(output_file, "at", encoding="utf8") as outfile:
        for ind, tup in enumerate(proc_ret_docs):
            str_id, score = tup[0], tup[1]
            outfile.write(
                "{qr_name}\tskip\t{str_id}\t{rank}\t{score:.4f}\tanhl\n".format(
                    qr_name=qr_name, str_id=str_id, rank=ind + 1, score=score
                )
            )
    return 0


def transform_docs(
    ret_docs: list[int] | dict[int, float], wordids: dict[int, str], bl: bool
):
    if bl is True:
        return sorted(
            [
                (get_storyID_from_docid(docid, wordids), 1.0)
                for docid, _ in ret_docs
            ]
        )
    return sorted(
        [
            (get_storyID_from_docid(docid, wordids), round(score, 4))
            for docid, score in ret_docs.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )


"""
Reader helper function
"""


def read_json_file(input_file: str):
    with gzip.open(input_file, "rt", encoding="utf8") as infile:
        json_str = infile.read()
        data = json.loads(json_str)
    return data


"""
function to assign each document to an integer index
"""


def create_docid(input_file: str):
    doc_id = {}
    cnt = 1
    data = read_json_file(input_file)
    # write_file_json("json_test.txt", data)
    for _, att in data.items():
        # att = corpus. Format: {corpus: [{...}, {...}, ...]}
        for obj in att:
            storyId = obj["storyID"]
            if storyId not in doc_id.keys():
                doc_id[storyId] = cnt
                cnt += 1
    return doc_id


"""def get_len_doc(
    input_file: str, docids: dict[str, int]
) -> dict[int | str, int]:
    doc_len = {}
    total_cnt = 0
    data = read_json_file(input_file)
    for att in data.values():
        for obj in att:
            docid = get_docid_from_storyID(obj["storyID"], docids)
            len_i = len(obj["text"])
            doc_len[docid] = len_i
            total_cnt += len_i
    doc_len["total"] = total_cnt  # |C|
    return doc_len"""


"""
API to get storyID from docid and vice versa
"""


def get_storyID_from_docid(docid: int, word_ids: dict[int, str]):
    return word_ids[docid]


def get_docid_from_storyID(story_id: str, docids: dict[str, int]):
    return docids[story_id]


"""
dict[term: str, postings:list[(int, int)]]
This func process 1 json object. Get Postings
Mainly for the side effect. Returns nothing
"""


def get_words_pos_doc(
    json_obj: dict[str, str],
    dictionary: dict[str, list[(int, int)]],
    docids: dict[str, int],
    doclens: dict[int, int],
) -> int:
    line = json_obj["text"].split()
    docid = get_docid_from_storyID(json_obj["storyID"], docids)
    doclens[docid] = len(line)
    pos_cnt = 0
    for word in line:
        pos_cnt += 1
        if word not in dictionary.keys():
            dictionary[word] = []
        dictionary[word].append((docid, pos_cnt))
    return len(line)


"""
function to process all json objects to get term dictionary of string keys and list<postings> value
"""


# process whole file with many json objects
def get_words_positions_dict(
    input_file: str, docids: dict[str, int], doclens: dict[int, int]
):
    term_dict = {}
    data = read_json_file(input_file)
    total_word_cnt = 0
    for _, att in data.items():
        for obj in att:
            total_word_cnt += get_words_pos_doc(
                obj, term_dict, docids, doclens
            )
        doclens["total"] = total_word_cnt
    return term_dict


"""
functions to process number of times a term occured in the each docid
term_dict has list of tuples (docid, pos)
"""


# dict: {term: [(docid, freq), ...]}
def get_word_freq_dict(
    term_dict: dict[str, list[(int, int)]]
) -> dict[str, dict[int, int]]:
    freq_dict = {}
    for term, postings in term_dict.items():
        d = {x: 0 for x, _ in postings}
        for docid, _ in postings:
            d[docid] += 1
        freq_dict[term] = d
    return freq_dict


"""
Extra function to get the largest frequency of the term in a doc and number of documents the term appears in
"""


def get_freq_num_docs(
    term_dict: dict[str, list[(int, int)]]
) -> dict[str, int]:
    freq_num_docs = {}
    for term, postings in term_dict.items():
        obj = {
            "most_freq": get_term_most_freq(postings),
            "total_freq": get_total_freq(postings),
            "num_doc": get_num_docs(postings),
        }
        freq_num_docs[term] = obj
    return freq_num_docs


def get_total_freq(postings: list[(int, int)]) -> int:
    return len(postings)


def get_term_most_freq(postings: list[(int, int)]) -> int:
    curr_most_freq = 0
    prev_docid = -1
    curr_freq = 0
    for posting in postings:
        if posting[0] != prev_docid:
            if curr_freq > curr_most_freq:
                curr_most_freq = curr_freq
            curr_freq = 0
        curr_freq += 1
        prev_docid = posting[0]
    return curr_most_freq


def get_num_docs(postings: list[(int, int)]) -> int:
    prev_docid = -1
    num_docs = 0
    for posting in postings:
        if prev_docid != posting[0]:
            num_docs += 1
        prev_docid = posting[0]
    return num_docs


"""
boolean retrieval
"""


def fin_bool_res(unproc_res: set):
    return [tuple([x, 1.0]) for x in unproc_res]


""""def intersect(lst1: list[int], lst2: list[int]):
    return [value for value in lst1 if value in lst2]"""


def boolean_and(
    words: list[str], freq_dict: dict[str, dict[int, int]]
) -> list[(int, int)]:
    all_docs = []
    for word in words:
        docs_with_word = [docid for docid in freq_dict[word].keys()]
        all_docs.append(docs_with_word)
    intersection = set.intersection(*map(set, all_docs))
    return fin_bool_res(intersection)
    """prev_intersect = [docid for docid, _ in freq_dict[words[0]].items()]
    for i in range(1, len(words)):
        next_lst = [docid for docid, _ in freq_dict[words[i]].items()]
        prev_intersect = intersect(prev_intersect, next_lst)
    set_intersect = set(prev_intersect)
    return fin_bool_res(set_intersect)"""


def boolean_or(words: list[str], freq_dict: dict[str, dict[int, int]]) -> list:
    all_docs = []
    for word in words:
        docs_with_word = [docid for docid in freq_dict[word].keys()]
        all_docs.extend(docs_with_word)
    union = set(all_docs)
    return fin_bool_res(union)


"""QL retrieval"""


# individual query
def ql_word(
    fqi_D: int,
    neu: int,
    c_qi: int,
    words_col: int,
    words_doc: int,
):
    return math.log((fqi_D + neu * (c_qi / words_col)) / (words_doc + neu))


def get_ql(
    words: list[str],
    freq_dict: dict[str, dict[int, int]],
    doclens: dict[int | str, int],
):
    ql_scores = {}
    neu = 300
    """for word in words:
        term_freq = freq_dict[word].items()
        for docid, freq in term_freq:
            if docid not in ql_scores.keys():
                ql_scores[docid] = 0
            c_qi = get_c_qi(freq_dict, word)
            ql_scores[docid] += ql_word(
                freq, neu, c_qi, doclens["total"], doclens[docid]
            )"""
    all_docs = boolean_or(words, freq_dict)
    for word in words:
        for docid, _ in all_docs:
            fi = 0
            if docid not in ql_scores.keys():
                ql_scores[docid] = 0
            if docid in freq_dict[word].keys():
                fi = freq_dict[word][docid]
            c_qi = get_c_qi(freq_dict, word)
            ql_scores[docid] += ql_word(
                fi, neu, c_qi, doclens["total"], doclens[docid]
            )
    return ql_scores


# get number of times word appear in collection
def get_c_qi(freq_dict: dict[str, dict[int, int]], word: str):
    if word not in freq_dict.keys():
        return 0
    return sum(freq_dict[word].values())


"""BM25 Retrieval"""


def get_avdl(doclens: dict[int, int]):
    return doclens["total"] / (len(doclens) - 1)


def calc_bigk(dl: int, k1: float, b: float, avdl: float):
    return k1 * ((1 - b) + b * (dl / avdl))


def bm25_word(
    N: int, ni: int, bigk: float, k1: float, k2: float, qfi: int, fi: int
):
    denom = (N - ni + 0.5) / (ni + 0.5)
    k1fi = ((k1 + 1) * fi) / (bigk + fi)
    k2qfi = ((k2 + 1) * qfi) / (k2 + qfi)
    return math.log(denom) * k1fi * k2qfi


def get_bm25(
    freq_dict: dict[str, dict[int, int]],
    doclens: dict[int, int],
    words: list[str],
):
    k1 = 1.8
    k2 = 5.0
    b = 0.75
    avdl = get_avdl(doclens)
    N = len(doclens) - 1
    qfs = [(word, words.count(word)) for word in words]
    bm25_scores = {}
    for word_i, qfi in qfs:
        term_freq = freq_dict[word_i].items()
        ni = len(term_freq)
        for docid, fi in term_freq:
            dl = doclens[docid]
            bigk = calc_bigk(dl, k1, b, avdl)
            if docid not in bm25_scores:
                bm25_scores[docid] = 0
            bm25_scores[docid] += bm25_word(N, ni, bigk, k1, k2, qfi, fi)
    return bm25_scores


"""
Starter code functions
"""


def buildIndex(inputFile, docids, doclens):
    # Your function start here ...
    # word_id = {v: k for k, v in docids.items()}
    # dict[term: str, List<Postings: (doc_id, positions)>]
    terms_dictionary = get_words_positions_dict(inputFile, docids, doclens)
    # freqs_num_docs = get_freq_num_docs(terms_dictionary)
    # write_terms_postings("terms_postings.txt", terms_dictionary)
    return terms_dictionary


def runQueries(index, queriesFile, outputFile, docids, doclens):
    freq_dict = get_word_freq_dict(index)
    # in_most_doc = [(len(pair), term) for term, pair in freq_dict.items()]
    # print("most doc: {most_doc}".format(most_doc=max(in_most_doc)[1]))
    # most_occs = [(sum(pair), term) for term, pair in freq_dict.items()]
    # print(
    #     "most occ: {most_occ}\t{sumo}".format(
    #         most_occ=max(most_occs)[1], sumo=max(most_occs)[0]
    #     )
    # )
    # only_once = [(len(pair), term) for term, pair in freq_dict.items()]
    # print("once: \n")
    # once_terms = []
    # for length, term in only_once:
    #     if length == 1:
    #         once_terms.append(term)
    # print(len(once_terms))
    # write_terms_postings("freq_dict.txt", freq_dict)
    word_id = {v: k for k, v in docids.items()}
    with open(queriesFile, "rt", encoding="utf8") as query_file:
        for line in query_file:
            res_qr = None
            qr = line.lower().split()
            qr_type, qr_name, qrs = qr[0], qr[1], qr[2:]
            if qr_type in ["and", "or"]:
                if qr_type == "and":
                    res_qr = boolean_and(qrs, freq_dict)
                if qr_type == "or":
                    res_qr = boolean_or(qrs, freq_dict)
                res_qr = transform_docs(res_qr, word_id, True)
            if qr_type in ["ql", "bm25"]:
                if qr_type == "ql":
                    res_qr = get_ql(qrs, freq_dict, doclens)
                if qr_type == "bm25":
                    res_qr = get_bm25(freq_dict, doclens, qrs)
                res_qr = transform_docs(res_qr, word_id, False)
            write_result(outputFile, res_qr, qr_name)
    return res_qr


if __name__ == "__main__":
    # Read arguments from command line, or use sane defaults for IDE.
    argv_len = len(sys.argv)
    inputFile = (
        sys.argv[1]
        if argv_len >= 2
        else "C:\\Users\\lemin\\p3-446\\P3python\\sciam.json.gz"
    )
    queriesFile = (
        sys.argv[2]
        if argv_len >= 3
        else "C:\\Users\\lemin\\p3-446\\P3python\\P3train_copy.tsv"
    )
    outputFile = (
        sys.argv[3]
        if argv_len >= 4
        else "C:\\Users\\lemin\\p3-446\\P3python\\P3train1.trecrun"
    )
    doclen_dict = {}
    docid_dict = create_docid(inputFile)
    index = buildIndex(inputFile, docid_dict, doclen_dict)
    write_docids_lens("docid.txt", docid_dict, doclen_dict)
    if queriesFile == "showIndex":
        # Invoke your debug function here (Optional)
        print("running showindex")
    elif queriesFile == "showTerms":
        # Invoke your debug function here (Optional)
        print("running showTerms")
    else:
        runQueries(index, queriesFile, outputFile, docid_dict, doclen_dict)

    # Feel free to change anything
