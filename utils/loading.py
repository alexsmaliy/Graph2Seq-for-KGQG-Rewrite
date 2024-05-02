import json
from collections import defaultdict, OrderedDict
from typing import cast, Literal, Optional, TypedDict, Union

from nltk.tokenize import wordpunct_tokenize
import numpy as np

import config
from utils.logging import Logger
from utils.strings import EOS_TOKEN, normalize_string

##############
# DATA TYPES #
##############
class RawInGraph(TypedDict):
    g_node_names: dict[str, str]
    g_edge_types: dict[str, str]
    # map from node x to each of its neighbors a/b/c, with possibly multiple edges to each neighbor
    g_adj: dict[str, dict[str, Union[str, list[str]]]]

class RawAnswerData(TypedDict):
    answers: list[str]
    answer_ids: Optional[list[str]]
    outSeq: str
    qId: int
    inGraph: RawInGraph

Indicator = Literal[1] | Literal[2]

class PreparedGraph(TypedDict):
    g_node_ids: dict[str, int]
    g_node_name_words: list[str]
    g_node_type_words: list[str]
    g_node_type_ids: list[str]
    g_node_ans_match: list[Indicator]
    g_edge_type_words: list[str]
    g_edge_type_ids: list[str]
    # map from src node index to dest node idx
    g_adj: defaultdict[int, dict[int, int]]
    num_virtual_nodes: int
    num_virtual_edges: int

# original authors like to reassign different data-shapes to the same field
# here's a type to capture that difference
class SeqGraph(TypedDict):
    g_node_ids: dict[str, int]
    g_node_name_words: list[list[str]] # different from PreParedGraph
    g_node_type_words: list[list[str]] # different from PreParedGraph
    g_node_type_ids: list[str]
    g_node_ans_match: list[Indicator]
    g_edge_type_words: list[list[str]] # different from PreParedGraph
    g_edge_type_ids: list[str]
    # map from src node index to dest node idx
    g_adj: defaultdict[int, dict[int, int]]
    num_virtual_nodes: int
    num_virtual_edges: int

DataInstance = tuple["SeqWithGraph", "SeqWithStr", list["SeqWithStr"]]
Dataset = list[DataInstance]

class Datasets(TypedDict):
    train: Dataset
    dev: Dataset
    test: Dataset

##############
# OPERATIONS #
##############
def _tokenize_list(lst: list[str]) -> list[list[str]]:
    return [wordpunct_tokenize(w.lower()) for w in lst]

class SeqWithGraph(object):
    def __init__(self, data: PreparedGraph):
        graph = {
                **data,
                "g_node_name_words": _tokenize_list(data['g_node_name_words']),
                "g_node_type_words": _tokenize_list(data["g_node_type_words"]),
                "g_edge_type_words": _tokenize_list(data["g_edge_type_words"]),
            }
        graph = cast(SeqGraph, graph)
        self.graph = graph

class SeqWithStr(object):
    def __init__(self, data: str, /, end_sym=None):
        self.graph = None
        data = cast(str, data)
        lower = data.lower()
        toks = wordpunct_tokenize(lower)
        self.src = " ".join(toks)
        self.tokText = lower
        self.words = toks
        if end_sym is not None:
            self.tokText = f"{self.tokText} {end_sym}"
            self.words.append(end_sym)


# class Seq(object):
#     def __init__(self, data: PreparedGraph | str, /, is_graph=False, end_sym=None):
#         if is_graph:
#             data = cast(PreparedGraph, data)
#             graph = {
#                 **data,
#                 "g_node_name_words": _tokenize_list(data['g_node_name_words']),
#                 "g_node_type_words": _tokenize_list(data["g_node_type_words"]),
#                 "g_edge_type_words": _tokenize_list(data["g_edge_type_words"]),
#             }
#             graph = cast(SeqGraph, graph)
#             self.graph = graph
#         else:
#             self.graph = None
#             data = cast(str, data)
#             lower = data.lower()
#             toks = wordpunct_tokenize(lower)
#             self.src = " ".join(toks)
#             self.tokText = lower
#             self.words = toks
#             if end_sym is not None:
#                 self.tokText = f"{self.tokText} {end_sym}"
#                 self.words.append(end_sym)

def _blank_graph() -> PreparedGraph:
    return {
        "g_node_ids": {},
        "g_node_name_words": [],
        "g_node_type_words": [],
        "g_node_type_ids": [],
        "g_node_ans_match": [],
        "g_edge_type_words": [],
        "g_edge_type_ids": [],
        "g_adj": defaultdict(dict),
        "num_virtual_nodes": 0,
        "num_virtual_edges": 0,
    }

def _normalize_edge_type(raw_edge: str) -> str:
    """'place_of_birth' or '/people/person/place_of_birth' => 'place of birth'"""
    e = raw_edge.split("/")
    e = e[-1].split("_")
    return " ".join(e)

def _levi_graph_transform(struct: RawAnswerData, graph: PreparedGraph):
    """
        Levi graph: each edge is a new node
        X ----(Y)--> Z   =>   X --> Y --> Z
    """
    num_nodes = len(graph["g_node_ids"])
    edge_index = num_nodes
    virtual_edge_index = 0

    for src_node, neighbor_dict in struct["inGraph"]["g_adj"].items():
        source_idx = graph["g_node_ids"][src_node]
        for dest_node, edge_info in neighbor_dict.items():
            dest_idx = graph["g_node_ids"][dest_node]
            edge_id_list = [edge_info] if isinstance(edge_info, str) else edge_info
            adj = graph["g_adj"]
            for edge_id in edge_id_list:
                adj[source_idx][edge_index] = virtual_edge_index
                adj[edge_index][dest_idx] = virtual_edge_index + 1
                virtual_edge_index += 2

                raw_edge_type = struct["inGraph"]["g_edge_types"][edge_id]
                normal_edge_type = _normalize_edge_type(raw_edge_type)
                graph["g_edge_type_ids"].append(raw_edge_type)
                graph["g_edge_type_words"].append(normal_edge_type)
                edge_index += 1

    assert len(graph["g_edge_type_words"]) == edge_index - num_nodes
    graph["num_virtual_nodes"] = edge_index
    graph["num_virtual_edges"] = virtual_edge_index

def _populate_graph_from_struct(
    struct: RawAnswerData, graph: PreparedGraph,
    normalized_answers: set[str], answer_ids: set[str],
):
    # this is the vector encoding if a node is an expected answer
    answer_or_not = graph["g_node_ans_match"]

    for index, (node_key, node_name) in enumerate(struct["inGraph"]["g_node_names"].items()):
        graph["g_node_ids"][node_key] = index
        graph["g_node_name_words"].append(node_name)

        if len(answer_ids) > 0:
            indicator: Indicator = 1 if node_key in answer_ids else 2
        else:
            indicator: Indicator = 1 if normalize_string(node_name) in normalized_answers else 2
        answer_or_not.append(indicator)

    assert any(x == 1 for x in answer_or_not), \
        f"No matching answers! RAW:\n{json.dumps(struct, indent=3)} PARSED:\n{json.dumps(graph, indent=3)}"

def _process_line(line: str) -> tuple[PreparedGraph, str, list[str]]:
    struct: RawAnswerData = json.loads(
        line.strip(),
        object_pairs_hook=lambda tuples: OrderedDict(tuples),
    )

    assert len(struct.get("inGraph", {}).get("g_adj", {})) > 0, \
        f"Bad input: {line}"

    answers = struct["answers"]
    normalized_answers = set(normalize_string(answer) for answer in answers)

    answer_ids = set(struct.get("answer_ids", []))
    out_seq = struct["outSeq"]

    graph = _blank_graph()
    _populate_graph_from_struct(struct, graph, normalized_answers, answer_ids)
    _levi_graph_transform(struct, graph)
    return graph, out_seq, answers

def _get_dataset(data_fpath: str) -> tuple[Dataset, list[int]]:
    all_instances: list[DataInstance] = []
    all_seq_lens: list[int] = []
    with open(data_fpath, "r") as data_file:
        for line in data_file:
            graph, out_seq, answers = _process_line(line)
            graph_seq = SeqWithGraph(graph)
            out_seq_seq = SeqWithStr(out_seq, end_sym=EOS_TOKEN)
            answer_seqs = [SeqWithStr(answer) for answer in answers]
            all_instances.append((
                graph_seq,
                out_seq_seq,
                answer_seqs, # heterogenous list
            ))
            all_seq_lens.append(len(out_seq_seq.words))
    return all_instances, all_seq_lens

def _log_loading_diagnostics(stuff: tuple[str, Dataset, list[int]], logger: Logger):
    name, dataset, lengths = stuff
    logger.log(f"# of {name} examples: {len(dataset)}")
    logger.log(f"max {name} seq length: {np.max(lengths)}")
    logger.log(f"min {name} seq length: {np.min(lengths)}")
    logger.log(f"mean {name} seq length: {np.mean(lengths)}")

def get_datasets(logger: Logger) -> Datasets:
    train_set, train_seq_lens = _get_dataset(config.TRAINING_DATASET)
    dev_set, dev_seq_lens = _get_dataset(config.DEVELOP_DATASET)
    test_set, test_seq_lens = _get_dataset(config.TESTING_DATASET)
    loggables = [
        ("training", train_set, train_seq_lens),
        ("dev", dev_set, dev_seq_lens),
        ("test", test_set, test_seq_lens)
    ]
    for loggable in loggables:
        _log_loading_diagnostics(loggable, logger)
    return {
        "train": train_set,
        "dev": dev_set,
        "test": test_set,
    }

# WQ SAMPLE RAW ENTRY
# {
#     'g_node_ids': {
#         '/m/06mt91': 0,
#         '/m/09g79g8': 1,
#         '/m/02p5kp': 2
#     },
#     'g_node_name_words': [
#         'Rihanna',
#         'Rihanna: Live in Concert Tour',
#         'Saint Michael Parish'
#     ],
#     'g_node_type_words': [],
#     'g_node_type_ids': [],
#     'g_node_ans_match': [2, 2, 2],
#     'g_edge_type_words': [
#         'concert tours',
#         'place of birth'
#     ],
#     'g_edge_type_ids': [
#         '/music/artist/concert_tours',
#         '/people/person/place_of_birth'
#     ], 'g_adj': defaultdict(<class 'dict'>, {
#         0: {3: 0, 4: 2},
#         3: {1: 1},
#         4: {2: 3}
#     }),
#     'num_virtual_nodes': 5,
#     'num_virtual_edges': 4
# }

# {
#     'g_node_ids': {
#         '/m/06mt91': 0,
#         '/m/09g79g8': 1,
#         '/m/02p5kp': 2
#     },
#     'g_node_name_words': [
#         'Rihanna',
#         'Rihanna: Live in Concert Tour',
#         'Saint Michael Parish'
#     ],
#     'g_node_type_words': [],
#     'g_node_type_ids': [],
#     'g_node_ans_match': [2, 2, 1],
#     'g_edge_type_words': [
#         'concert tours',
#         'place of birth'
#     ],
#     'g_edge_type_ids': [
#         '/music/artist/concert_tours',
#         '/people/person/place_of_birth'
#     ],
#     'g_adj': defaultdict(<class 'dict'>, {
#         0: {3: 0, 4: 2},
#         3: {1: 1},
#         4: {2: 3}
#       }),
#       'num_virtual_nodes': 5,
#       'num_virtual_edges': 4
#   }


# {
#     "answers": ["Saint Michael Parish"],
#     "answer_ids": ["/m/02p5kp"],
#     "outSeq": "where was the main artist featured in the rihanna : live in concert tour raised ?",
#     "qId": 21037,
#     "inGraph": {
#         "g_node_names": {
#             "/m/06mt91": "Rihanna",
#             "/m/09g79g8": "Rihanna: Live in Concert Tour",
#             "/m/02p5kp": "Saint Michael Parish"
#         },
#         "g_edge_types": {
#             "/music/artist/concert_tours": "/music/artist/concert_tours",
#             "/people/person/place_of_birth": "/people/person/place_of_birth"
#         },
#         "g_adj": {
#             "/m/06mt91": {
#                 "/m/09g79g8": "/music/artist/concert_tours",
#                 "/m/02p5kp": "/people/person/place_of_birth"
#             }
#         }
#     }
# }

# PQ SAMPLE RAW ENTRY
# {
#     "answers": ["honolulu"],
#     "outSeq": "the place of death of kalama 's husband ?",
#     "qId": 12,
#     "inGraph": {
#         "g_node_names": {
#             "kalama": "kalama",
#             "kamehameha_iii": "kamehameha iii",
#             "honolulu": "honolulu"
#         },
#         "g_edge_types": {
#             "spouse": "spouse",
#             "place_of_death": "place_of_death"
#         },
#         "g_adj": {
#             "kalama": {
#                 "kamehameha_iii": ["spouse"]
#             },
#             "kamehameha_iii": {
#                 "honolulu": ["place_of_death"]
#             }
#         }
#     }
# }