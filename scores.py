from copy import deepcopy
from difflib import SequenceMatcher
from itertools import combinations
import json
from pathlib import Path
from os import listdir
import fnmatch
import re
import time
from typing import Any, List, Literal
import numpy as np
from pydantic import ValidationError
from pydantic_core import from_json
from structurizer import ConditionKnowledgeGraph
from pprint import pprint
from scipy.optimize import linear_sum_assignment

similarity_score = 0.6


def find_best_fit(gt: str, targets: List[str]):
    return max(
        map(
            lambda t: (SequenceMatcher(None, gt.lower(), t[1].lower()).ratio(), t[0]),
            enumerate(map(lambda a: a if a != None else "", targets)),
        )
    )


def false_score(obj: Any):
    if obj is None:
        return 0
    if type(obj) in [str, int, float]:
        return 1
    if type(obj) == list:
        return sum(map(false_score, obj))
    if type(obj) == dict:
        return sum(map(false_score, obj.values()))


def list_compare(gt: list, target: list):
    tp = 0
    fp = []
    fn = []

    while gt and target:
        gt_item = gt.pop(0)
        score, index = find_best_fit(gt_item, target)
        if score > similarity_score:
            tp += score
            target.pop(index)
        else:
            fn.append(gt_item)

    if target:
        fp += target
        target.clear()
    if gt:
        fn += gt
        gt.clear()
    pprint([tp, fp, fn])
    return tp, fp, fn


def dict_list_compare(gt: list, target: list):
    tp = []
    fp = []
    fn = []

    while gt and target:
        key = "name" if "name" in gt[0] else "label"
        gt_item = gt.pop(0)
        target_names = list(map(lambda a: a.get(key, ""), target))
        score, index = find_best_fit(gt_item[key], target_names)
        if score > similarity_score:
            tp.append((gt_item, target.pop(index), score))
            target_names.pop(index)
        else:
            fn.append(gt_item)

    if target:
        fp += target
        target.clear()
    if gt:
        fn += gt
        gt.clear()

    pprint([tp, fp, fn])
    return tp, fp, fn


def single_compare(gt, target):
    if type(gt) != type(target):
        return 0
    if type(gt) == str:
        score = SequenceMatcher(None, gt.lower(), target.lower()).ratio()
        return score if score > similarity_score else 0
    if type(gt) == int or type(gt) == float:
        return 1 if gt == target else 0
    if type(gt) == list:
        return [single_compare(a, b) for a, b in zip(sorted(gt), sorted(target))]
    return 0

def dict_add(a, b):
    for k, v in b.items():
        a[k] += v
    
def rec_analysis(ground_truth, target):
    matrix = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0
    }

    if type(ground_truth) == type(target):
        if ground_truth == None:
            matrix["true_negative"] += 1
            return matrix
        if type(ground_truth) == str:
            score = SequenceMatcher(None, ground_truth.lower(), target.lower()).ratio()
            if score > similarity_score:
                matrix["true_positive"] += score
            else:
                matrix["false_positive"] += 1
                matrix["false_negative"] += 1
            return matrix

        if type(ground_truth) == int or type(ground_truth) == float:
            if ground_truth == target:
                matrix["true_positive"] += 1
            else:
                matrix["false_positive"] += 1
                matrix["false_negative"] += 1
            return matrix
            return matrix
        if type(ground_truth) == dict:
            for key in ground_truth:
                ret = rec_analysis(ground_truth[key], target.get(key))
                dict_add(matrix, ret)
            return matrix
        if type(ground_truth) == list:
            
            if not ground_truth:
                matrix["false_positive"] += false_score(target)
                return matrix
            if not target:
                matrix["false_negative"] += false_score(ground_truth)
                return matrix

            n = max(len(ground_truth), len(target))
            ground_truth_padded = ground_truth + [None] * (n - len(ground_truth))
            target_padded = target + [None] * (n - len(target))

            scores_list = [[None] * n for _ in range(n)]

            # Build cost matrix
            cost_matrix = np.zeros((n, n))
            for i, gt_item in enumerate(ground_truth_padded):
                for j, target_item in enumerate(target_padded):
                    if gt_item is None or target_item is None:
                        cost_matrix[i, j] = 0   # no benefit for pairing with dummy
                    else:
                        ret = rec_analysis(gt_item, target_item)
                        scores_list[i][j] = ret
                        cost_matrix[i, j] = -(ret['true_positive'] + ret['true_negative']) + ret['false_positive'] + ret['false_negative']


            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Extract best pairs
            best_match = []

            for i, j in zip(row_ind, col_ind):
                gt_item, target_item = ground_truth_padded[i], target_padded[j]
                if gt_item is not None and target_item is not None:
                    best_match.append((gt_item, target_item, scores_list[i][j]))
                    dict_add(matrix, scores_list[i][j])
                elif gt_item is not None:
                    matrix["false_negative"] += false_score(gt_item)
                elif target_item is not None:
                    matrix['false_positive'] += false_score(target_item)
            return matrix
    else:
        matrix["false_positive"] += false_score(target)
        matrix["false_negative"] += false_score(ground_truth)
        return matrix


def analysis(gt, target):
    valid = True

    try:
        ConditionKnowledgeGraph.model_validate_json(json.dumps(target))
    except ValidationError:
        valid = False

    matrix = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "valid_structure": valid,
    }

    # observations
    o_tp, o_fp, o_fn = dict_list_compare(gt["observations"], target.get("observations"))
    matrix["false_positive"] += false_score(o_fp)
    matrix["false_negative"] += false_score(o_fn)
    for gt_observation, t_observation, score in o_tp:
        matrix["true_positive"] += score
        print(matrix)
        pprint("NEW OBSERVATION")
        # context
        c_tp, c_fp, c_fn = list_compare(gt_observation["context"], t_observation.get("context"))
        matrix["true_positive"] += c_tp
        matrix["false_positive"] += false_score(c_fp)
        matrix["false_negative"] += false_score(c_fn)

        # patient inferences
        if not t_observation.get("patient_inferences"):
            matrix["false_negative"] += false_score(gt_observation["patient_inferences"])

        else:
            print(matrix)
            pprint("NEW INFERENCE")
            patinf_scores = []
            for t_patinf in t_observation["patient_inferences"]:
                gt_patinf = deepcopy(gt_observation["patient_inferences"][0])
                inf_score = {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "target": deepcopy(t_patinf),
                }
                if t_patinf.get(
                    "gender"
                ):  # shortcut, because we know that ground truth gender is always None
                    inf_score["false_positive"] += 1

                if t_patinf.get(
                    "age"
                ):  # shortcut, because we know that ground truth age is always None
                    inf_score["false_positive"] += 1

                # inferences
                i_tp, i_fp, i_fn = dict_list_compare(
                    gt_patinf["inferences"], t_patinf.get("inferences")
                )
                inf_score["false_positive"] += false_score(i_fp)
                inf_score["false_negative"] += false_score(i_fn)
                for gt_inf, t_inf, score in i_tp:
                    inf_score["true_positive"] += score

                    # value_criterion
                    gt_val = gt_inf.get("value_criterion")
                    t_val = t_inf.get("value_criterion")
                    if gt_val and t_val:
                        correct = 0
                        total = 0
                        for key in gt_val:
                            singles = single_compare(gt_val[key], t_val.get(key))
                            if type(singles) == list:
                                correct += sum(singles)
                                total += len(singles)
                            else:
                                correct += singles
                                total += 1
                        inf_score["true_positive"] += correct
                        inf_score["false_positive"] += total - correct
                        inf_score["false_negative"] += total - correct
                patinf_scores += [inf_score]
            best = max(patinf_scores, key=lambda x: x["true_positive"])
            patinf_scores.remove(best)
            pprint(patinf_scores)
            matrix["false_positive"] += false_score(list(map(lambda x: x["target"], patinf_scores)))
            matrix["true_positive"] += best["true_positive"]
            matrix["false_positive"] += best["false_positive"]
            matrix["false_negative"] += best["false_negative"]
            print(matrix)
    pprint("CONDITIONS")
    c_tp, c_fp, c_fn = dict_list_compare(gt["conditions"], target.get("conditions"))
    matrix["false_positive"] += false_score(c_fp)
    matrix["false_negative"] += false_score(c_fn)
    for gt, t, score in c_tp:
        pprint("NEW CONDITION")
        matrix["true_positive"] += score
        o_tp, o_fp, o_fn = list_compare(gt["observations"], t["observations"])
        matrix["true_positive"] += o_tp
        matrix["false_positive"] += false_score(o_fp)
        matrix["false_negative"] += false_score(o_fn)
        print(matrix)

    return matrix


home_path = f"{Path.home()}/BA/"

GLLiteral = Literal['hypertension', 'acute_appendicitis']

def score_all(save=True):

    results = {}

    for guideline, ground_truth in ground_truths.items():
        results[guideline] = {}
        for model in models:
            model_str = model.split("/")[-1]
            results[guideline][model_str] = {}
            model_path = f"{home_path}results_extra_old/{model}/"
            # get all filenames belonging to a guideline and model
            outputs = fnmatch.filter(listdir(model_path), f"{guideline}*.json")
            for output in outputs:
                # extract model parameters from filename
                params = re.findall("^.*_(t.+)\.json", output)[0]
                with open(Path(f"{model_path}{output}"), "r") as file:
                    # print(key, model_str, params)
                    json_string = file.read()
                    
                    valid_json = True
                    try:
                        target = json.loads(json_string)
                    except:
                        target = from_json(json_string, allow_partial="on")
                        valid_json = False

                    valid_structure = True
                    try:
                        ConditionKnowledgeGraph.model_validate_json(json.dumps(target))
                    except ValidationError:
                        valid_structure = False

                    matrix = rec_analysis(ground_truths[guideline], target)
                    matrix["valid_json"] = valid_json
                    matrix["valid_structure"] = valid_structure
                    results[guideline][model_str][params] = matrix

    if save:
        with open(Path(f"{home_path}results_{time.asctime()}.json"), "w") as file:
            json.dump(results, file)


def score_single_from_file(filepath:str, guideline:GLLiteral):
    with open(Path(filepath), "r") as file:
        json_string = file.read()

        valid_json = True
        try:
            target = json.loads(json_string)
        except:
            target = from_json(json_string, allow_partial="on")
            valid_json = False

        valid_structure = True
        try:
            ConditionKnowledgeGraph.model_validate_json(json.dumps(target))
        except ValidationError:
            valid_structure = False

        matrix = rec_analysis(ground_truths[guideline], target)
        matrix["valid_json"] = valid_json
        matrix["valid_structure"] = valid_structure
        return matrix


def score_single(guideline:GLLiteral, model:str, params:str):

    model_path = f"{home_path}results_extra_old/{model}/"

    return score_single_from_file(f"{model_path}{guideline}_{params}.json", guideline)

def score_ChatGPT():
    scores_ChatGPT = {}
    scores_ChatGPT['hypertension'] = score_single_from_file(f"{home_path}ChatGPT_acute_appendicitis.json", 'acute_appendicitis')
    scores_ChatGPT['acute_appendicitis'] = score_single_from_file(f"{home_path}ChatGPT_hypertension.json", 'hypertension')

    with open(Path(f"{home_path}data/results_ChatGPT.json"), "w") as file:
        json.dump(scores_ChatGPT, file)

ground_truths = {}


ground_truths["hypertension"] = json.load(
    open(Path(f"{home_path}hypertension_ground truth_p8.json"), "r")
)
ground_truths["acute_appendicitis"] = json.load(
    open(Path(f"{home_path}acute_appendicitis_ground truth_p3f.json"))
)

models = [
    "llama3.1_8b-instruct-q8_0",
    "mistral_7b-instruct-v0.2-q8_0",
    "gemma3_4b",
    "thewindmom/llama3-med42-8b_latest",
    "z-uo/llava-med-v1.5-mistral-7b_q8_0",
    "alibayram/medgemma_4b",
]


score_all()
# score_all(False)
# score_single("acute_appendicitis", models[2], "t0.8_k40_p0.2")
score_ChatGPT()

