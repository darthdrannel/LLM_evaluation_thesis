from copy import deepcopy
import json
from pathlib import Path
from pprint import pprint
import re
from jinja2 import DictLoader
import matplotlib.cbook as cbook

import numpy as np

boxplot_header = "lw lq med uq uw avg pos"

numbers_dict = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'vStruct': 0,
    'vJSON': 0,
    'total': 0
}

best_dict = {
    "temp": 0,
    "top_p": 0,
    "top_k": 0,
    "value": 0
}

def weird_div(a, b):
    return a / b if b else 0

def dict_add(a, b):
    for k, v in b.items():
        a[k] += v


def calc_numbers(values: dict):
    
    tp = values['true_positive']
    fp = values['false_positive']
    fn = values['false_negative']
    tn = values['true_negative']
    
    precision = weird_div(tp, tp + fp)
    recall = weird_div(tp, tp + fn)
    
    return {
        "accuracy": [weird_div(tn + tp, tp + fp + fn + tn)],
        "precision": [precision],
        "recall": [recall],
        "f1": [2 * weird_div(precision * recall, precision + recall)],
        'vJSON': 1 if values['valid_json'] else 0,
        'vStruct': 1 if values['valid_structure'] else 0,
        'total': 1
    }

def calculate_parameterwise_results(scores:dict):

    parameter_results = {
        "temp": {},
        "top_p": {},
        "top_k": {},
    }

    guidelines_results = {}

    model_variant = ["llama_V", "mistral_V", "gemma_V"]
    models_results = {}
    for variant in model_variant:
        models_results[variant] = deepcopy(numbers_dict)

    best_results = {
        "precision": {},
        "recall": {},
        "f1": {}
    }

    for top_p in map(str,np.arange(0.0,1.1,0.2)):
        if not parameter_results['top_p'].get(top_p):
            parameter_results['top_p'][top_p] = {}
        for temperature in map(str,np.arange(0.0, 1.1, 0.2)):
            if not parameter_results['temp'].get(temperature):
                parameter_results['temp'][temperature] = {}
            for top_k in [20,40,80]:
                if not parameter_results['top_k'].get(top_k):
                    parameter_results['top_k'][top_k] = {}

                params = f"t{temperature}_k{top_k}_p{top_p}"

                for guideline in scores:
                    variant_idx = 0

                    guideline_abbr = guideline[:3]
                    if not guidelines_results.get(guideline_abbr):
                        guidelines_results[guideline_abbr] = {}
                    
                    for model in scores[guideline]:
                        
                        model_abbr = ""
                        str_list = model.split("/")[-1].split("-")
                        if len(str_list) > 1:
                            model_abbr = str_list[0][:5] + str_list[1][:3]
                        else:
                            model_abbr = str_list[0][:5]


                        if not models_results.get(model_abbr):
                            models_results[model_abbr] = deepcopy(numbers_dict)
                        
                        if not guidelines_results[guideline_abbr].get(model_abbr):
                            guidelines_results[guideline_abbr][model_abbr] = deepcopy(numbers_dict)

                        values = scores[guideline][model][params]
                            
                        numbers = calc_numbers(values)

                        for metric in ['precision', 'recall', 'f1']:
                            if not best_results[metric].get(model_abbr):
                                best_results[metric][model_abbr] = deepcopy(best_dict)
                            if numbers[metric][0] > best_results[metric][model_abbr]['value']:
                                best_results[metric][model_abbr]['temp'] = temperature
                                best_results[metric][model_abbr]['top_p'] = top_p
                                best_results[metric][model_abbr]['top_k'] = top_k
                                best_results[metric][model_abbr]['value'] = numbers[metric][0]

                        for param, value in [('temp', temperature),('top_p', top_p),('top_k', top_k)]:
                            if not parameter_results[param][value].get(model_abbr):
                                parameter_results[param][value][model_abbr] = numbers
                            else:
                                dict_add(parameter_results[param][value][model_abbr], numbers)
                        
                        dict_add(guidelines_results[guideline_abbr][model_abbr], numbers)
                        dict_add(models_results[model_abbr], numbers)
                        dict_add(models_results[model_variant[variant_idx]], numbers)
                        variant_idx = (variant_idx + 1) % 3

    
    return parameter_results, guidelines_results, models_results, best_results

# transform to .dat tables

def boxplot_values(values:list, pos: int):
    boxplot = cbook.boxplot_stats(values, whis=(0,100))[0]
                    
    return " ".join(map(str,[boxplot['whislo'], boxplot['q1'], boxplot['med'], boxplot['q3'], boxplot['whishi'], boxplot['mean'], pos]))

def lines_to_file(lines:list, filename:str):
    with open(Path(f"{home_path}data/{filename}"), "w") as file:
            file.write('\n'.join(lines))

# output per parameter results
def parameter_results_to_file(results):
    for metric, params in results.items():
        lines_avg = [f"{metric}"]
        pos = 1
        lines_boxplot = {}
        first = True
        for param, models in params.items():
            
            s = f"{param}"
            for model, values in models.items():
                for key, value in values.items():
                    if first:
                        lines_avg[0] += f" {model}_{key}"

                    if key == 'total':
                        s += f" {value}"
                        continue

                    if type(value) == list:
                        
                        if not lines_boxplot.get(key):
                            lines_boxplot[key] = []

                        line = boxplot_values(value, pos)
                        
                        lines_boxplot[key].append(line)

                        accum = sum(value)
                    else:
                        accum = value
                    res = accum / values['total']
                    s += f" {res}"
                pos += 1
            first = False
            lines_avg.append(s)
            
            pos += 3

        for key, param in lines_boxplot.items():
            param.insert(0, boxplot_header)
            lines_to_file(param, f"bp_{metric}_{key}.dat")

        lines_to_file(lines_avg, f"avg_{metric}.dat")

def results_to_file(results, name:str):
    models_boxplot = {}
    models_avg = ["idx"]
    first = True
    pos = 1
    for id, values in results.items():
        s = f"{id}"
        for key, value in values.items():
        
            if first:
                models_avg[0] += f" {key}"

            if key == 'total':
                s += f" {value}"
                continue

            if type(value) == list:

                if not models_boxplot.get(key):
                    models_boxplot[key] = [boxplot_header]
                
                models_boxplot[key].append(boxplot_values(value, pos))

                accum = sum(value)
            else:
                accum = value
            res = accum / values['total']
            s += f" {res}"
        models_avg.append(s)
        pos += 1
        first = False

    lines_to_file(models_avg, f"avg_{name}.dat")
    for key, lines in models_boxplot.items():
        lines_to_file(lines, f"bp_{name}_{key}.dat")

def chatGPT_res():
    with open(Path(f"{home_path}results_ChatGPT.json"), "r") as file:
        scores_ChatGPT = json.load(file)

        scores_ChatGPT["total"] = deepcopy(scores_ChatGPT["hypertension"])
        for k,v in scores_ChatGPT["acute_appendicitis"].items():
            scores_ChatGPT["total"][k] = (scores_ChatGPT["total"][k] + v)/2 


        lines = ["target", "", "", ""]
        first = True
        i = 1
        for target,values in scores_ChatGPT.items():
            lines[i] += target
            numbers = calc_numbers(values)
            for k,v in numbers.items():
                if k != "total":
                    if first:
                        lines[0] += f" {k}"
                    if type(v) == list:
                        v = v[0]
                    lines[i] += f" {v}"
            i += 1
            first = False

        with open(Path(f"{home_path}data/results_ChatGPT.dat"), "w") as file:
            file.write("\n".join(lines))

def best_results_to_file(results):
    for metric, models in results.items():
        lines = ["model"]
        first = True
        for model, values in models.items():
            s = f"{model}"
            for k,v in values.items():
                if first:
                    lines[0] += f" {k}"
                s += f" {v}"
            lines.append(s)
            first = False
        lines_to_file(lines, f"best_{metric}.dat")

def guideline_results_to_file(results):
    guidelines_boxplot = {}
    pos = 1
    lines = ["MPG"]
    first = True
    for guideline, models in results.items():
        s = f"{guideline}"
        for model, values in models.items():
            for key,value in values.items():
                if first:
                    lines[0] += f" {model}_{key}"

                if key == 'total':
                    s += f" {value}"
                    continue


                if type(value) == list:

                    if not guidelines_boxplot.get(key):
                        guidelines_boxplot[key] = [boxplot_header]
                    
                    guidelines_boxplot[key].append(boxplot_values(value, pos))

                    accum = sum(value)
                else:
                    accum = value
                res = accum / values['total']
                s += f" {res}"
            pos += 1
        lines.append(s)
        pos += 3
        first = False

    for key, param in guidelines_boxplot.items():
        lines_to_file(param, f"bp_guidelines_{key}.dat")
    lines_to_file(lines, "avg_guidelines.dat")


home_path = f"{Path.home()}/BA/"
scores = {}
with open(Path(f"{home_path}results_final.json"), "r") as file:
    scores = json.load(file)

parameter_results, guidelines_results, models_results, best_results = calculate_parameterwise_results(scores)
parameter_results_to_file(parameter_results)
results_to_file(models_results, "models")
guideline_results_to_file(guidelines_results)
best_results_to_file(best_results)
chatGPT_res()

