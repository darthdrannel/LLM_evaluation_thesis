from typing import Any, List, Literal, Optional
import uuid
from IPython.display import display_javascript, display_html
import json
from datetime import datetime

from ollama import GenerateResponse


class RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
        document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)

class Responses(List):

    def __init__(self):
        return super().__init__()

    def save_responses(self, filename):
        fp = open(f"{filename}-{datetime.today().strftime('%Y%m%d')}.json", "a")
        json.dump(self, fp)
        fp.close()

    def load_responses(self, filename):
        fp = open(f"{filename}-{datetime.today().strftime('%Y%m%d')}.json", "r")
        self.append(json.load(fp, ensure_ascii=False))
        fp.close()


# from enum import Enum
# from pydantic import BaseModel, Field


# class Target(str, Enum):
#     greater=">"
#     greater_equal="≥"
#     less_equal="≤"
#     less="<"

# class TargetValue(BaseModel):
#     threshold: int | float = Field(description="The value to compare the the measurements to.")
#     target: Target = Field(description="Describes if the Measurement should be larger or smaller than the given value.", examples=[">","≥","≤","<"])

# class Range(BaseModel):
#     lower: int | float = Field(description="The lower bound of the value range")
#     upper: int | float = Field(description="The upper bound of the value range")

# class Abbreviation(BaseModel):
#     abbreviation: str = Field(description="The abbreviation or acronym of a longer Phrase", min_length=1)
#     full_text: str = Field(description="The full Phrase of the abbreviation", min_length=1)

# class Measurement(BaseModel):
#     name: str | Abbreviation = Field(description="The name of the patient value that has been measured", examples=["Age", "Heigth", Abbreviation(abbreviation="BP", full_text="Blood Pressure")])
#     value: TargetValue | Range = Field(description="The measurement value.", examples=["120", "1.5", Range(lower=100, upper=200)])
#     unit: str = Field(description="The unit the value is measured in.", examples=["years","%", "ml", "mmHg"])

# class Measurements(BaseModel):
#     measurements: List[Measurement] = Field(description="A List of Measurements.")

# class Gender(Enum):
#     male="male"
#     female="female"
#     other="other"

# class Patient(BaseModel):
#     gender: str = Optional[Gender]
#     age: Optional[TargetValue | Range]


from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ObservationContext(BaseModel):
    name: str  # e.g., "Emergency Department"
    relevance: float  # 0.0 to 1.0


class PatientAge(BaseModel):
    operator: Optional[Literal["<", "<=", "=", ">=", ">"]] = None
    value: Union[float, List[float]]  # e.g., 65 or [20, 40]
    unit: Literal["years", "months", "days"]


class ValueCriterion(BaseModel):
    operator: Optional[Literal["<", "<=", "=", ">=", ">", "approx"]] = None
    value: Union[float, List[float]]
    unit: str  # e.g., "bpm", "%", "mmHg"


class Inference(BaseModel):
    label: Optional[str] = Field(examples=["low", "normal", "elevated", "high"])
    value_criterion: Optional[ValueCriterion] = None

class PatientInference(BaseModel):
    gender: Optional[Literal["female", "male", "nonbinary"]] = None
    age: Optional[PatientAge] = None
    inferences: List[Inference]

class Observation(BaseModel):
    name: str = Field(description="name of a patient value or state", examples=["Blood Pressure", "abdominal pain", "respiration rate"]) # e.g., "Respiratory Rate"
    context: List[str]
    patient_inferences: List[PatientInference]

class ObservationReference(BaseModel):
    name: str
    relevance_to_condition: float


class Condition(BaseModel):
    name: str  # e.g., "Sepsis"
    observations: List[str]


class ConditionKnowledgeGraph(BaseModel):
    observations: List[Observation]
    conditions: List[Condition]



# import outlines
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# class Model:
#     generator: Any

#     def generate(self, messages, **kwargs):
#         return self.generator(messages, **kwargs)

# class GeneralModel(Model):
#     generator: Any

#     def __init__(self, model_id):
#         self.generator = pipeline('text-generation', 
#                      model=AutoModelForCausalLM.from_pretrained(f"./Models/{model_id}/", torch_dtype=torch.bfloat16),
#                      tokenizer=AutoTokenizer.from_pretrained(f"./Models/{model_id}/"),
#                      device="cpu",
#                      temperature=0.5,
#                      top_p=0.9)

# class OutlinesModel(Model):
#     generator: Any

#     def __init__(self, model_id):
#         self.generator = outlines.from_transformers(
#             model=AutoModelForCausalLM.from_pretrained(f"./Models/{model_id}/", torch_dtype=torch.bfloat16),
#             tokenizer_or_processor=AutoTokenizer.from_pretrained(f"./Models/{model_id}/")
#         )
    
#     def generate(self, messages, **kwargs):
#         return super().generate(str(messages), **kwargs)

import pymupdf4llm

class Guideline:
    content: str = ""

    def __init__(self, file_name, start_page, end_page):
        self.content = pymupdf4llm.to_markdown(doc=file_name, pages=range(start_page - 1,end_page))
    
    def __str__(self):
        return self.content
    
    def __format__(self, format_spec):
        return f"{self.content:{format_spec}}"
    
from pathlib import Path

def save_response(prompt:str, response: GenerateResponse, options: dict[str, Any]):
    
    path = f"./{response.model}/ctx{options['num_ctx']}_t{options['temperature']}_k{options['top_k']}_p{options['min_p']}-{options['typical_p']}-{options['top_p']}_{response.created_at}.json"
    
    path = path.translate({ord(c): "_" for c in '*"\<>:|?'})

    file_path = Path(path)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    save = {
        'prompt': prompt,
        'response': json.loads(response.response),
        'options': options,
        'details': response.__dict__
    }

    save["details"]["context"] = []

    with open(file_path, 'w') as file:
        file.write(json.dumps(save))