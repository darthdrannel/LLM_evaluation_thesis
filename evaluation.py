from pathlib import Path
from typing import Literal
from urllib import response
from structurizer import ConditionKnowledgeGraph
from ollama import AsyncClient, GenerateResponse
import numpy as np
import asyncio
import time


client = AsyncClient(
    host='http://localhost:11434'
)

schema = ConditionKnowledgeGraph

guidelines = {
    "hypertension": """
2. Introduction
This  2024  document  updates  the  2018  ESC/European  Society  of
Hypertension (ESH) Guidelines on the management of arterial hyper-tension.1 While  the  current  document  builds  on  prior  guidelines,  it
also incorporates important updates and new recommendations based
on current evidence. For example: (1) The title has changed from ‘Guidelines on the management of ar-terial hypertension’ to ‘Guidelines on the management of elevated
blood pressure and hypertension’. This is based on evidence that
the  risk  for  cardiovascular  disease  (CVD)  attributable  to  blood
pressure (BP) is on a continuous exposure scale, not a binary scale
of  normotension  vs.  hypertension.2,3 Updated  evidence  also  in-creasingly  demonstrates  the  benefit  on  CVD  outcomes  of
BP-lowering medications among persons with high CVD risk and
BP levels that are elevated but that do not meet traditional thresh-olds  used  to  define  hypertension.  The  term  ‘arterial’  is  removed
from the title of the 2024 Guidelines, as arterial hypertension can
also occur in the pulmonary arteries, which is not a focus here.(2) The 2024 Guidelines continue to define hypertension as office sys-tolic BP of ≥140 mmHg or diastolic BP of ≥90 mmHg. However, a
new BP category called ‘Elevated BP’ is introduced. Elevated BP is
defined as an office systolic BP of 120–139 mmHg or diastolic BP
of 70–89 mmHg.(3) A major, evidence-based change in the 2024 Guidelines is the rec-ommendation  to  pursue  a  target  systolic  BP  of  120–129 mmHg
among adults receiving BP-lowering medications. There are several
important  caveats  to  this  recommendation,  including:  (i)  the  re-quirement  that  treatment  to  this  BP  target  is  well  tolerated  by
the patient, (ii) the fact that more lenient BP targets can be consid-ered in persons with symptomatic orthostatic hypotension, those
aged  85  years  or  over,  or  those  with  moderate-to-severe  frailty
Table 2  Levels of evidenceLevel of
evidence A
Level of
evidence B
Level of
evidence C
Data derived from multiple randomized clinical trials
or meta-analyses.
Data derived from a single randomized clinical trial
or large non-randomized studies.
Consensus of opinion of the experts and/or small studies,
retrospective studies, registries.
©ESC 2024
©ESC2024
ESC Guidelines                                                                                                                                                                                           3919
Downloaded from https://academic.oup.com/eurheartj/article/45/38/3912/7741010 by guest on 18 June 2025
or  limited  life  expectancy,  and  (iii)  a  strong  emphasis  on
out-of-office  BP  measurement  to  confirm  the  systolic  BP  target
of 120–129 mmHg is achieved. For those selected individual cases
where a target systolic BP of 120–129 mmHg is not pursued, either
due  to  intolerance  or  the  existence  of  conditions  that  favour  a
more  lenient  BP  target,  we  recommend  targeting  a  BP  that  is  as
low as reasonably achievable. Personalized clinical decision-making
and shared decisions with the patient are also emphasized.(4) Another  important  change  in  the  2024  Guidelines  compared  with
earlier  versions  is  the  increased  focus  on  evidence  related  to  fatal
and non-fatal CVD outcomes  rather  than surrogate outcomes  such
as  BP  lowering  alone.  Except  for  lifestyle  interventions  and  low-risk
non-pharmacological interventions aimed at  implementation or  care
delivery, the current guidelines require that, for a Class I recommen-dation to be made for a drug or procedural intervention, the evidence
must show benefit on CVD outcomes and not only BP lowering.(5) The  task  force  comprised  of  a  balanced  representation  of  males
and females.(6) The present guidelines consider sex and gender as an integral com-ponent throughout the document, rather than in a separate section
at the end. In this document, sex is the biological condition of being
female or male from conception, based on genes, and gender is the
socio-cultural dimension of being a woman or a man in a given soci-ety, based on gender roles, gender norms, gender identity, and gen-der relations valid in the respective society at a given timepoint.4,5
(7) The  2024  Guidelines  are  written  to  make  them  more  ‘user  friendly’.
Input  from  general  practitioners  (GPs)  was  obtained  in  this  regard,
and  one  task  force  member  is  a  GP.  Given  the  ageing  population  in
Europe, there was also a focus on tailoring treatment with respect to
frailty  and  into  older  age,  which  is  addressed  in  multiple  sections.
Moreover,  patient  input  and  their  lived  experiences  are  considered
throughout.  We  also  now  include  evidence  tables  in  the
Supplementary  section  to  provide  improved  transparency  regarding
our recommendations. As appropriate, readers who wish to seek add-itional details and information are referred to the Supplementary data
online and to the  ESC CardioMed.6
(8) The task force recognized that a major challenge in guideline usage
is poor implementation. This likely contributes to suboptimal con-trol of hypertension.7–9 To address this, a dedicated section on im-plementation  is  included  in  the  Supplementary  data  online.
Moreover, through a new initiative, we include information from
national  societies following a survey on guideline  implementation
completed during the national society peer review of the guidelines
document. It is hoped this information may help inform national so-cieties about potential barriers to implementation.
2.1. What is new
These 2024 Guidelines contain a number of new and revised recom-mendations, which are summarized in Tables 3 and 4, respectively.
Table 3  New recommendations
Recommendations Classa Levelb
5. Measuring blood pressure
It is recommended to measure BP using a validated and calibrated device, to enforce the correct measurement technique, and to apply a
consistent approach to BP measurement for each patient. I B
Out-of-office BP measurement is recommended for diagnostic purposes, particularly because it can detect both white-coat hypertension
and masked hypertension. Where out-of-office measurements are not logistically and/or economically feasible, then it is recommended that
the diagnosis be confirmed with a repeat office BP measurement using the correct standardized measurement technique.
I B
Most automated oscillometric monitors have not been validated for BP measurement in AF; BP measurement should be considered using a
manual auscultatory method in these circumstances, where possible. IIa C
An assessment for orthostatic hypotension (≥20 systolic BP and/or ≥10 diastolic BP mmHg drop at 1 and/or 3 min after standing) should be
considered at least at the initial diagnosis of elevated BP or hypertension and thereafter if suggestive symptoms arise. This should be
performed after the patient is first lying or sitting for 5 min.
IIa C
""",
"acute_appendicitis":"""
TABLE 2
Diagnostic Tools for the Evaluation of Suspected Appendicitis
Alvarado score  Sign/symptom Points
Migration of pain 1
Anorexia 1
Nausea/vomiting 1
Right lower quadrant tenderness 2
Rebound pain 1
Temperature ≥ 37.3°C (99.1°F) 1
Leukocytosis ≥ 10,000 per μL (10.0 × 109 per L)2PMN ≥ 75% 1
Total possible score 10

CRP = C-reactive protein;  PMN = polymorphonucleocytes.
Information from references 10 through 12.
Pediatric Appendicitis Score  Sign/symptom Points
Migration of pain 1
Anorexia 1
Nausea/vomiting 1
Right lower quadrant tenderness 2
Rebound pain 2
Right lower quadrant pain with coughing/hopping/percussion 2
Temperature ≥ 38°C (100.4°F) 1
Leukocytosis ≥ 10,000 per μL 1PMN ≥ 75% 1
Total possible score 12

Appendicitis Inflammatory Response score  Sign/symptom Points
Vomiting 1
Right Iliac fossa pain 1
Rebound pain, light 1
Rebound pain, medium 2
Rebound pain, strong 3
Temperature ≥ 38.5°C (101.3°F) 1
Leukocytosis ≥ 10,000 to 14,900 per μL (10.0 to 14.9 × 109 per L) 1
Leukocytosis ≥ 15,000 per μL (15.0 × 109 per L) 2
PMN 70% to 84% 1
PMN ≥ 85% 2
CRP 10 to 49 g per L 1
CRP ≥ 50 g per L 2
Total possible score 12
"""
}

prompt = f"""
You are a clinical information extraction assistant. Your task is to read medical practice guidelines and extract structured information to populate a observation-centric knowledge graph.

You must produce a JSON object containing two lists:
1. A list of `observations` containing information about the patient status, each including a list of possible source contexts and patient-specific value-based inferences.
2. A list of `conditions`, each refering to relevant observations by name.

JSON SCHEMA:

// RULE: If `value` is a single number, operator is REQUIRED
// If `value` is a range (e.g., [20, 30]), operator MUST be omitted
{schema.model_json_schema()}
"""
"""
---

EXTRACTION RULES:

- Only use terminology directly describing the measurement result.

- Only extract directly observable signs, symptoms, or measurement results (e.g., "temperature", "leukocytosis", "nausea").

- DO NOT extract scores or composite tools (e.g., “Alvarado Score”, “Pediatric Appendicitis Score”) as observations. These are derived from multiple patient findings and are not valid for the observations list.

- Extract the components of scores (e.g., “migration of pain”, “anorexia”, “rebound pain”) individually if they describe a direct clinical observation.

- Do not duplicate observation details in the condition block — only reference them by name.

- Use the `context` to describe the **location where the patient is currently being diagnosed** (e.g., "ICU", "Inpatient", "Primary Care").

- If no such setting is mentioned, leave the context list empty.

- `inference.label` should describe the **interpretation of the observation** in relation to the normal value, not the **severity of the condition**.

- Avoid condition-centric language in the observation labels.

- Omit `operator` if `value` is a range.

- Use `null` for gender or age if not defined.

- Use exact terminology from the guideline for observation names and condition names.

---

EXAMPLE:

Guideline Input:
> “In patients over 60, a heart rate between 90 and 120 bpm is considered elevated in the ICU in cases of cardiac instability.”

Extracted Output:
```json
{
  "observations": [
    {
      "name": "Heart Rate",
      "context": [
        "ICU"
      ],
      "patient_inferences": [
        {
          "gender": null,
          "age": {
            "value": 60,
            "unit": "years",
            "operator": ">"
          },
          "inferences": [
            {
              "label": "elevated",
              "value_criterion": {
                "value": [90, 120],
                "unit": "bpm"
              }
            }
          ]
        }
      ]
    }
  ],
  "conditions": [
    {
      "name": "Cardiac Instability",
      "observations": [
          "Heart Rate"
        }
      ]
    }
  ]
}
```
---

GUIDELINE:


"""

models = [
    'llama3.1:8b-instruct-q8_0',
    'thewindmom/llama3-med42-8b:latest',
    'mistral:7b-instruct-v0.2-q8_0',
    'z-uo/llava-med-v1.5-mistral-7b_q8_0',
    'gemma3:4b',
    'alibayram/medgemma:4b'
]

sem = asyncio.Semaphore(2)

def save(response: GenerateResponse,guideline_name: str, options: dict):
  path = f"./results_extra/{response.model}/{guideline_name}_t{options['temperature']}_k{options['top_k']}_p{options['top_p']}.json"

  path = path.translate({ord(c): "_" for c in '*"\<>:|?'})

  file_path = Path(path)
  file_path.parent.mkdir(exist_ok=True, parents=True)
  with open(file_path, 'w') as file:
        file.write(response.response)


async def evaluate(model: str, guideline_name:Literal["hypertension", "acute_appendicitis"], options:dict):
  print(f"[{time.strftime('%X')}]{model} - {guideline_name}: Starting t={options['temperature']},p={options['top_p']}k={options['top_k']}")

  before = time.time()
  response = await client.generate(model=model,
    prompt=f"{prompt}{guidelines[guideline_name]}",
    options=options,
    format=schema.model_json_schema())

  diff = time.time()- before
  print(f"[{time.strftime('%X')}]{model} - {guideline_name}: saving after {time.strftime('%X', time.localtime(diff))}, calc={response.total_duration / (10**9)} for t={options['temperature']},p={options['top_p']},k={options['top_k']}")
  save(response, guideline_name, options)

ctx = 8000
predict = 3000


async def evaluate_model(model: str):
  async with sem:
    for top_p in reversed(np.arange(0.0,1.1,0.2)):
      for temperature in reversed(np.arange(0.0, 1.1, 0.2)):
        for top_k in [20,40,80]:
          options = {
            # Context size for new token generation
            'num_ctx': ctx,
            'num_predict': predict,
            # increased Temperature == more creativity
            'temperature': temperature,
            # higher top_k == more diverse answers (>=0)
            'top_k': top_k,
            # higher top_p == more diverse answers (0.0-1.0)
            'top_p': top_p
          }
          task_hyp = asyncio.create_task(evaluate(model, "hypertension", options))
          task_app = asyncio.create_task(evaluate(model, "acute_appendicitis", options))
          await task_app
          await task_hyp


async def main():

  print(f"started at {time.strftime('%X')}, ctx={ctx}, predict={predict}")
  tasks = []
  for model in models:
    tasks.append(asyncio.create_task(evaluate_model(model)))

  for task in tasks:
    await task
  print(f"finished at {time.strftime('%X')}")

# asyncio.run(main())

print(f"{prompt}{guidelines['acute_appendicitis']}")
