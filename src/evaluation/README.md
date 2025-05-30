# Evaluation

This python module provides evaluation scripts that evaluate RAG system on:\
Given *.PDF files, Question Sets\
And produce metrics + AI-generated Experiment Summary

## Overview

Evaluation consists of 4 Stages (execute consequently):

1. Extraction: converting *.pdf to JSON-formatted paragraphs
2. Answers Generation: Generating 2 Sets of Answers: Golden, and Predicted (RAG Based)
3. Evaluation: LLM assigns labels for Golden and Predicted Answers, independently, (Golden Evals, Pred Evals) are used to compose metrics with confidence intervals
4. Analysis: LLM analyses logs of experiments and metrics to compose a conclusion and recommendations (for each file in test set separately)

Stages (2, 3, 4) use LLM calls:\
Whereas Stages (2, 4) may use cheaper models like Gemini-2.0-flash, Stage 3 is needy for a more sophisticated model e.g. Claude 3.7, gpt-4o


Currently, only locally stored embeddings are supported: Redis and Milvus, OpenAI's Vector Storage -- not supported.

## Known Limitations

Not all PDFs are extracted correctly. Some of them are extracted only partially, which causes some chaos in metrics.\
While you will see no error from such PDFs, you can manually inspect what really was extracted from your document:
```sh
cat chat-with-pdf/evaluations/XXXX/stage1_extraction/paragraphs_readable/XXX.txt
```

## Creating and Using a Dataset

To evaluate a set of files with your questions, a Dataset must be created

Dataset -- a directory with a custom name e.g. 'dataset-test-1'

Dataset must have a following flat structure

```text
dataset-test-1
    questions_str.json
    pdf_file_0.pdf
    ...
    pdf_file_N.pdf
```

To evaluate a created dataset:
1. Locate it in a `datasets` directory
2. Specify dataset name (dir name e.g. dataset-test-1) when running evaluation: `-dat`, `--dataset`

WHERE:
questions_str.json -- a JSON with following structure: `List[str]`:
```text
[
  "question_0",
  ...,
  "question_n"
]
```


After you started at least 1 evaluation that uses this dataset, `.metadata` file will be created\
this file preserves *.pdf names and checksums of questions, so dataset's content may no longer be changed

In case if you modified dataset and started one more evaluation on it, expect to see an error.\
You can either roll back changes made on dataset or circumvent deleting .metadata file.

Warning: circumvention is not recommended as with dataset changed there will be no longer possible compare experiments 

## Execution

Evaluation MUST be run INSIDE a container where actual Chat-with-pdf server runs.\
As it inserts data, such as paragraphs and embeddings into DB, which is accessed during Stage 2: Answers Generation,\
where model makes calls to LLM_ENDPOINT=llmtools, which calls Chat-with-pdf server. 

alter model names in config according to your preferences
```sh
vim chat-with-pdf/configs/eval_config.yaml
```

To execute evaluation, run the following commands:
```sh
docker exec -it docs-mcp bash
```
Then
```sh
python src/evaluation/main.py --dataset dataset-test-1 --eval_details pass
```
OR a single-line command
```sh
docker exec -it docs-mcp bash -c "python src/evaluation/main.py --dataset dataset-test-1 --eval_details pass"
```
If you want to use an example dataset
```sh
docker exec -it docs-mcp bash -c "python src/evaluation/main.py --dataset /app/assets/datasets/dataset-example --eval_details pass"
```

Note: replace `--dataset` with your own dataset name 

After execution is completed, you may find results in `chat-with-pdf/evaluations/XXXX`,\
where XXXX is the biggest number

## Approach Description

This evaluation approach is based on LLM-as-a-Judge technique \
Where: \
LLM compares Golden Answers and Predicted Answers assigning boolean labels to both answers' sets separately.

Please note: conversations and prompts below are simplified for README's readability. In a reality they are more complex

### Extraction

PDFs from the input (located in /assets/eval/*.pdf), on script's start are enqueued for the extraction.\
Each PDF gets transformed into a set of Paragraphs each of which includes text, coordinates, id, and other metadata.\
When all PDFs get extracted, requests for embeddings are made (text-embedding-3-small) is used.

Note: embedding requests are not included in the metering, as expenses for embeddings are rather negligible.

After paragraphs are extracted and embeddings are fetched using cloud-hosted model, data of them is inserted into DB.\
DB will vary upon user's config, currently there's an option of 2: redis or milvus, where milvus is currently recommended.

When everything is extracted, fetched and inserted, Stage 1: Extraction is complete. \
Deliverables of Stage: Extraction are located under `chat-with-pdf/evaluations/XXXX/stage1_extraction` for analysis when needed.

### Answers Generation

During Stage 2: Answers Generation paragraphs from previous stage are used to compose 2 sets of answers:\
Golden Answers and Predicted (RAG) Answers.

The actor of stage 2 is an LLM, that composes answers basing on these 2 strategies.

Answers are collected for an original set of questions of 19, whereas Questions Split set of 63 is only used for evaluation.

### Golden Answers

Golden Answers -- Ground Truth answers \
Where for each question of 19, the model receives the whole extracted document text as an input. \
Having an entire document, model should give a ground truth answer -- the one we also would aim to have for RAG approach.

#### Example Conversation
A single iteration is needed for this approach
```txt
System: You're a helpful assistant
User: Here's document you should use to answer my question: [document_text]
User: Who are the manufacturers? Is XXX a verified manufacturerer?
Assistant: Based on the context specified in the document above here's my answer...
```

### Predicted Answers (RAG Approach)

RAG Approach produces Set of "Predicted Answers" \
Where for each question of 19, the model does NOT receive the whole extracted document text as an input. \
However, model receives a set of Tools is can operate to receive context from that document -- a single tool in our case "search_in_doc" \
In other words, model gains an ability to chat with itself (limited for 5 iterations), where it can ask for materials using different queries.

#### Example Conversation:
Iteration 0
```txt
System: Dear model, you have document "XXX.pdf" at your disposal
User: Who are the manufacturers? Is XXX a verified manufacturerer?
Assistant: I would use tool search_in_doc to retrive needed context... [TOOL_CALLS]
```
Iteration 1
```txt
System: Dear model, you have document "XXX.pdf" at your disposal
User: Who are the manufacturers? Is XXX a verified manufacturerer?
Assistant: I would use tool search_in_doc to retrive needed context... [TOOL_CALLS]

Tool: [paragrah-XXX-0, paragrah-XXX-1, ... paragrah-XXX-N]
Assistant: Here's Answer based on a context you gave me...
```

When for each document in initial set of documents we 2 set of answers: Golden and Predicted, Phase 2: Answers Generation is complete.\
Deliverables of Phase 2: Answers Generation are located upon `chat-with-pdf/evaluations/XXXX/stage2_answers` for examination when needed.

There are 3 directories in `stage2_answers`:
1. Golden Answers: Golden Answers. grouped per file
2. RAG Answers: Predicted Answers, grouped per file
3. RAG Results: messages history used to produce a final answer -- Predicted Answers, grouped per file.


### Evaluation

After we received 2 sets of answers: Golden Answers and Predicted Answers, they are to be evaluated (independently)\
Which means, using LLM-as-a-Judge technique, a dedicated model call will evaluate how well answer is composed.\
For that, that model would produce 4 labels:

1. is_question_answered
2. requires_additional_information
3. is_speculative
4. is_confident

Example Conversation (LLM-as-a-Judge call):
```txt
System: You're a helpful assistant
User: Question: [question], Answer: [answer]. Please, assign those 4 labels to each answer.
Assistant: json { is_question_answered: true, requires_additional_information: false...}
```

Important! For the evaluation we are not using Question List (19 pcs), but a Split Question List (63 pcs)\
Which means, for each file we receive:
```txt
is_question_answered: [false, true, true, ..60pcs]
requires_additional_information: [true, true, true, ..60pcs]
is_speculative: [false, false, true, ..60pcs]
is_confident: [false, true, false, ..60pcs]
```

On top of those 4 labels, we compose a target label -- comprehensive_answer\
Which is: 
```
is_question_answered AND !requires_additional_information AND !is_speculative AND is_confident
```
Means, for a single file we have:
```txt
comprehensive_answer: [false, false, false, true, ..59pcs]
```

As we had evaluated both Golden Answers and Predicted Answers, we have 2 sets of comprehensive_answer labels:
```txt
GOLDEN:
comprehensive_answer: [false, true, true, true, ..59pcs]

PRED:
comprehensive_answer: [false, false, false, true, ..59pcs]
```


### Metrics

Golden answers are also important, because some questions can't be answered even if we have all document as a context.\
If pred gives a positive answer, and golden -- negative, pred is likely producing a False Positive Result (FP)

For that, we have a bunch of metrics:
1. Accuracy: Percentage of correct predictions overall 
2. Precision: Percentage of true positives among all positive predictions 
3. Recall: Percentage of true positives captured from all actual positives 
4. F1: Harmonic mean of precision and recall 
5. Cohen's Kappa: Agreement between predictions and gold standard beyond random chance

#### Metrics' interpretation

```txt
Accuracy (0-1): Higher = more correct predictions overall

Precision (0-1): Higher = fewer false positives, more reliable "yes" answers

Recall (0-1): Higher = fewer false negatives, better at finding all correct answers

F1 (0-1): Higher = better balance between precision and recall

Cohen's Kappa (-1 to 1):
    <0: Worse than random
    0: Random agreement
    0-0.2: Slight agreement
    0.2-0.4: Fair agreement
    0.4-0.6: Moderate agreement
    0.6-0.8: Substantial agreement
    0.8-1.0: Almost perfect agreement
```

In addition to metrics, for each one of those, we have Confidence Intervals: \
Confidence intervals (CI) -- represent an interval in which 95% of the values of the metric are located. \
E.g. (0.5, 0.7) CI (Confidence Interval) means, 95% of values of metric A are located withing values (0.5 and 0.7) \

#### Interpretation
```txt
Narrow intervals (e.g., 0.75-0.78): High certainty about the true value
Wide intervals (e.g., 0.60-0.90): High uncertainty about the true value

Large and more homogenous datasets produce more narrow CIs.

e.g. 
for metric [0, 1], CI width > 0.3: Results are highly uncertain  
for metric [0, 1], CI wdth > ~0.4: Results are too uncertain, we may disregard thin metric.    
```

#### Comparing systems (experiments)
```txt
Having 2 pair of systems, CIs:

    System A: [0.68---0.72]
                        [0.71---0.75] System B
                        
    System C: [0.65-----------0.75]
                  [0.67-----------0.77] System D
                  
It's conclusive that System B performs better then System A, due to low overlapping in their CIs
However, in a case of System C and System D, it's inconclusive to say which one performs better
due to high overlapping of their CIs. 
The difference may likely be caused by random variation in test samples rather then true difference in a peroformance.
```

When set of Golden Answers, and Predicted (RAG) Answers is labeled using LLM model, and evaluated using metrics, and CIs are calculated, 
Phase 3: Evaluation is complete. \
Deliverables of this phase are located upon: `chat-with-pdf/evaluations/XXXX/stage3_evaluation`

There are 2 subdirectories:
1. LLM Judge -- logs for a step when labels are applied to Golden and Predicted Answers. Those logs are used in Stage 4: Analysis
2. Metrics -- all sets of metrics collected in metrics step. Also is used in Stage 4 for further analysis.

The most important files in metrics (basically results of evaluation) are:
1. comprehensive_answer.json -- metrics per file and for all files overall, along with CI
2. passed_overall.json -- which question passed, and which did not per each file

### Analysis 

Final Stage -- Analysis is performed to actually analyse and interpret results from Stage 3: Evaluation.\
We not only get report based on metrics and their CIs, but also passing logs from that step.\
Passing logs is important, because the model that composes report also sees actual questions and answers on those questions.\
So model notices potential flaws in current RAG Approach, and is able to identify specific cases that fails systematically.\
Along with said report model that performs analysis also produces practical recommendations how to improve current RAG approach.

In my case, model notices, that when Golden Answer say there's no information, model says it confidently,\
but when Predicted Answer also has no information, it is not confident, because it didn't see the whole document,\
so, funny enough, that case does not pass as correct, and we count it as a flaw.

When model noticed such behaviour in evaluator, in noted, that evaluator's system prompt should be adjusted, \
or we should consider fine-tuning of Answer-generating model, which is not a good idea for our case.

Basically, analysis request is as straightforward as evaluation:
```txt
System: you are a helpful AI model
User: [rag_evals], [golden_evals], [metrics]. Having all that data, perform analysis and report
Assistant: Here's my report basing on information provided...
```

Such reports are generated for each files in particular, because fitting all files' logs and metrics into a single context window \
is barely possible and not a good practice overall.

When for each file we receive a report, Stage 4: Analysis is said to be complete.\
It's worth to mention, that along all steps we've been collected model usage,\ 
which is presented at `chat-with-pdf/evaluations/XXXX/metering.json`:
```json
{
  "stage1": {},
  "stage2": {
    "gemini-2.0-flash": {
      "requests_cnt": 57,
      "messages_sent_cnt": 180,
      "tokens_in": 175923,
      "tokens_out": 4544
    }
  },
  "stage3": {
    "gpt-4o": {
      "requests_cnt": 38,
      "messages_sent_cnt": 38,
      "tokens_in": 20434,
      "tokens_out": 10432
    }
  },
  "stage4": {
    "gemini-2.0-flash": {
      "requests_cnt": 1,
      "messages_sent_cnt": 1,
      "tokens_in": 22206,
      "tokens_out": 2027
    }
  }
}
```
As said in the beginning, only Stages (2, 3, 4) are using LLMs, which can be seen in metering.json.\
Also, as already noted we disregard usage from embedding models, as it's rather negligible.

Tokens count is received when completion request is complete from Cloud Provider.\
We do not provide total price in dollars as we don't store pricing for each model in this repository.\
However, a simple script with that data can be used to sum up tokens and calculate a total amount.

Stage 4: Analysis produces deliverables located at `chat-with-pdf/evaluations/XXXX/stage4_analysis`\
There are 2 subdirectories:
1. Analysis Results -- *.txt reports for each document
2. User Messages -- messages and context that model received as an input to generate report, for user examination if needed.

Big descriptive analysis along with metrics are transforming into brief, human-readable analyse reports in .md format,
located at `chat-with-pdf/evaluations/XXXX/stage4_analysis/analysis_results_md`

Those human-readable reports from all files in a dataset will then be consumed one more time by a model to generate a final report of the experiment
located at `chat-with-pdf/evaluations/XXXX/analysis_overall.md`
