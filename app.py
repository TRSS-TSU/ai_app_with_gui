
from transformers import AutoModelForQuestionAnswering, pipeline, AutoTokenizer
from utils.dataloader import loader
from utils.metrics import compute_exact
import numpy as np
from utils.metrics import compute_exact
import pandas as pd
import os
import warnings
from flask import Flask, render_template, request, flash
warnings.filterwarnings("ignore")


app = Flask(__name__)
app.secret_key = "mamabear333"


@app.route("/ai")
def index():
    flash("")
    return render_template("index.html")


def read_file(path):
    with open(path, 'r') as file:
        context = file.read().replace('\n', '')
    return context


def save_results(question, result, ans1, ans2, ans3):
    csv_file = "result/results.csv"
    if os.path.exists(csv_file):
        headers = False
    else:
        headers = True

    df = pd.DataFrame(list(zip([question], [result], [ans1], [ans2], [ans3])),
                      columns=["question", "prediction", "model1_result", "model2_result", "model3_result"
                               ])

    df.to_csv(csv_file, mode='a', encoding='utf-8',
              index=False, header=headers)


def preprocess(text):

    if "?" not in text:
        text = text+"?"

    return text


# Constants
NAMES = ["models/bert-large-uncased-whole-word-masking-squad2",
         "models/roberta-base-squad2",
         "models/minilm-uncased-squad2"
         ]
CONTEXT = read_file("data/context/NTLM_context.txt")


# model and tokenizer intialization
model1 = AutoModelForQuestionAnswering.from_pretrained(NAMES[0])
model2 = AutoModelForQuestionAnswering.from_pretrained(NAMES[1])
model3 = AutoModelForQuestionAnswering.from_pretrained(NAMES[2])

tokenizer1 = AutoTokenizer.from_pretrained(
    NAMES[0], model_max_length=int(1e30))
tokenizer2 = AutoTokenizer.from_pretrained(
    NAMES[1], model_max_length=int(1e30))
tokenizer3 = AutoTokenizer.from_pretrained(
    NAMES[2], model_max_length=int(1e30))


@app.route("/greet", methods=["POST", "GET"])
def greet():

    # start of loop
    exit_conditions = ("exit")
    question = str(request.form['name_input'])
    while True:
        print("To quit type exit")

        if question in exit_conditions:
            break
        else:
            question = preprocess(question)
            tokenizer1.encode(question, truncation=True,
                              padding=True, verbose=False)
            tokenizer2.encode(question, truncation=True,
                              padding=True, verbose=False)
            tokenizer3.encode(question, truncation=True,
                              padding=True, verbose=False)

            # init pipeline
            m1 = pipeline("question-answering",
                          model=model1, tokenizer=tokenizer1)
            m2 = pipeline("question-answering",
                          model=model2, tokenizer=tokenizer2)
            m3 = pipeline("question-answering",
                          model=model3, tokenizer=tokenizer3)

            # generating prediction
            answer1 = m1({'question': question, 'context': CONTEXT})
            answer2 = m2({'question': question, 'context': CONTEXT})
            answer3 = m3({'question': question, 'context': CONTEXT})

            ans = [answer1["answer"], answer2["answer"], answer3["answer"]]

            a1 = compute_exact(ans[0], ans[1]) + compute_exact(ans[0], ans[2])
            a2 = compute_exact(ans[1], ans[0]) + compute_exact(ans[1], ans[2])
            a3 = compute_exact(ans[2], ans[0]) + compute_exact(ans[2], ans[1])
            calc = [a1, a2, a3]

            if np.sum(calc) == 0:
                result = "no answer"
            else:
                result = ans[calc.index(max(calc))]

            save_results(question, result,
                         answer1["answer"], answer2["answer"], answer3["answer"])

            flash(result)
            return render_template("index.html")


# ------------------------------------------------------------------------------------------------------------
