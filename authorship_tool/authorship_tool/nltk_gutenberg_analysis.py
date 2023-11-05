"""NLTKのGutenbergコーパスを用いた著者判定の分析を行うモジュール"""

import os
import re
from os import path
from typing import Final

import nltk
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from nltk.corpus import gutenberg

from authorship_tool.lgbm.model import LGBMResultModel, LGBMSourceModel
from authorship_tool.lgbm.trainer import learn_until_succeed
from authorship_tool.util import (
    ArrayDimensionReshaper,
    FeatureCalculator,
    FeatureDatasetGenerator,
    TabulateUtil,
)

load_dotenv()
DATASET_PATH: Final[str] = path.join(
    path.dirname(path.abspath(".env")),
    os.getenv("path.dataset") or "",
)

DESIRED_SCORE: Final[float] = 0.88
AUTHOR_A: Final[str] = "chesterton"
AUTHOR_B: Final[str] = "austen"

nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

print()

for idx, file_id in enumerate(gutenberg.fileids()):
    print(f"#{idx+1}\t{file_id}")

print()

authors = set()

for file_id in gutenberg.fileids():
    match = re.search(r"^(.+?)-", file_id)
    if match:
        authors.add(match.group(1))

book_data_dict = {}

for index, author in enumerate(authors):
    books = [
        gutenberg.paras(file_id) for file_id in gutenberg.fileids() if author in file_id
    ]
    para_num = len([paras for book in books for paras in book])
    book_data_dict[author] = para_num

paragraph_num_by_author_num: dict[str, int] = dict(
    sorted(book_data_dict.items(), key=lambda pd: pd[1], reverse=True)
)

for idx, item in enumerate(paragraph_num_by_author_num.items()):
    print(f"{idx + 1}:\t{item[0]} - {item[1]} paragraphs")

print()

books_a = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_A in file_id
]
paras_a = [paras for book in books_a for paras in book]

for para in paras_a[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_A}, {len(paras_a)} paragraphs\n\n")


books_b = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_B in file_id
]
paras_b = [paras for book in books_b for paras in book]

for para in paras_b[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_B}, {len(paras_b)} paragraphs\n\n")


all_tags: set[str] = set()

for paras in paras_a + paras_b:
    for para in paras:
        all_tags.update(FeatureCalculator.all_pos_frequency(para).keys())

print(sorted(all_tags))


dataset_generator = FeatureDatasetGenerator(all_tags)
data = []
correctness = []

for para_a in paras_a:
    x, y = dataset_generator.reshape_and_generate(para_a, all_tags, True)
    data.append(x)
    correctness.append(y)

for para_b in paras_b:
    x, y = dataset_generator.reshape_and_generate(para_b, all_tags, False)
    data.append(x)
    correctness.append(y)


df = pd.DataFrame(data, columns=dataset_generator.columns)
nd_correctness = np.array(correctness)

TabulateUtil.display(df.head(10))
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

result: LGBMResultModel = learn_until_succeed(
    LGBMSourceModel(DESIRED_SCORE, df, nd_correctness)
)

print(f"auc-roc score: {result.auc_roc_score}")
TabulateUtil.display(result.pred_crosstab())


result.dump()


exp = shap.TreeExplainer(result.model)
test_shap_val = exp.shap_values(result.test_data)[1]


pd.DataFrame(test_shap_val).to_csv(
    path.join(DATASET_PATH, "test_shap_val.csv"), index=False, header=False
)


shap.initjs()
shap.force_plot(
    exp.expected_value[1], test_shap_val[0], result.test_data.iloc[0], matplotlib=True
)


shap.decision_plot(exp.expected_value[1], test_shap_val[0], result.test_data.iloc[0])


shap.summary_plot(test_shap_val, result.test_data)


shap.summary_plot(test_shap_val, result.test_data, plot_type="bar")
