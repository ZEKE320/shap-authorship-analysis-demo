"""NLTKのGutenbergコーパスを用いた著者判定の分析を行うモジュール"""
# %%

import re
from typing import Final

import matplotlib.pyplot as plt
import nltk
import numpy as np
import shap
from nltk.corpus import gutenberg
from pandas import DataFrame

from authorship_tool.util import (
    LGBMResultModel,
    LGBMSourceModel,
    ArrayDimensionReshaper,
    FeatureDatasetGenerator,
    TabulateUtil,
    PosFeature,
    TypeGuardUtil,
    PathUtil,
    LGBMTrainerUtil,
)

nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# %%

DESIRED_SCORE: Final[float] = 0.88
AUTHOR_A: Final[str] = "chesterton"
AUTHOR_B: Final[str] = "austen"

for idx, file_id in enumerate(gutenberg.fileids()):
    print(f"#{idx+1}\t{file_id}")

# %%
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

    para_num = len([para for book in books for para in book])
    book_data_dict[author] = para_num

paragraph_num_by_author_num: dict[str, int] = dict(
    sorted(book_data_dict.items(), key=lambda pd: pd[1], reverse=True)
)

for idx, item in enumerate(paragraph_num_by_author_num.items()):
    print(f"{idx + 1}:\t{item[0]} - {item[1]} paragraphs")


# %%

books_a: list[list[list[list[str]]]] = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_A in file_id
]

paras_a: list[list[list[str]]] = [para for book in books_a for para in book]

for para in paras_a[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_A}, {len(paras_a)} paragraphs\n\n")

# %%
books_b: list[list[list[list[str]]]] = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_B in file_id
]

paras_b: list[list[list[str]]] = [para for book in books_b for para in book]

for para in paras_b[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_B}, {len(paras_b)} paragraphs\n\n")

# %%
all_paras: list[list[list[str]]] = (
    paras_a + paras_b
    if TypeGuardUtil.are_paras(paras_a) and TypeGuardUtil.are_paras(paras_b)
    else []
)

if not all_paras:
    raise TypeError("paras_a or paras_b is not list[list[list[str]]] type.")

pos_set: set[str] = set(
    tag for tag in PosFeature(all_paras).classify_subcategories().pos_set
)

all_pos: list[str] = sorted(pos_set)
print(all_pos)

# %%
dataset_generator = FeatureDatasetGenerator(all_pos)
data = []
correctness = []

for para_a in paras_a:
    x, y = dataset_generator.reshape_and_generate(para_a, all_pos, True)
    data.append(x)
    correctness.append(y)

for para_b in paras_b:
    x, y = dataset_generator.reshape_and_generate(para_b, all_pos, False)
    data.append(x)
    correctness.append(y)


df = DataFrame(data, columns=dataset_generator.columns)
nd_correctness = np.array(correctness)

TabulateUtil.display(df.head(10))

# %%
print(df.shape)
# %%
print(df.dtypes)
# %%
print(df.isna().sum())

# %%
result: LGBMResultModel = LGBMTrainerUtil.learn_until_succeed(
    LGBMSourceModel(DESIRED_SCORE, df, nd_correctness)
)


# %%
print(f"auc-roc score: {result.auc_roc_score}")


# %%
TabulateUtil.display(result.pred_crosstab())

# %%
result.dump()

explainer = shap.explainers.TreeExplainer(result.model)
test_shap_val = explainer.shap_values(result.test_data)[1]

DataFrame(test_shap_val).to_csv(
    PathUtil.DATASET_PATH.joinpath("test_shap_val.csv"), index=False, header=False
)

# %%
shap.initjs()

# %%
shap.force_plot(
    explainer.expected_value[1],
    test_shap_val[0],
    result.test_data.iloc[0],
)
shap.force_plot(
    explainer.expected_value[1],
    test_shap_val[0],
    result.test_data.iloc[0],
    matplotlib=True,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_PATH.joinpath("shap_force_plot.svg"), bbox_inches="tight"
)

# %%
shap.decision_plot(
    explainer.expected_value[1],
    test_shap_val[0],
    result.test_data.iloc[0],
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_PATH.joinpath("shap_decision_plot.svg"), bbox_inches="tight"
)
plt.show()


# %%
shap.summary_plot(
    test_shap_val,
    result.test_data,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_PATH.joinpath("shap_summary_plot.svg"), bbox_inches="tight"
)
plt.show()

# %%
shap.summary_plot(
    test_shap_val,
    result.test_data,
    plot_type="bar",
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_PATH.joinpath("shap_summary_plot_bar.svg"), bbox_inches="tight"
)
plt.show()

# %%
