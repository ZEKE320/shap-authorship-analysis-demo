"""NLTKのGutenbergコーパスを用いた著者判定の分析を行うモジュール"""
# %%

import re
from typing import Final, TypeAlias
from authorship_tool.type_alias import Para

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
for idx, file_id in enumerate(gutenberg.fileids()):
    print(f"#{idx+1}\t{file_id}")

# %%
Author: TypeAlias = str
authors = set()

for file_id in gutenberg.fileids():
    match = re.search(r"^(.+?)-", file_id)
    if match:
        authors.add(match.group(1))

para_num_by_author: dict[Author, int] = {}

for index, author in enumerate(authors):
    books = [
        gutenberg.paras(file_id) for file_id in gutenberg.fileids() if author in file_id
    ]

    para_num = len([para for book in books for para in book])
    para_num_by_author[author] = para_num

sorted_para_num_by_author: dict[Author, int] = dict(
    sorted(para_num_by_author.items(), key=lambda pnba: pnba[1], reverse=True)
)

for idx, item in enumerate(sorted_para_num_by_author.items()):
    print(f"{idx + 1}:\t{item[0]} - {item[1]} paragraphs")


# %%
AUTHOR_A: Final[str] = "chesterton"
books_a: list[list[Para]] = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_A in file_id
]

paras_a: list[Para] = [para for book in books_a for para in book]

for para in paras_a[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_A}, {len(paras_a)} paragraphs\n\n")

# %%
AUTHOR_B: Final[str] = "austen"
books_b: list[list[Para]] = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if AUTHOR_B in file_id
]

paras_b: list[Para] = [para for book in books_b for para in book]

for para in paras_b[:10]:
    print(ArrayDimensionReshaper.para2str(para))
print(f"...\n\nAuthor: {AUTHOR_B}, {len(paras_b)} paragraphs\n\n")

# %%
all_paras: list[Para] = (
    paras_a + paras_b
    if TypeGuardUtil.are_paras(paras_a) and TypeGuardUtil.are_paras(paras_b)
    else []
)

if not all_paras:
    raise TypeError("paras_a or paras_b is not list[list[list[str]]] type.")

tag_tuple: tuple[str, ...] = (
    PosFeature(word_list=all_paras).tag_subcategories().pos_tuple
)

print(sorted(tag_tuple))

# %%
dataset_generator = FeatureDatasetGenerator(tags=tag_tuple)
data: list[tuple[float, ...]] = []
correctness: list[bool] = []

for para_a in paras_a:
    x, y = dataset_generator.generate_from_paragraph(para=para_a, correctness=True)
    data.append(x)
    correctness.append(y)

for para_b in paras_b:
    x, y = dataset_generator.generate_from_paragraph(para=para_b, correctness=False)
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
# shap.initjs()

# %%
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
# plt.show()

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
# plt.show()


# %%
shap.summary_plot(
    test_shap_val,
    result.test_data,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_PATH.joinpath("shap_summary_plot.svg"), bbox_inches="tight"
)
# plt.show()

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
# plt.show()
