# %%
import re
from os import makedirs
from typing import Final, TypeAlias

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from nltk.corpus import gutenberg
from pandas import DataFrame

from authorship_tool.types import Para2dStr, Tag
from authorship_tool.util import dim_reshaper, type_guard
from authorship_tool.util.feature.generator import FeatureDatasetGenerator
from authorship_tool.util.feature.pos import PosFeature
from authorship_tool.util.lgbm import trainer as lgbm_trainer
from authorship_tool.util.lgbm.model import LGBMResultModel, LGBMSourceModel
from authorship_tool.util.path_util import PathUtil

# 必要に応じてダウンロード
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# %%
AUTHOR_A: Final[str] = "chesterton"
AUTHOR_B: Final[str] = "austen"
DESIRED_ROC_SCORE: Final[float] = 0.88


# %%
for idx, file_id in enumerate(iterable=gutenberg.fileids()):
    print(f"#{idx+1}\t{file_id}")


# %%
Author: TypeAlias = str
NumOfParas: TypeAlias = int

authors: set[Author] = {
    match.group(1)
    for file_id in gutenberg.fileids()
    if (match := re.search(r"^(.+?)-", file_id)) is not None
}

para_size_by_author: dict[Author, NumOfParas] = {}

for index, author in enumerate(iterable=authors):
    books: list[list[Para2dStr]] = [
        gutenberg.paras(fileids=file_id)
        for file_id in gutenberg.fileids()
        if author in file_id
    ]  # type: ignore

    para_num: NumOfParas = len([para for paras in books for para in paras])
    para_size_by_author[author] = para_num

sorted_para_size_by_author: dict[Author, NumOfParas] = dict(
    sorted(para_size_by_author.items(), key=lambda item: item[1], reverse=True)
)

for idx, item in enumerate(sorted_para_size_by_author.items()):
    print(f"{idx + 1}:\t{item[0]} - {item[1]} paragraphs")


# %%
books_a: list[list[Para2dStr]] = [
    gutenberg.paras(fileids=file_id)
    for file_id in gutenberg.fileids()
    if AUTHOR_A in file_id
]  # type: ignore

paras_a: list[Para2dStr] = [para for paras in books_a for para in paras]
if len(paras_a) == 0 or not type_guard.are_paras(paras_a):
    raise ValueError("paras_a is empty or not list[Para]")

for para in paras_a[:10]:
    print(dim_reshaper.para_to_str(para))

print(f"...\n\nAuthor: {AUTHOR_A}, {len(paras_a)} paragraphs\n")


# %%

books_b: list[list[Para2dStr]] = [
    gutenberg.paras(fileids=file_id)
    for file_id in gutenberg.fileids()
    if AUTHOR_B in file_id
]  # type: ignore

paras_b: list[Para2dStr] = [para for paras in books_b for para in paras]
if len(paras_b) == 0 or not type_guard.are_paras(paras_b):
    raise ValueError("paras_a is empty or not list[list[str]]")

for para in paras_b[:10]:
    print(dim_reshaper.para_to_str(para))

print(f"...\n\nAuthor: {AUTHOR_B}, {len(paras_b)} paragraphs\n")


# %%
if not (type_guard.are_paras(paras_a) and type_guard.are_paras(paras_b)):
    raise TypeError("paras_a or paras_b is not list[Para]")
all_paras: list[Para2dStr] = paras_a + paras_b

pos_list: list[Tag] = PosFeature(all_paras).tag_subcategories().pos_list

print(pos_list)


# %%
dataset_generator = FeatureDatasetGenerator(tags=pos_list)
data: list[tuple[float, ...]] = []
correctness: list[bool] = []

for para_a in paras_a:
    (x, y) = dataset_generator.generate_from_paragraph(para_a, True)
    data.append(x)
    correctness.append(y)

for para_b in paras_b:
    (x, y) = dataset_generator.generate_from_paragraph(para_b, False)
    data.append(x)
    correctness.append(y)


# %%
df = DataFrame(data, columns=dataset_generator.columns)
nd_correctness = np.array(correctness)

pd.set_option("display.max_columns", 1000)
display(df.head(10))
pd.reset_option("display.max_columns")


# %%
print(df.shape)


# %%
print(df.dtypes)


# %%
print(df.isna().sum())


# %%
result: LGBMResultModel = lgbm_trainer.learn_until_succeed(
    LGBMSourceModel(DESIRED_ROC_SCORE, df, nd_correctness)
)


# %%
print(f"auc-roc score: {result.auc_roc_score}")


# %%
display(result.pred_crosstab())


# %%
result.dump("gutenberg")


# %%
explainer = shap.TreeExplainer(result.model)
test_shap_val = explainer.shap_values(result.test_data)[1]

DataFrame(test_shap_val).to_csv(
    PathUtil.DATASET_DIR.joinpath("test_shap_val.csv"), index=False, header=False
)


# %%
shap.initjs()


# %%
makedirs(PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg"), exist_ok=True)

# %%
shap.force_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    result.test_data.iloc[0],
)
shap.force_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    result.test_data.iloc[0],
    matplotlib=True,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg", "shap_force_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()


# %%
shap.decision_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    result.test_data.iloc[0],
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg", "shap_decision_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()


# %%
shap.summary_plot(
    test_shap_val,
    result.test_data,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg", "shap_summary_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()


# %%
shap.summary_plot(
    test_shap_val,
    result.test_data,
    plot_type="bar",
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg", "shap_summary_plot_bar.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()
