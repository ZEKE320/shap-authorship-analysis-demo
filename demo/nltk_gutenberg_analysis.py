# %%
import re
from typing import Final, TypeAlias

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from nltk.corpus import gutenberg
from pandas import DataFrame
from shap import Explainer

from authorship_tool.types import Para2dStr, Tag
from authorship_tool.util import dim_reshaper, type_guard
from authorship_tool.util.feature.dataset_generator import (
    ParagraphFeatureDatasetGenerator,
)
from authorship_tool.util.feature.pos import PosFeature
from authorship_tool.util.ml import trainer as lgbm_trainer
from authorship_tool.util.ml.model import LGBMResult, LGBMSource
from authorship_tool.util.path_util import PathUtil

# 必要に応じてダウンロード
nltk.download("gutenberg")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# %%
AUTHOR_A: Final[str] = "chesterton"
AUTHOR_B: Final[str] = "bryant"

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

for index, author in enumerate(authors):
    books_of_author: list[list[Para2dStr]] = [
        gutenberg.paras(fileids=file_id)
        for file_id in gutenberg.fileids()
        if author in file_id
    ]  # type: ignore

    para_num: NumOfParas = len([para for paras in books_of_author for para in paras])
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

for para in paras_a[:20]:
    print(dim_reshaper.two_dim_to_str(para))

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

for para in paras_b[:20]:
    print(dim_reshaper.two_dim_to_str(para))

print(f"...\n\nAuthor: {AUTHOR_B}, {len(paras_b)} paragraphs\n")

# %%
print(f"total: {len(paras_a + paras_b)} paragraphs (samples)")

# %%
if not (type_guard.are_paras(paras_a) and type_guard.are_paras(paras_b)):
    raise TypeError("paras_a or paras_b is not list[Para]")
all_paras: list[Para2dStr] = paras_a + paras_b

pos_list: list[Tag] = PosFeature(all_paras).tag_subcategories().pos_list

print(pos_list)

# %%
dataset_generator = ParagraphFeatureDatasetGenerator(tags=pos_list)


# %%
para_and_correctness_list: list[tuple[Para2dStr, bool]] = list(
    [(para, True) for para in paras_a] + [(para, False) for para in paras_b]
)

# %%
dataset, categories = zip(
    *[
        dataset_generator.generate_from_paragraph(para, is_correct)
        for para, is_correct in para_and_correctness_list
    ]
)

# %%
df = DataFrame(dataset, columns=dataset_generator.columns)
nd_category = np.array(categories, dtype=bool)

display(df.head(10))

# %%
print(df.shape)


# %%
print(df.dtypes)


# %%
print(df.isna().sum())


# %%
result: LGBMResult = lgbm_trainer.train_once(LGBMSource(df, nd_category))


# %%
score = result.score

# %%
print(f"auc-roc score: {score.auc_roc_score}")


# %%
print(f"f1 score: {score.f1_score}")

# %%
print(f"accuracy score: {score.accuracy_score}")

# %%
display(result.pred_crosstab())


# %%
result.dump("gutenberg")


# %%
test_data: DataFrame = result.splitted_dataset.test_data
explainer: Explainer = result.shap_data.explainer
shap_expected_val: float = result.shap_data.shap_positive_expected_val
shap_vals = result.shap_data.shap_positive_vals

FIRST_DATA_INDEX: Final[int] = 0


# %%
shap.initjs()


# %%
PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg").mkdir(exist_ok=True)

# %%
shap.force_plot(
    shap_expected_val,
    shap_vals[FIRST_DATA_INDEX],
    test_data.iloc[FIRST_DATA_INDEX],
)
shap.force_plot(
    shap_expected_val,
    shap_vals[FIRST_DATA_INDEX],
    test_data.iloc[FIRST_DATA_INDEX],
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
    shap_expected_val,  # type: ignore
    shap_vals[FIRST_DATA_INDEX],
    test_data.iloc[FIRST_DATA_INDEX],
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
    shap_vals,
    test_data,
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
    shap_vals,
    test_data,
    plot_type="bar",
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("gutenberg", "shap_summary_plot_bar.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()
