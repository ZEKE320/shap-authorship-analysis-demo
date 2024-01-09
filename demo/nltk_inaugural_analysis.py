# %%
from typing import TypeAlias

import nltk
import numpy as np
import pandas as pd
import shap
from IPython.display import display
from matplotlib import pyplot as plt
from nltk.corpus import inaugural

from authorship_tool.types import Tag, TwoDimStr
from authorship_tool.util import dim_reshaper, type_guard
from authorship_tool.util.feature.dataset_generator import (
    ParagraphFeatureDatasetGenerator,
)
from authorship_tool.util.feature.pos import PosFeature
from authorship_tool.util.ml import trainer as lgbm_trainer
from authorship_tool.util.ml.model import LGBMResult, LGBMSource
from authorship_tool.util.path_util import PathUtil

nltk.download("inaugural")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# %%
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# %%
PRESIDENT_A = "Obama"
PRESIDENT_B = "Bush"

# %%
for idx, file_id in enumerate(inaugural.fileids()):
    print(f"#{idx+1}\t{file_id}")


# %%
President: TypeAlias = str
NumOfParas: TypeAlias = int

presidents: set[President] = {file_id[5:-4] for file_id in inaugural.fileids()}

president_data_dict: dict[President, NumOfParas] = {}

for index, president in enumerate(presidents):
    speeches: list[list[TwoDimStr]] = [
        # inaugural.sents(file_id)
        inaugural.paras(fileids=file_id)
        for file_id in inaugural.fileids()
        if president in file_id
    ]  # type: ignore

    para_num: NumOfParas = len([para for paras in speeches for para in paras])
    president_data_dict[president] = para_num

sorted_para_size_by_president: dict[President, NumOfParas] = dict(
    sorted(president_data_dict.items(), key=lambda item: item[1], reverse=True)
)

for idx, item in enumerate(sorted_para_size_by_president.items()):
    print(f"{idx + 1}:\t{item[0]} - {item[1]} paragraphs")


# %%
speeches_a: list[list[TwoDimStr]] = [
    inaugural.paras(file_id)
    for file_id in inaugural.fileids()
    if PRESIDENT_A in file_id
]  # type: ignore

paras_a: list[TwoDimStr] = [para for paras in speeches_a for para in paras]
if len(paras_a) == 0 or not type_guard.are_paras(paras_a):
    raise ValueError("paras_a is empty or not list[Para]")

for para in paras_a[:20]:
    print(dim_reshaper.two_dim_to_str(para))

print(f"...\n\nSpeaker: President {PRESIDENT_A}, {len(paras_a)} paragraphs\n")

# %%
speeches_b: list[list[TwoDimStr]] = [
    inaugural.paras(file_id)
    for file_id in inaugural.fileids()
    if PRESIDENT_B in file_id
]  # type: ignore

paras_b: list[TwoDimStr] = [para for paras in speeches_b for para in paras]
for para in paras_b[:20]:
    print(dim_reshaper.two_dim_to_str(para))

print(f"...\n\nSpeaker: President {PRESIDENT_B}, {len(paras_b)} paragraphs\n")

# %%
print(f"total: {len(paras_a + paras_b)} samples (paragraphs)")

# %%
if not (type_guard.are_paras(paras_a) and type_guard.are_paras(paras_b)):
    raise ValueError("paras_a or sents_b is not list[Para]")
all_paras: list[TwoDimStr] = paras_a + paras_b

pos_list: list[Tag] = PosFeature(all_paras).tag_subcategories().pos_list

print(pos_list)

# %%
dataset_generator = ParagraphFeatureDatasetGenerator(pos_list)
data: list[tuple[float, ...]] = []
correctness: list[bool] = []

for para in paras_a:
    x, y = dataset_generator.generate_from_paragraph(para, True)
    data.append(x)
    correctness.append(y)

for para in paras_b:
    x, y = dataset_generator.generate_from_paragraph(para, False)
    data.append(x)
    correctness.append(y)

# %%
df = pd.DataFrame(data, columns=dataset_generator.columns)
nd_correctness = np.array(correctness)

display(df.head(10))


# %%
print(df.shape)


# %%
print(df.dtypes)


# %%
print(df.isna().sum())


# %%
result: LGBMResult = lgbm_trainer.train_once(LGBMSource(df, nd_correctness))


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
result.dump("inaugural")


# %%
test_data = result.splitted_dataset.test_data
explainer = result.shap_data.explainer
test_shap_val = result.shap_data.test_shap_val


# %%
shap.initjs()


# %%
PathUtil.SHAP_FIGURE_DIR.joinpath("inaugural").mkdir(exist_ok=True)

# %%
shap.force_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    test_data.iloc[0],
)
shap.force_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    test_data.iloc[0],
    matplotlib=True,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("inaugural", "shap_force_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()

# %%
shap.decision_plot(
    explainer.expected_value[1],  # type: ignore
    test_shap_val[0],
    test_data.iloc[0],
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("inaugural", "shap_decision_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()


# %%
shap.summary_plot(
    test_shap_val,
    test_data,
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("inaugural", "shap_summary_plot.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()

# %%
shap.summary_plot(
    test_shap_val,
    test_data,
    plot_type="bar",
    show=False,
)
plt.savefig(
    PathUtil.SHAP_FIGURE_DIR.joinpath("inaugural", "shap_summary_plot_bar.svg"),
    bbox_inches="tight",
)
plt.show()
plt.clf()
