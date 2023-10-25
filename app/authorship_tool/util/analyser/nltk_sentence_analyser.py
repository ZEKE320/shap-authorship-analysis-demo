# %%
import lightgbm as lgb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import shap
from sandbox.tools.feature_counter import DatasetGenerator, FrequencyCalculator
from sandbox.apps.trainer.lightgbm_trainer import LightGBMTrainer as learn
from nltk.corpus import inaugural
from sandbox.tools.table_generator import TableCreator as tc
from tabulate import tabulate

TARGET_PRESIDENT = "Biden"
NON_TARGET_PRESIDENT = "Obama"

plt.ion()
nltk.download("inaugural")

# %%
for idx, fileid in enumerate(inaugural.fileids()):
    print(f"#{idx+1} {fileid}")


# %%
presidents = set([fileid[5:-4] for fileid in inaugural.fileids()])
president_data_dict = {}

for index, president in enumerate(presidents):
    speeches = [inaugural.sents(file_id) for file_id in inaugural.fileids() if president in file_id]
    sent_num = len([sent for speech in speeches for sent in speech])
    president_data_dict[president] = sent_num

sorted_dict: dict[str, int] = dict(
    sorted(president_data_dict.items(), key=lambda pd: pd[1], reverse=True)
)

for idx, president_item in enumerate(sorted_dict.items()):
    print(f"{idx + 1}: {president_item[0]} - {president_item[1]} sentences")


# %%

nontarget_speeches = [
    inaugural.sents(file_id) for file_id in inaugural.fileids() if NON_TARGET_PRESIDENT in file_id
]
target_speeches = [
    inaugural.sents(file_id) for file_id in inaugural.fileids() if TARGET_PRESIDENT in file_id
]

nontarget_sents = [sent for speech in nontarget_speeches for sent in speech]
target_sents = [sent for speech in target_speeches for sent in speech]

for sent in nontarget_sents[:50]:
    print(" ".join(sent))
print(f"...\n\nSpeaker: President {NON_TARGET_PRESIDENT}, {len(nontarget_sents)} sentences")


# %%
for sent in target_sents[:50]:
    print(" ".join(sent))

print(f"...\n\nSpeaker: President {TARGET_PRESIDENT}, {len(target_sents)} sentences")


# %%

fc = FrequencyCalculator()
all_tags: set[str] = set()

for sent in nontarget_sents + target_sents:
    all_tags.update(fc.all_pos_frequency(sent).keys())

print(all_tags)


# %%

dg = DatasetGenerator(all_tags)
data = []
correctness = []

for b_sent in nontarget_sents:
    x, y = dg.generate_dataset_sent(b_sent, all_tags, False)
    data.append(x)
    correctness.append(y)

for b_sent in target_sents:
    x, y = dg.generate_dataset_sent(b_sent, all_tags, True)
    data.append(x)
    correctness.append(y)

df = pd.DataFrame(data, columns=dg.columns)
nd_correctness = np.array(correctness)

tc.display(df.head(10))


# %%
print(df.dtypes)


# %%
print(df.isna().sum())


# %%


val = 0
while val < 0.9:
    model, X_test, y_test, y_pred, val = learn(df, nd_correctness)

print(f"auc-rocスコア: {val}")


# %%
tc.display(pd.crosstab(y_test, y_pred))


# %%

exp = shap.TreeExplainer(model)
sv_test = exp.shap_values(X_test)[1]

# shap.initjs()
shap.force_plot(exp.expected_value[1], sv_test[0], X_test.iloc[0], matplotlib=True)


# %%
shap.decision_plot(exp.expected_value[1], sv_test[0], X_test.iloc[0])


# %%
shap.summary_plot(sv_test, X_test)


# %%
shap.summary_plot(sv_test, X_test, plot_type="bar")
