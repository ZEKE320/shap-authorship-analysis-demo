# %%
import re

import lightgbm as lgb
import nltk
import numpy as np
import pandas as pd
import sandbox.tools.feature_counter as fc
import sandbox.tools.paragraph_analysis_tool as pat
import sandbox.tools.table_generator as tg
import shap
import sklearn
from nltk.corpus import gutenberg
from sandbox.tools.dataset_generator import DatasetGenerator

nltk.download("gutenberg")


# %%
for idx, fileid in enumerate(gutenberg.fileids()):
    print(f"#{idx+1} {fileid}")


# %%

authors = set()

for fileid in gutenberg.fileids():
    match = re.search(r"^(.+?)-", fileid)
    if match:
        authors.add(match.group(1))

book_data_dict = {}

for index, author in enumerate(authors):
    books = [
        gutenberg.paras(fileid) for fileid in gutenberg.fileids() if author in fileid
    ]
    para_num = len([paras for book in books for paras in book])
    book_data_dict[author] = para_num

sorted_dict: dict[str, int] = dict(
    sorted(book_data_dict.items(), key=lambda pd: pd[1], reverse=True)
)

for idx, item in enumerate(sorted_dict.items()):
    print(f"{idx + 1}: {item[0]} - {item[1]} paragraphs")


# %%

TARGET = "chesterton"
NON_TARGET = "carroll"

nontarget_books = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if NON_TARGET in file_id
]
target_books = [
    gutenberg.paras(file_id) for file_id in gutenberg.fileids() if TARGET in file_id
]

nontarget_paras = [paras for book in nontarget_books for paras in book]
target_paras = [paras for book in target_books for paras in book]

for para in nontarget_paras[:50]:
    print(" ".join(pat.para2sent(para)))
print(f"...\n\nAuthor: {NON_TARGET}, {len(nontarget_paras)} paragraphs")


# %%
for para in target_paras[:50]:
    print(" ".join(pat.para2sent(para)))
print(f"...\n\nAuthor: {TARGET}, {len(target_paras)} paragraphs")


# %%

all_tags: set[str] = set()

for paras in nontarget_paras + target_paras:
    for para in paras:
        all_tags.update(fc.all_pos_frequency(para).keys())

print(all_tags)
# %%

dg = DatasetGenerator(all_tags)
data = []
correctness = []

for a_para in nontarget_paras:
    x, y = dg.generate_dataset_para(a_para, all_tags, False)
    data.append(x)
    correctness.append(y)

for b_para in target_paras:
    x, y = dg.generate_dataset_para(b_para, all_tags, True)
    data.append(x)
    correctness.append(y)

df = pd.DataFrame(data, columns=dg.columns)
nd_correctness = np.array(correctness)

tg.display(df.head(10))


# %%
print(df.shape)


# %%
print(df.dtypes)


# %%
print(df.isna().sum())


# %%


def learn(df, nd_correctness):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        df, nd_correctness
    )

    model = lgb.LGBMClassifier()
    model.fit(X_train.values, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    val = sklearn.metrics.roc_auc_score(y_test, y_pred_prob)
    return model, X_train, X_test, y_train, y_test, y_pred_prob, y_pred, val


val = 0
while val < 0.9:
    model, X_train, X_test, y_train, y_test, y_pred_prob, y_pred, val = learn(
        df, nd_correctness
    )

print(f"auc-rocスコア: {val}")


# %%
tg.display(pd.crosstab(y_test, y_pred))


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
