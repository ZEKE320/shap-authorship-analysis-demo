"""
特徴量計算モジュールテスト
Feature calculator module test
"""

from authorship_tool.types_ import Para2dStr
from authorship_tool.util.feature.calculator import UnivKansasFeatures
from authorship_tool.util.tokenizer import tokenize_para

text_1: str = (
    "This is a famous pen, Mr. Smith."
    + " It was designed by the famous artist Amy Mahogany,"
    + " and it looks very classy, don't you think so?"
)
text_2: str = (
    "Oh no! Someone has stolen my pen... (I cannot believe this is happening!)"
)
text_3: str = (
    "I'm glad to hear that you found it! This pen is well-designed. I really like it."
)
para_1: Para2dStr = tokenize_para(text_1)
para_2: Para2dStr = tokenize_para(text_2)
para_3: Para2dStr = tokenize_para(text_3)


@staticmethod
def test_v1_sentences_per_paragraph() -> None:
    """段落ごとの文の数を計算するテスト"""
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_1) == 2
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_2) == 2
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_3) == 3


@staticmethod
def test_v2_words_per_paragraph() -> None:
    """段落ごとの単語数を計算するテスト"""
    assert UnivKansasFeatures.v2_words_per_paragraph(para_1) == 31
    assert UnivKansasFeatures.v2_words_per_paragraph(para_2) == 19
    assert UnivKansasFeatures.v2_words_per_paragraph(para_3) == 20


@staticmethod
def test__char_present() -> None:
    """指定された文字が含まれているかどうかを判定するテスト"""
    assert UnivKansasFeatures._char_present(  # pylint: disable=protected-access
        para_1, ","
    )
    assert UnivKansasFeatures._char_present(  # pylint: disable=protected-access
        para_2, "."
    )
    assert not UnivKansasFeatures._char_present(  # pylint: disable=protected-access
        para_3, "?"
    )


@staticmethod
def test_v3_close_parenthesis_present() -> None:
    """閉じ括弧が含まれているかどうかを判定するテスト"""
    assert not UnivKansasFeatures.v3_close_parenthesis_present(para_1)
    assert UnivKansasFeatures.v3_close_parenthesis_present(para_2)
    assert not UnivKansasFeatures.v3_close_parenthesis_present(para_3)


@staticmethod
def test_v4_dash_present() -> None:
    """ダッシュが含まれているかどうかを判定するテスト"""
    assert not UnivKansasFeatures.v4_dash_present(  # pylint: disable=protected-access
        para_1
    )
    assert not UnivKansasFeatures.v4_dash_present(  # pylint: disable=protected-access
        para_2
    )
    assert UnivKansasFeatures.v4_dash_present(  # pylint: disable=protected-access
        para_3
    )


@staticmethod
def test__contains_specific_word() -> None:
    """指定された単語が含まれているかどうかを判定するテスト"""
    assert not UnivKansasFeatures._contains_word_including_derived_forms(  # pylint: disable=protected-access
        para_1, "thin"
    )
    assert not UnivKansasFeatures._contains_word_including_derived_forms(  # pylint: disable=protected-access
        para_2, "an"
    )
    assert UnivKansasFeatures._contains_word_including_derived_forms(  # pylint: disable=protected-access
        para_3, "likes"
    )


@staticmethod
def test_v5_semi_colon_or_colon_present() -> None:
    """セミコロンまたはコロンが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v6_question_mark_present() -> None:
    """疑問符が含まれているかどうかを判定するテスト"""
    assert UnivKansasFeatures.v6_question_mark_present(para_1)
    assert not UnivKansasFeatures.v6_question_mark_present(para_2)
    assert not UnivKansasFeatures.v6_question_mark_present(para_3)


@staticmethod
def test_v7_apostrophe_present() -> None:
    """アポストロフィが含まれているかどうかを判定するテスト"""
    assert UnivKansasFeatures.v7_apostrophe_present(para_1)
    assert not UnivKansasFeatures.v7_apostrophe_present(para_2)
    assert UnivKansasFeatures.v7_apostrophe_present(para_3)


@staticmethod
def test_v8_standard_deviation_of_sentence_length() -> None:
    """文の長さの標準偏差を計算するテスト"""
    # TODO Implement here


@staticmethod
def test_v9_length_difference_for_consecutive_sentences() -> None:
    """連続する文の長さの差を計算するテスト"""
    # TODO Implement here


@staticmethod
def test_v10_sentence_with_lt_11_words() -> None:
    """11語未満の文が含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v11_sentence_with_gt_34_words() -> None:
    """34語を超える文が含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v12_contains_although() -> None:
    """althoughが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v13_contains_however() -> None:
    """howeverが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v14_contains_but() -> None:
    """butが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v15_contains_because() -> None:
    """becauseが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v16_contains_this() -> None:
    """thisが含まれているかどうかを判定するテスト"""
    # TODO Implement here


@staticmethod
def test_v17_contains_others_or_researchers() -> None:
    """othersまたはresearchersが含まれているかどうかを判定するテスト"""

    para_v17_1: Para2dStr = tokenize_para("The researchers conducted the experiment.")
    para_v17_2: Para2dStr = tokenize_para(
        "The other person did not agree with the result."
    )
    para_v17_3: Para2dStr = tokenize_para("He conducted a research.")
    para_v17_4: Para2dStr = tokenize_para("SHe discriminated against others.")
    assert UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_1)
    assert not UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_2)
    assert not UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_3)
    assert UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_4)


@staticmethod
def test_v19_contains_2_times_more_capitals_than_period() -> None:
    """ピリオドよりも大文字が2倍多く含まれているかどうかを判定するテスト"""

    assert UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_1)
    assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_2)
    assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_3)


@staticmethod
def test_v20_contains_et() -> None:
    """etが含まれているかどうかを判定するテスト"""

    assert not UnivKansasFeatures.v20_contains_et(para_1)
    assert not UnivKansasFeatures.v20_contains_et(para_2)
    assert not UnivKansasFeatures.v20_contains_et(para_3)

    para_v20: Para2dStr = tokenize_para("Jack Smith, et al. conducted the experiment.")
    assert UnivKansasFeatures.v20_contains_et(para_v20)
