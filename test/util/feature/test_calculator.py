"""
    特徴量計算モジュールテスト
    Feature calculator module test
"""

import nltk

from authorship_tool.types import Para2dStr
from authorship_tool.util.feature.calculator import UnivKansasFeatures
from authorship_tool.util.tokenizer import tokenize_to_para

nltk.download("stopwords")

text_1: str = (
    "This is a famous pen, Mr. Smith."
    + " It was designed by the famous artist Amy Mahogany,"
    + " and it looks very classy, don't you think so?"
)
text_2: str = "Oh no! Someone has stolen my pen... (I can't believe this is happening!)"
text_3: str = (
    "I'm glad to hear that you found it! This pen is well-designed. I really like it."
)
para_1: Para2dStr = tokenize_to_para(text_1)
para_2: Para2dStr = tokenize_to_para(text_2)
para_3: Para2dStr = tokenize_to_para(text_3)


class TestUnivKansasFeatures:
    @staticmethod
    def test_v1_sentences_per_paragraph():
        assert UnivKansasFeatures.v1_sentences_per_paragraph(para_1) == 2
        assert UnivKansasFeatures.v1_sentences_per_paragraph(para_2) == 2
        assert UnivKansasFeatures.v1_sentences_per_paragraph(para_3) == 3

    @staticmethod
    def test_v2_words_per_paragraph():
        assert UnivKansasFeatures.v2_words_per_paragraph(para_1) == 31
        assert UnivKansasFeatures.v2_words_per_paragraph(para_2) == 19
        assert UnivKansasFeatures.v2_words_per_paragraph(para_3) == 20

    @staticmethod
    def test__char_present():
        assert UnivKansasFeatures._char_present(para_1, ",")
        assert UnivKansasFeatures._char_present(para_2, ".")
        assert not UnivKansasFeatures._char_present(para_3, "?")

    @staticmethod
    def test_v3_close_parenthesis_present():
        assert not UnivKansasFeatures.v3_close_parenthesis_present(para_1)
        assert UnivKansasFeatures.v3_close_parenthesis_present(para_2)
        assert not UnivKansasFeatures.v3_close_parenthesis_present(para_3)

    @staticmethod
    def test_v4_dash_present():
        assert not UnivKansasFeatures.v4_dash_present(para_1)
        assert not UnivKansasFeatures.v4_dash_present(para_2)
        assert UnivKansasFeatures.v4_dash_present(para_3)

    @staticmethod
    def test__contains_specific_word():
        assert not UnivKansasFeatures._contains_specific_word(para_1, "thin")
        assert not UnivKansasFeatures._contains_specific_word(para_2, "an")
        assert UnivKansasFeatures._contains_specific_word(para_3, "likes")

    @staticmethod
    def test_v17_contains_others_or_researchers():
        para_v17_1: Para2dStr = tokenize_to_para(
            "The researchers conducted the experiment."
        )
        para_v17_2: Para2dStr = tokenize_to_para(
            "The other person did not agree with the result."
        )
        para_v17_3: Para2dStr = tokenize_to_para("He conducted a research.")
        para_v17_4: Para2dStr = tokenize_to_para("SHe discriminated against others.")
        assert UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_1)
        assert not UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_2)
        assert not UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_3)
        assert UnivKansasFeatures.v17_contains_others_or_researchers(para_v17_4)

    @staticmethod
    def test_v19_contains_2_times_more_capitals_than_period():
        assert UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_1)
        assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(
            para_2
        )
        assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(
            para_3
        )

    @staticmethod
    def test_v20_contains_et():
        assert not UnivKansasFeatures.v20_contains_et(para_1)
        assert not UnivKansasFeatures.v20_contains_et(para_2)
        assert not UnivKansasFeatures.v20_contains_et(para_3)

        para_v20: Para2dStr = tokenize_to_para(
            "Jack Smith, et al. conducted the experiment."
        )
        assert UnivKansasFeatures.v20_contains_et(para_v20)
