"""pos.pyのテスト"""

# from io import TextIOWrapper
# from typing import Final

# import pytest
# import pytest_mock

# from authorship_tool.util import PathUtil
# from authorship_tool.util.feature._pos import PosFeature


# def test_initialize_past_participle_adjective_dataset(
#     monkeypatch: pytest.MonkeyPatch, mocker: pytest_mock.MockerFixture
# ) -> None:
#     PROJECT_ROOT_DIR: Final[str] = "path/to/project_root_dir"
#     DATASET_PATH: Final[str] = "path/to/dataset"

#     monkeypatch.setattr(PathUtil, "PROJECT_ROOT_PATH", PROJECT_ROOT_DIR)
#     monkeypatch.setenv("path_adjective_past_participle_dataset", DATASET_PATH)

#     mock_open = mocker.mock_open(read_data="word1\nword2\nword3")
#     mocker.patch("buildins.open")

#     PosFeature.initialize_past_participle_adjective_dataset()

#     assert PosFeature.__PAST_PARTICIPLE_ADJECTIVE_DATASET == [
#         "some",
#         "adjectives",
#         "past",
#         "participle",
#         "words",
#     ]


import nltk

from authorship_tool.util.feature.pos import PosFeature


def test_sentence_contains_extraposition_1():
    """文が外置形容詞を含むかどうかを判定するテスト1"""

    pos_feature = PosFeature(
        nltk.pos_tag(nltk.word_tokenize("It is obvious that you have been misled."))
    )
    result = pos_feature.tag_jj_extraposition()
    assert "JJ_exp" in [tag for _, tag in result.tagged_tokens]


def test_sentence_contains_extraposition_2():
    """文が外置形容詞を含むかどうかを判定するテスト2"""

    pos_feature = PosFeature(
        nltk.pos_tag(
            nltk.word_tokenize("It's a shame what happened to you and your sister.")
        )
    )
    result = pos_feature.tag_jj_extraposition()
    assert "JJ_exp" not in [tag for _, tag in result.tagged_tokens]


def test_sentence_contains_extraposition_3():
    """文が外置形容詞を含むかどうかを判定するテスト3"""
    pos_feature = PosFeature(
        nltk.pos_tag(
            nltk.word_tokenize(
                "It might be a good idea to wear a respirator mask when you're working with fiberglass."
            )
        )
    )
    result = pos_feature.tag_jj_extraposition()
    assert "JJ_exp" in [tag for _, tag in result.tagged_tokens]


def test_sentence_contains_extraposition_4():
    """文が外置形容詞を含むかどうかを判定するテスト4"""
    pos_feature = PosFeature(
        nltk.pos_tag(
            nltk.word_tokenize(
                "It's likely that the enemy simply dropped back off the hilltop once they'd grabbed all the weapons they could carry."
            )
        )
    )
    result = pos_feature.tag_jj_extraposition()
    assert "JJ_exp" in [tag for _, tag in result.tagged_tokens]


def test_sentence_contains_extraposition_5():
    """文が外置形容詞を含むかどうかを判定するテスト5"""
    pos_feature = PosFeature(
        nltk.pos_tag(
            nltk.word_tokenize(
                "It surprised everybody that Marlene had so much energy and strength."
            )
        )
    )
    result = pos_feature.tag_jj_extraposition()
    assert "JJ_exp" not in [tag for _, tag in result.tagged_tokens]
