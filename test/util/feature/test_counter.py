from authorship_tool.types import Para2dStr
from authorship_tool.util.feature.calculator import UnivKansasFeatures
from authorship_tool.util.tokenizer import tokenize_to_para

text_1: str = "This is a pen, Mr. Smith. This pen is so expensive, don't you think? Oh, no! He stole my pen...!"
text_2: str = "You got another pen? (It's so nice of you!)"
text_3: str = "This pen is well-designed. I like it."
para_1: Para2dStr = tokenize_to_para(text_1)
para_2: Para2dStr = tokenize_to_para(text_2)
para_3: Para2dStr = tokenize_to_para(text_3)


def test_v1_sentences_per_paragraph():
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_1) == 5
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_2) == 1
    assert UnivKansasFeatures.v1_sentences_per_paragraph(para_3) == 1


def test_v2_words_per_paragraph():
    assert UnivKansasFeatures.v2_words_per_paragraph(para_1) == 29
    assert UnivKansasFeatures.v2_words_per_paragraph(para_2) == 14
    assert UnivKansasFeatures.v2_words_per_paragraph(para_3) == 9


def test__char_exists():
    assert UnivKansasFeatures._char_exists(para_1, ",")
    assert not UnivKansasFeatures._char_exists(para_2, ".")
    assert not UnivKansasFeatures._char_exists(para_3, "!")


def test_v3_close_parenthesis_exists():
    assert not UnivKansasFeatures.v3_close_parenthesis_present(para_1)
    assert UnivKansasFeatures.v3_close_parenthesis_present(para_2)
    assert not UnivKansasFeatures.v3_close_parenthesis_present(para_3)


def test_v4_dash_exists():
    assert not UnivKansasFeatures.v4_dash_present(para_1)
    assert not UnivKansasFeatures.v4_dash_present(para_2)
    assert UnivKansasFeatures.v4_dash_present(para_3)


def test__contains_word_with_stemming():
    assert not UnivKansasFeatures._contains_specific_word(para_1, "thin")
    assert not UnivKansasFeatures._contains_specific_word(para_2, "an")
    assert UnivKansasFeatures._contains_specific_word(para_3, "likes")


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


def test_v19_contains_2_times_more_capitals_than_period():
    assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_1)
    assert UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_2)
    assert not UnivKansasFeatures.v19_contains_2_times_more_capitals_than_period(para_3)


def test_v20_contains_et():
    assert not UnivKansasFeatures.v20_contains_et(para_1)
    assert not UnivKansasFeatures.v20_contains_et(para_2)
    assert not UnivKansasFeatures.v20_contains_et(para_3)

    para_v20: Para2dStr = tokenize_to_para(
        "Jack Smith, et al. conducted the experiment."
    )
    assert UnivKansasFeatures.v20_contains_et(para_v20)
