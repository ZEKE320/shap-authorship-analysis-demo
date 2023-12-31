"""特徴計算モジュール
Feature calculation module"""

import re
from typing import Any

import nltk
import numpy as np
from nltk import SnowballStemmer
from rich.console import Console
from nltk import WordNetLemmatizer

from authorship_tool.types import Para2dStr, Sent1dStr, Tag, TaggedTokens, TokenStr
from authorship_tool.util import dim_reshaper
from authorship_tool.util.feature.pos import PosFeature

console = Console(highlight=False)
stemmer: SnowballStemmer = nltk.SnowballStemmer("english")
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()


def word_variation(sent: Sent1dStr) -> float:
    """文章中の単語の豊富さを計算する: (単語の種類の数)/(文章中の単語数)
    Calculate the word variation in a sentence:
    (number of unique words)/(total number of words in the sentence)
    """
    return count_individual_tokens(sent) / sentence_length(sent)


def average_word_length(sent: Sent1dStr) -> float:
    """文章中の単語の平均文字列長を計算する
    Calculate the average word length in a sentence"""
    return sum(len(word) for word in sent) / sentence_length(sent)


def non_alphabetic_characters_frequency(sent: Sent1dStr) -> float:
    """文章内で出現する記号の割合を計算する
    Calculate the frequency of non-alphabetic characters in a sentence"""
    return count_non_alphabetic_characters(sent) / sentence_length(sent)


def uncommon_word_frequency(sent: Sent1dStr) -> float:
    """特徴的な単語の割合を計算する
    Calculate the frequency of uncommon words in a sentence"""
    return count_uncommon_words(sent) / sentence_length(sent)


def all_pos_frequency(sent: Sent1dStr) -> dict[Tag, float]:
    """文章中の各品詞の割合を計算する
    Calculate the frequency of each part of speech in a sentence"""
    pos_feature: PosFeature = PosFeature(sent).tag_subcategories()
    tagged_tokens: list[TaggedTokens] = pos_feature.tagged_tokens

    # TODO 過去分詞形容詞を確認する為なので、後々削除する
    if "JJ_pp" in set(pos for (_, pos) in tagged_tokens):
        console.print(f"{pos_feature}\n")

    freq_dist = nltk.FreqDist(tagged_tokens)

    total_tags: int = freq_dist.N()
    return {tag[1]: count / total_tags for (tag, count) in freq_dist.items()}


def sentence_length(sent: Sent1dStr) -> int:
    """文章中に出現する単語数を計算する"""
    return len(sent)


def count_individual_tokens(sent: Sent1dStr) -> int:
    """文章中に出現する単語の種類数を計算する"""
    return len(set(sent))


def count_character(sent: Sent1dStr, character: str) -> int:
    """文章内で出現する指定した文字の合計を計算する"""
    return sent.count(character)


def count_non_alphabetic_characters(sent: Sent1dStr) -> int:
    """文章内で出現する記号の合計を計算する"""
    pattern = r"[^a-zA-Z\s]"
    matches: list[str] = re.findall(pattern=pattern, string=" ".join(sent))
    return len(matches)


def count_numeric_characters(sent: Sent1dStr) -> int:
    """文章内で出現する数字の合計を計算する"""
    pattern = r"[\d]"
    matches: list[str] = re.findall(pattern=pattern, string=" ".join(sent))
    return len(matches)


def count_uncommon_words(sent: Sent1dStr) -> int:
    """ストップワードではない単語の数を計算する"""
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return len([word for word in sent if word not in stop_words])


class UnivKansasFeatures:
    """カンザス大学の先行研究に基づいた特徴量計算クラス"""

    @staticmethod
    def v1_sentences_per_paragraph(para: Para2dStr) -> int:
        """段落内の文数を計算する"""
        return len(para)

    @staticmethod
    def v2_words_per_paragraph(para: Para2dStr) -> int:
        """段落内で出現する単語の合計を計算する"""
        sent: Sent1dStr = dim_reshaper.para_to_1d(para)
        return sentence_length(sent)

    @staticmethod
    def _char_exists(para: Para2dStr, char: str) -> bool:
        """段落内に指定した文字が存在するかどうかを判定する"""
        return any(char in word for sent in para for word in sent)

    @staticmethod
    def v3_close_parenthesis_present(para: Para2dStr) -> bool:
        """段落内に括弧閉じが存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, ")")

    @staticmethod
    def v4_dash_present(para: Para2dStr) -> bool:
        """段落内にダッシュが存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, "-")

    @staticmethod
    def _semi_colon_present(para: Para2dStr) -> bool:
        """段落内にセミコロンが存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, ";")

    @staticmethod
    def _colon_present(para: Para2dStr) -> bool:
        """段落内にコロンが存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, ":")

    @staticmethod
    def v5_semi_colon_or_colon_present(para: Para2dStr) -> bool:
        """段落内にセミコロンまたはコロンが存在するかどうかを判定する"""
        return UnivKansasFeatures._semi_colon_present(
            para
        ) or UnivKansasFeatures._colon_present(para)

    @staticmethod
    def v6_question_mark_present(para: Para2dStr) -> bool:
        """段落内に疑問符が存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, "?")

    @staticmethod
    def v7_apostrophe_present(para: Para2dStr) -> bool:
        """段落内にアポストロフィが存在するかどうかを判定する"""
        return UnivKansasFeatures._char_exists(para, "'")

    @staticmethod
    def v8_standard_deviation_of_sentence_length(para: Para2dStr) -> float:
        """段落内の文長の標準偏差を計算する"""
        sent_lengths: list[int] = [len(sentence) for sentence in para]
        standard_deviation: np.floating[Any] = np.std(sent_lengths)
        return float(standard_deviation)

    @staticmethod
    def v9_length_difference_for_consecutive_sentences(para: Para2dStr) -> float:
        """段落内の文の前後で、文長の差の平均を計算する"""
        sent_lengths: list[int] = [len(sentence) for sentence in para]
        diffs: list[int] = [
            abs(sent_lengths[i] - sent_lengths[i + 1])
            for i in range(len(sent_lengths) - 1)
        ]
        return sum(diffs) / len(diffs)

    @staticmethod
    def v10_sentence_with_lt_11_words(para: Para2dStr) -> bool:
        """段落内に単語数が11未満の文が存在するかどうかを判定する"""
        return any(len(sentence) < 11 for sentence in para)

    @staticmethod
    def v11_sentence_with_gt_34_words(para: Para2dStr) -> bool:
        """段落内に単語数が34より多い文が存在するかどうかを判定する"""
        return any(len(sentence) > 34 for sentence in para)

    @staticmethod
    def _contains_word(para: Para2dStr, word: str) -> bool:
        """段落内にalthoughが存在するかどうかを判定する"""
        lower_word: str = word.lower()
        return any(
            lower_word == token.lower() for sentence in para for token in sentence
        )

    @staticmethod
    def v12_contains_although(para: Para2dStr) -> bool:
        """段落内にalthoughが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_word(para, "although")

    @staticmethod
    def v13_contains_however(para: Para2dStr) -> bool:
        """段落内にhoweverが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_word(para, "however")

    @staticmethod
    def v14_contains_but(para: Para2dStr) -> bool:
        """段落内にbutが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_word(para, "but")

    @staticmethod
    def v15_contains_because(para: Para2dStr) -> bool:
        """段落内にbecauseが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_word(para, "because")

    @staticmethod
    def v16_contains_this(para: Para2dStr) -> bool:
        """段落内にthisが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_word(para, "this")

    @staticmethod
    def _contains_specific_word(para: Para2dStr, word: str) -> bool:
        """段落内に指定した単語が存在するかどうかを判定する"""
        stem: str = stemmer.stem(word)
        matched_tokens: set[TokenStr] = {
            token
            for sentence in para
            for token in sentence
            if stem == stemmer.stem(token)
        }

        if len(matched_tokens) == 0:
            return False

        return UnivKansasFeatures._check_word_lemma(matched_tokens, word)

    @staticmethod
    def _check_word_lemma(tokens: set[str], target_word: str) -> bool:
        """lemma化した単語が一致するかどうかを判定する"""
        lemmatized_word: str = lemmatizer.lemmatize(target_word)
        return any(lemmatized_word == lemmatizer.lemmatize(token) for token in tokens)

    @staticmethod
    def v17_contains_others_or_researchers(para: Para2dStr) -> bool:
        """段落内にothersまたはresearchersが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_specific_word(
            para, "others"
        ) or UnivKansasFeatures._contains_specific_word(para, "researchers")

    @staticmethod
    def v18_contains_numbers(para: Para2dStr) -> bool:
        """段落内に数字が存在するかどうかを判定する"""
        return any(
            char.isdigit() for sentence in para for token in sentence for char in token
        )

    @staticmethod
    def _count_char(para: Para2dStr, target_char: str) -> int:
        """段落内に指定した文字の合計を計算する"""
        return sum(
            target_char == char
            for sentence in para
            for token in sentence
            for char in token
        )

    @staticmethod
    def v19_contains_2_times_more_capitals_than_period(para: Para2dStr) -> bool:
        """段落内にピリオドよりも大文字が2倍以上存在するかどうかを判定する"""
        return (
            sum(
                char.isupper()
                for sentence in para
                for token in sentence
                for char in token
            )
            > UnivKansasFeatures._count_char(para, ".") * 2
        )

    @staticmethod
    def v20_contains_et(para: Para2dStr) -> bool:
        """段落内にetが存在するかどうかを判定する"""
        return UnivKansasFeatures._contains_specific_word(para, "et")
