"""
特徴量計算モジュール
Feature calculation module
"""

import re

import nltk
import numpy as np
from nltk import SnowballStemmer, WordNetLemmatizer
from numpy.typing import NDArray
from rich.console import Console

from authorship_tool.types_ import (
    Char,
    Para2dStr,
    Sent1dStr,
    Tag,
    TaggedToken,
    TokenStr,
)
from authorship_tool.util import dim_reshaper
from authorship_tool.util.feature.pos import PosFeature
from authorship_tool.util.feature.regex import NUMERIC_VALUE_PATTERN

console = Console(highlight=False)

stop_words = set(nltk.corpus.stopwords.words("english"))
stemmer: SnowballStemmer = SnowballStemmer("english")
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()


class SentenceCalculator:
    """
    文特徴量計算クラス
    Sentence feature calculation class
    """

    @staticmethod
    def count_total_tokens(sent: Sent1dStr) -> int:
        """
        文中のトークン数を計算する
        Count the number of tokens in a sentence

        Args:
            sent (Sent1dStr): 文章 (Sentence)

        Returns:
            int: トークン数 (Number of tokens)
        """
        return len(sent)

    @staticmethod
    def count_individual_tokens(sent: Sent1dStr) -> int:
        """
        文中に出現する一意の単語数を計算する
        Count the number of unique words in a sentence

        Args:
            sent (Sent1dStr): 文章 (Sentence)
        Returns:
            int: 出現する一意の単語数 (Number of unique words)
        """
        return len(set(sent))

    @staticmethod
    def count_total_characters(sent: Sent1dStr) -> int:
        """
        文中の文字数を計算する
        Count the number of characters in a sentence

        Args:
            sent (Sent1dStr): 文章 (Sentence)

        Returns:
            int: 文中の文字数 (Number of characters in a sentence)
        """
        return sum(len(token) for token in sent)

    @staticmethod
    def count_character(sent: Sent1dStr, character: str) -> int:
        """
        文中で出現する指定した文字の合計を計算する
        Count the total number of specified characters in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)
            character (str): 文字 (Character)

        Returns:
            int: 文中で出現する指定した文字の合計 (Total number of specified characters in a sentence)
        """
        if len(character) != 1:
            raise ValueError("character must be a single character")

        return sum(token.count(character) for token in sent)

    @staticmethod
    def count_non_alphabetic_characters(sent: Sent1dStr) -> int:
        """
        文中で出現するアルファベット以外の文字数を計算する
        Count the number of non-alphabetic characters in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            int: 文中で出現するアルファベット以外の文字数 (Number of non-alphabetic characters in a sentence)
        """
        non_alpha_list: list[Char] = [
            char for token in sent for char in token if not char.isalpha()
        ]

        return len(non_alpha_list)

    @staticmethod
    def count_numeric_characters(sent: Sent1dStr) -> int:
        """
        文中で出現する数字の文字数を計算する
        Count the number of numeric characters in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            int: 文中で出現する数字の文字数 (Number of numeric characters in a sentence)
        """
        numeric_chars: list[Char] = [
            char for token in sent for char in token if char.isdecimal()
        ]

        return len(numeric_chars)

    @staticmethod
    def count_numeric_values(sent: Sent1dStr) -> int:
        """
        文内で出現する数値の出現数を計算する
        Count the number of numeric values in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            int: 文中で出現する数値の出現数 (Number of numeric values in a sentence)
        """
        numeric_values: list[str] = [
            matched
            for token in sent
            for matched in re.findall(NUMERIC_VALUE_PATTERN, token)
        ]

        return len(numeric_values)

    @staticmethod
    def count_uncommon_words(sent: Sent1dStr) -> int:
        """
        ストップワードではない単語の数を計算する
        Count the number of words that are not stop words

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            int: ストップワードではない単語の数 (Number of words that are not stop words)
        """
        not_stop_words: list[TokenStr] = [
            word for word in sent if word not in stop_words
        ]

        return len(not_stop_words)

    @staticmethod
    def word_variation(sent: Sent1dStr) -> np.float64:
        """
        文中の単語の豊富さを計算する
        Calculate the word variation in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: 単語の豊富さ (Word variation)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_individual_tokens(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def average_word_length(sent: Sent1dStr) -> np.float64:
        """
        文中の単語の平均単語長を計算する
        Calculate the average word length in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: 平均文字列長 (Average word length)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_total_characters(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def non_alphabetic_characters_frequency(sent: Sent1dStr) -> np.float64:
        """
        文内で出現する記号の割合を計算する
        Calculate the frequency of non-alphabetic characters in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: 記号の割合 (Frequency of symbols)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_non_alphabetic_characters(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def uncommon_word_frequency(sent: Sent1dStr) -> np.float64:
        """
        文中の特徴的な単語の割合を計算する
        Calculate the frequency of uncommon words in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: 特徴的な単語の割合 (Frequency of uncommon words)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_uncommon_words(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def non_alphabetic_character_frequency(sent: Sent1dStr) -> np.float64:
        """
        文中で出現するアルファベット以外の文字の割合を計算する
        Calculate the frequency of non-alphabetic characters in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: アルファベット以外の文字の割合 (Frequency of non-alphabetic characters)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_non_alphabetic_characters(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def numeric_value_frequency(sent: Sent1dStr) -> np.float64:
        """
        文中の数値の割合を計算する
        Calculate the frequency of numeric values in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            np.float64: 数値の割合 (Frequency of numeric values)
        """
        if (sent_length := SentenceCalculator.count_total_tokens(sent)) == 0:
            return np.float64(0)

        return np.divide(
            SentenceCalculator.count_numeric_values(sent),
            sent_length,
            dtype=np.float64,
        )

    @staticmethod
    def pos_frequencies(sent: Sent1dStr) -> dict[Tag, np.float64]:
        """
        文中の各品詞の割合を計算する
        Calculate the frequency of each part of speech in a sentence

        Args:
            sent (Sent1dStr): 文 (Sentence)

        Returns:
            dict[Tag, np.float64]: 品詞とその割合 (POS and its frequency)
        """
        pos_feature: PosFeature = PosFeature(sent).tag_subcategories()
        tagged_tokens: list[TaggedToken] = pos_feature.tagged_tokens

        tags: list[Tag] = [tag for (_, tag) in tagged_tokens]
        # NOTE 各タグの出現数を計算するため、ここでsetに変換してはいけない
        # Do not convert to a set here to calculate the number of occurrences of each tag

        total_token_count: int = len(tags)

        return {
            tag: (
                np.divide(
                    tags.count(tag),
                    total_token_count,
                    dtype=np.float64,
                )
                if total_token_count != 0
                else np.float64(0)
            )
            for tag in set(tags)
        }


class ParagraphCalculator:
    """
    段落特徴量計算クラス
    Paragraph feature calculation class
    """

    @staticmethod
    def count_total_tokens(para: Para2dStr) -> int:
        """
        段落中の単語数を計算する
        Calculate the number of tokens in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            int: 段落中の単語数 (Number of tokens in a paragraph)
        """
        return SentenceCalculator.count_total_tokens(dim_reshaper.reduce_dim(para))

    @staticmethod
    def count_total_sentences(para: Para2dStr) -> int:
        """
        段落中の文数を計算する
        Calculate the number of sentences in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            int: 段落中の文数 (Number of sentences in a paragraph)
        """
        return len(para)

    @staticmethod
    def word_variation(para: Para2dStr) -> np.float64:
        """
        段落中の単語の豊富さを計算する
        Calculate the word variation in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 単語の豊富さ (Word variation)
        """

        return SentenceCalculator.word_variation(dim_reshaper.reduce_dim(para))

    @staticmethod
    def average_word_length(para: Para2dStr) -> np.float64:
        """
        段落中の単語の平均単語長を計算する
        Calculate the average word length in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 平均文字列長 (Average word length)
        """

        return SentenceCalculator.average_word_length(dim_reshaper.reduce_dim(para))

    @staticmethod
    def non_alphabetic_characters_frequency(para: Para2dStr) -> np.float64:
        """
        段落内で出現する記号の割合を計算する
        Calculate the frequency of non-alphabetic characters in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 記号の割合 (Frequency of symbols)
        """

        return SentenceCalculator.non_alphabetic_characters_frequency(
            dim_reshaper.reduce_dim(para)
        )

    @staticmethod
    def uncommon_word_frequency(para: Para2dStr) -> np.float64:
        """
        特徴的な単語の割合を計算する
        Calculate the frequency of uncommon words in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 特徴的な単語の割合 (Frequency of uncommon words)
        """

        return SentenceCalculator.uncommon_word_frequency(dim_reshaper.reduce_dim(para))

    @staticmethod
    def numeric_value_frequency(para: Para2dStr) -> np.float64:
        """
        数値の割合を計算する
        Calculate the frequency of numeric values in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 数値の割合 (Frequency of numeric values)
        """

        return SentenceCalculator.numeric_value_frequency(dim_reshaper.reduce_dim(para))

    @staticmethod
    def pos_frequencies(para: Para2dStr) -> dict[Tag, np.float64]:
        """
        段落中の各品詞の割合を計算する
        Calculate the frequency of each part of speech in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            dict[Tag, np.float64]: 品詞とその割合 (POS and its frequency)
        """

        return SentenceCalculator.pos_frequencies(dim_reshaper.reduce_dim(para))


class UnivKansasFeatures:
    """
    カンザス大学の先行研究に基づく特徴量計算クラス
    University of Kansas's previous research based feature calculation class
    """

    @staticmethod
    def v1_sentences_per_paragraph(para: Para2dStr) -> int:
        """
        段落内の文数を計算する
        Calculate the number of sentences in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            int: 段落内の文数 (Number of sentences in a paragraph)
        """
        return ParagraphCalculator.count_total_sentences(para)

    @staticmethod
    def v2_words_per_paragraph(para: Para2dStr) -> int:
        """
        段落内で出現する単語の合計を計算する
        Calculate the total number of words in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            int: 段落内で出現する単語の合計 (Total number of words in a paragraph)
        """
        sent: Sent1dStr = dim_reshaper.reduce_dim(para)
        return SentenceCalculator.count_total_tokens(sent)

    @staticmethod
    def _char_present(para: Para2dStr, char: str) -> bool:
        """
        段落内に指定した文字が存在するかどうかを判定する
        Determine whether the specified character exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)
            char (str): 文字 (Character)

        Returns:
            bool: 文字の存在有無 (Presence or absence of character)
        """

        if len(char) != 1:
            raise ValueError("char must be a single character")

        return any(char in word for sent in para for word in sent)

    @staticmethod
    def v3_close_parenthesis_present(para: Para2dStr) -> bool:
        """
        段落内に括弧閉じが存在するかどうかを判定する
        Determine whether the closing parenthesis exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: 括弧閉じの存在有無 (Presence or absence of closing parenthesis)
        """
        return UnivKansasFeatures._char_present(para, ")")

    @staticmethod
    def v4_dash_present(para: Para2dStr) -> bool:
        """
        段落内にダッシュが存在するかどうかを判定する
        Determine whether the dash exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: ダッシュの存在有無 (Presence or absence of dash)
        """
        return UnivKansasFeatures._char_present(para, "-")

    @staticmethod
    def _semi_colon_present(para: Para2dStr) -> bool:
        """
        段落内にセミコロンが存在するかどうかを判定する
        Determine whether the semicolon exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: セミコロンの存在有無 (Presence or absence of semicolon)
        """
        return UnivKansasFeatures._char_present(para, ";")

    @staticmethod
    def _colon_present(para: Para2dStr) -> bool:
        """
        段落内にコロンが存在するかどうかを判定する
        Determine whether the colon exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: コロンの存在有無 (Presence or absence of colon)
        """
        return UnivKansasFeatures._char_present(para, ":")

    @staticmethod
    def v5_semi_colon_or_colon_present(para: Para2dStr) -> bool:
        """
        段落内にセミコロンまたはコロンが存在するかどうかを判定する
        Determine whether the semicolon or colon exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: セミコロンまたはコロンの存在有無 (Presence or absence of semicolon or colon)
        """
        return UnivKansasFeatures._semi_colon_present(
            para
        ) or UnivKansasFeatures._colon_present(para)

    @staticmethod
    def v6_question_mark_present(para: Para2dStr) -> bool:
        """
        段落内に疑問符が存在するかどうかを判定する
        Determine whether the question mark exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: 疑問符の存在有無 (Presence or absence of question mark)
        """
        return UnivKansasFeatures._char_present(para, "?")

    @staticmethod
    def v7_apostrophe_present(para: Para2dStr) -> bool:
        """
        段落内にアポストロフィが存在するかどうかを判定する
        Determine whether the apostrophe exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: アポストロフィの存在有無 (Presence or absence of apostrophe)
        """
        return UnivKansasFeatures._char_present(para, "'")

    @staticmethod
    def v8_standard_deviation_of_sentence_length(para: Para2dStr) -> np.float64:
        """
        段落内の文長の標準偏差を計算する
        Calculate the standard deviation of sentence length in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 段落内の文長の標準偏差 (Standard deviation of sentence length in a paragraph)
        """
        sent_lengths: list[int] = [len(sentence) for sentence in para]
        standard_deviation: np.float64 = np.std(sent_lengths, dtype=np.float64)
        return standard_deviation

    @staticmethod
    def v9_length_difference_for_consecutive_sentences(para: Para2dStr) -> np.float64:
        """
        段落内の文の前後で、文長の差の平均を計算する
        Calculate the average difference in sentence length between sentences before and after a sentence in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            np.float64: 文の前後での文長の差の平均値 (Average difference in sentence length between sentences before and after a sentence)
        """
        sent_lengths: NDArray[np.int64] = np.array([len(sentence) for sentence in para])
        diffs: NDArray[np.int64] = np.abs(np.diff(sent_lengths))

        if diffs.size == 0:
            return np.float64(0)

        return np.divide(
            np.sum(diffs),
            diffs.size,
            dtype=np.float64,
        )

    @staticmethod
    def v10_sentence_with_lt_11_words(para: Para2dStr) -> bool:
        """
        段落内に単語数が11未満の文が存在するかどうかを判定する
        Determine whether there are sentences with less than 11 words in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: 11未満の単語数を持つ文が存在する場合はTrue、それ以外はFalse（真偽値）
        """
        return any(len(sentence) < 11 for sentence in para)

    @staticmethod
    def v11_sentence_with_gt_34_words(para: Para2dStr) -> bool:
        """
        段落内に単語数が34より多い文が存在するかどうかを判定する
        Determine whether there are sentences with more than 34 words in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: 34より多い単語数を持つ文が存在する場合はTrue、それ以外はFalse（真偽値）
        """
        return any(len(sentence) > 34 for sentence in para)

    @staticmethod
    def _contains_word(para: Para2dStr, word: str) -> bool:
        """
        段落内に指定した単語が存在するかどうかを判定する
        Determine whether the specified word exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)
            word (str): 単語 (Word)

        Returns:
            bool: 単語の存在有無 (Presence or absence of word)
        """
        lower_word: str = word.lower()

        return any(
            lower_word == token.lower() for sentence in para for token in sentence
        )

    @staticmethod
    def v12_contains_although(para: Para2dStr) -> bool:
        """
        段落内に"although"が存在するかどうかを判定する
        Determine whether "although" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)
        Returns:
            bool: "although"の存在有無 (Presence or absence of "although")
        """

        return UnivKansasFeatures._contains_word(para, "although")

    @staticmethod
    def v13_contains_however(para: Para2dStr) -> bool:
        """
        段落内に"however"が存在するかどうかを判定する
        Determine whether "however" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: "however"の存在有無 (Presence or absence of "however")
        """

        return UnivKansasFeatures._contains_word(para, "however")

    @staticmethod
    def v14_contains_but(para: Para2dStr) -> bool:
        """
        段落内に"but"が存在するかどうかを判定する
        Determine whether "but" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)
        Returns:
            bool: "but"の存在有無 (Presence or absence of "but")
        """

        return UnivKansasFeatures._contains_word(para, "but")

    @staticmethod
    def v15_contains_because(para: Para2dStr) -> bool:
        """
        段落内に"because"が存在するかどうかを判定する
        Determine whether "because" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: "because"の存在有無 (Presence or absence of "because")
        """

        return UnivKansasFeatures._contains_word(para, "because")

    @staticmethod
    def v16_contains_this(para: Para2dStr) -> bool:
        """
        段落内にthisが存在するかどうかを判定する

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: "this"の存在有無 (Presence or absence of "this")
        """

        return UnivKansasFeatures._contains_word(para, "this")

    @staticmethod
    def _contains_word_including_derived_forms(para: Para2dStr, word: str) -> bool:
        """
        段落内に指定した単語が派生形を含めて存在するかどうかを判定する
        Determine whether the specified word exists in the paragraph, including derived forms

        Args:
            para (Para2dStr): 段落 (Paragraph)
            word (str): 単語 (Word)

        Returns:
            bool: 派生形を含めた単語の存在有無 (Presence or absence of word, including derived forms)
        """

        matched_tokens: set[TokenStr] = UnivKansasFeatures._obtain_words_matching_stem(
            para, word
        )

        if len(matched_tokens) == 0:
            return False

        return UnivKansasFeatures._check_word_lemma(matched_tokens, word)

    @staticmethod
    def _obtain_words_matching_stem(para: Para2dStr, word: str) -> set[TokenStr]:
        """
        段落内に指定した語幹を持つ単語を取得する
        Obtain words in a paragraph that have the specified stem

        Args:
            para (Para2dStr): 段落 (Paragraph)
            word (str): 単語 (Word)

        Returns:
            set[TokenStr]: 指定した語幹を持つ単語 (Words with the specified stem)
        """

        stem: str = stemmer.stem(word)

        return {
            token
            for sentence in para
            for token in sentence
            if stemmer.stem(token) == stem
        }

    @staticmethod
    def _check_word_lemma(tokens: set[str], target_word: str) -> bool:
        """
        レンマ化した単語が一致するかどうかを判定する
        Determine whether the lemmatized words match

        Args:
            tokens (set[str]): 単語 (Words)
            target_word (str): 比較対象の単語 (Word to be compared)

        Returns:
            bool: レンマ化した単語の一致有無 (Presence or absence of matching lemmatized words)
        """

        lemmatized_word: str = lemmatizer.lemmatize(target_word)

        return any(lemmatized_word == lemmatizer.lemmatize(token) for token in tokens)

    @staticmethod
    def v17_contains_others_or_researchers(para: Para2dStr) -> bool:
        """
        段落内にothersまたはresearchersが存在するかどうかを判定する
        Determine whether "others" or "researchers" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: "others"または"researchers"の存在有無 (Presence or absence of "others" or "researchers")
        """
        return UnivKansasFeatures._contains_word_including_derived_forms(
            para, "others"
        ) or UnivKansasFeatures._contains_word_including_derived_forms(
            para, "researchers"
        )

    @staticmethod
    def v18_contains_numbers(para: Para2dStr) -> bool:
        """
        段落内に数字が存在するかどうかを判定する
        Determine whether numbers exist in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: 数字の存在有無 (Presence or absence of numbers)
        """
        return any(
            char.isdigit() for sentence in para for token in sentence for char in token
        )

    @staticmethod
    def _count_char(para: Para2dStr, target_char: str) -> int:
        """
        段落内に指定した文字の合計を計算する
        Count the total number of specified characters in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)
            target_char (str): 文字 (Character)

        Returns:
            int: 段落内に指定した文字の合計 (Total number of specified characters in a paragraph)
        """
        if len(target_char) != 1:
            raise ValueError("target_char must be a single character")

        return sum(token.count(target_char) for sentence in para for token in sentence)

    @staticmethod
    def v19_contains_2_times_more_capitals_than_period(para: Para2dStr) -> bool:
        """
        段落内にピリオドよりも大文字が2倍以上存在するかどうかを判定する
        Determine whether there are more than twice as many capital letters as periods in a paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: ピリオドの2倍以上の大文字が存在するかどうか (Whether there are more than twice as many capital letters as periods)
        """

        return np.greater(
            sum(
                char.isupper()
                for sentence in para
                for token in sentence
                for char in token
            ),
            np.multiply(UnivKansasFeatures._count_char(para, "."), 2),
            dtype=bool,
        )

    @staticmethod
    def v20_contains_et(para: Para2dStr) -> bool:
        """
        段落内にetが存在するかどうかを判定する
        Determine whether "et" exists in the paragraph

        Args:
            para (Para2dStr): 段落 (Paragraph)

        Returns:
            bool: "et"の存在有無 (Presence or absence of "et")
        """
        return UnivKansasFeatures._contains_word_including_derived_forms(para, "et")
