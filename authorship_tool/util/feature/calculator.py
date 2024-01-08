"""特徴計算モジュール
Feature calculation module"""

import nltk
from rich.console import Console

from authorship_tool.types import Sent, Tag, TaggedToken
from authorship_tool.util.feature import counter as f_counter
from authorship_tool.util.feature.pos import PosFeature

console = Console(highlight=False)


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

