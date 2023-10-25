import authorship_tool.main.util.feature_counter as fct
import authorship_tool.main.util.frequency_calculator as fcal


class DatasetGenerator:
    def __init__(self, tags: list[str] = None):
        self.columns = [
            "word variation",
            "uncommon word frequency",
            "sentence length",
            "average word length",
        ]
        self.columns.extend(tags)

    def generate_dataset_sent(
        self, sentence: list[str], tags: list[str], correctness: bool
    ) -> tuple[list[float], bool]:
        """文章のリストから特徴量のリストを生成する"""
        freq_dict = fcal.all_pos_frequency(sentence)
        return (
            [
                fcal.word_variation(sentence),
                fcal.uncommon_word_frequency(sentence),
                fct.sentence_length(sentence),
                fcal.average_word_length(sentence),
            ]
            + [freq_dict.get(tag, 0) for tag in tags],
            correctness,
        )

    def generate_dataset_para(
        self, paragraph: list[list[str]], tags: list[str], correctness: bool
    ) -> tuple[list[float], bool]:
        """段落のリストから特徴量のリストを生成する"""
        sentence = [word for sentence in paragraph for word in sentence]
        return self.generate_dataset_sent(sentence, tags, correctness)
