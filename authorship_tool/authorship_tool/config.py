"""設定モジュール(Configuration Module))"""

from authorship_tool.types import EnvKey, PathStr

PATHS: dict[EnvKey, PathStr] = {
    "path_adjective_past_participle_dataset": "data/john_blake_2023/wordLists/adjectivesPastParticiple",
    "path_text_data_dir": "dump/text_data",
    "path_dump_dataset_dir": "dump/dataset",
    "path_dump_lgbm_model_dir": "dump/lgbm/model",
    "path_dump_shap_figure_dir": "dump/shap/figure",
}
