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
