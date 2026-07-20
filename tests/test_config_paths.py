from asg_transformer.config import PROJECT_ROOT, settings


def test_default_data_directory_is_project_relative_and_exists():
    assert settings.data_dir == (PROJECT_ROOT / "data/processed").resolve()
    assert (settings.data_dir / "techniques.json").is_file()
