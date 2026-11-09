import json
from functools import lru_cache
from importlib.resources import files

from .state import ModelSpec


@lru_cache(maxsize=1)
def load_catalog() -> list[ModelSpec]:
    package_files = files("backend.agents.model_selector")
    records: list[dict] = []
    for filename in ("models.json", "models_extended.json"):
        catalog_path = package_files.joinpath(filename)
        records.extend(json.loads(catalog_path.read_text(encoding="utf-8")))

    models = [ModelSpec.model_validate(record) for record in records]
    model_ids = [model.model_id for model in models]
    if len(model_ids) != len(set(model_ids)):
        duplicates = sorted(
            model_id for model_id in set(model_ids) if model_ids.count(model_id) > 1
        )
        raise ValueError(f"模型目录存在重复 model_id：{', '.join(duplicates)}")
    return models

