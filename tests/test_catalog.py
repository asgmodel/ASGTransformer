from pathlib import Path
from asg_transformer.core.catalog import KnowledgeCatalog

def test_catalog_counts():
    catalog=KnowledgeCatalog(Path("data/processed"))
    assert len(catalog.techniques)==78
    assert len(catalog.tactics)==12
    assert len(catalog.software)==19
    assert len(catalog.groups)==13
