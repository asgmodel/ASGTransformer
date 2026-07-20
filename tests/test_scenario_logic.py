from pathlib import Path
from asg_transformer.core.catalog import KnowledgeCatalog

def test_tactic_order_is_stable():
    catalog=KnowledgeCatalog(Path("data/processed"))
    values=list(catalog.tactic_order.values())
    assert values == sorted(values)
