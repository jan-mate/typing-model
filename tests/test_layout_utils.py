import os
import pytest
from src.config import DATA_ROOT
from src.enrichment.engine import EnrichmentEngine
from src.utils.layout_utils import build_layout_and_engine

REQUIRED_LAYOUT_KEYS = {"layout_path", "layout_map_path", "shifts_path"}


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_ROOT, "data/layouts/qwerty_us.json")),
    reason="Layout data files not found",
)
def test_build_layout_and_engine_no_rpt():
    layout_paths, engine, bigrams_file = build_layout_and_engine(DATA_ROOT, rpt_key=False)

    assert set(layout_paths.keys()) == REQUIRED_LAYOUT_KEYS
    assert isinstance(engine, EnrichmentEngine)
    assert bigrams_file.endswith(".json")
    assert os.path.exists(bigrams_file)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_ROOT, "data/layouts/qwerty_us_rpt_trg.json")),
    reason="RPT layout data files not found",
)
def test_build_layout_and_engine_rpt():
    layout_paths, engine, bigrams_file = build_layout_and_engine(DATA_ROOT, rpt_key=True)

    assert set(layout_paths.keys()) == REQUIRED_LAYOUT_KEYS
    assert isinstance(engine, EnrichmentEngine)
    assert "rpt_trg" in bigrams_file


@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_ROOT, "data/layouts/qwerty_us_rpt_trg.json")),
    reason="RPT layout data files not found",
)
def test_build_layout_and_engine_rpt_j_variant():
    layout_paths, engine, bigrams_file = build_layout_and_engine(
        DATA_ROOT, rpt_key=True, rpt_variant="j"
    )
    assert "rpt_j" in layout_paths["layout_map_path"]
