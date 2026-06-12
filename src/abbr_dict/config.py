import os

from src.config import STORAGE_ROOT

# repeat-key regime for the whole abbr_dict pipeline; set False for the rpt-off run
RPT_KEY = True


def data_path(name: str) -> str:
    stem, ext = os.path.splitext(name)
    suffix = "" if RPT_KEY else "_rptoff"
    return os.path.join(STORAGE_ROOT, "data/abbr_dict", f"{stem}{suffix}{ext}")
