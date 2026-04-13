"""
Bypass huggingface-hub version check that breaks transformers imports in the NxDI venv
(huggingface-hub==1.10.1 vs transformers' require <1.0).

Import this module BEFORE any `from transformers import ...` statement.
"""
import sys
import importlib.util

_VERSIONS_PATH = (
    "/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/lib/python3.12/site-packages/"
    "transformers/utils/versions.py"
)

if "transformers.utils.versions" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("transformers.utils.versions", _VERSIONS_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod._compare_versions = lambda *a, **kw: None
    _mod.require_version = lambda *a, **kw: None
    _mod.require_version_core = lambda *a, **kw: None
    sys.modules["transformers.utils.versions"] = _mod
