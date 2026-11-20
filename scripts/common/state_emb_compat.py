"""Launch the official STATE CLI with a legacy pickle-module alias.

The checkpoint refers to vci.data.loader.VCIDatasetSentenceCollator, which is
state.emb.data.loader.VCIDatasetSentenceCollator in the pinned STATE revision.
This process-local alias does not change model code or checkpoint contents.
"""
import sys
import types
from state.emb.data import loader

vci = types.ModuleType('vci')
vci.__path__ = []
data = types.ModuleType('vci.data')
data.__path__ = []
vci.data = data
data.loader = loader
sys.modules['vci'] = vci
sys.modules['vci.data'] = data
sys.modules['vci.data.loader'] = loader

from state.__main__ import main

if __name__ == '__main__':
    sys.exit(main())
