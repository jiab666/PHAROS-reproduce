"""Run the unchanged official preprocessing script with a SciPy API alias.

Scanpy 1.11.5 calls getnnz on a SciPy csr_array during Scrublet. SciPy
exposes the identical operation as _getnnz on sparse arrays. The alias is
process-local and does not change the preprocessing algorithm or source.
"""
import runpy
import sys

from scipy.sparse import csr_array

if not hasattr(csr_array, "getnnz"):
    csr_array.getnnz = csr_array._getnnz
    print("Compatibility alias: csr_array.getnnz = csr_array._getnnz", flush=True)

script = sys.argv[1]
sys.argv = [script, *sys.argv[2:]]
runpy.run_path(script, run_name="__main__")
