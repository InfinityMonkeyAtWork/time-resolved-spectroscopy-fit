Fit Results
===========

Completed-fit inspection, comparison, and plotting. A ``FitResults`` is
an immutable view over persisted fit records (``SavedFitSlot``), obtained
from ``Project.results`` (in-session) or ``FitResults.load(path)``
(archives) — the accessors and plot methods behave identically on both.
The ``File.get_*`` / ``File.plot_*`` / ``File.compare_models`` methods
are thin delegates into this class.

.. autoclass:: trspecfit.fit_results.FitResults
   :members:
   :show-inheritance:

.. autoclass:: trspecfit.utils.lmfit.MCMCResult
   :members:
   :show-inheritance:
