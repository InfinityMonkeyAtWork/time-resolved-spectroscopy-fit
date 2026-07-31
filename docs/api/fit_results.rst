Fit Results
===========

Completed-fit inspection, comparison, and plotting. A ``FitResults`` is
an immutable view over persisted fit records (``SavedFitSlot``), obtained
from ``Project.results`` (in-session) or ``FitResults.load(path)``
(archives) — the per-slot accessors and plot methods behave identically
on both. Project-level joint fit records (``JointFitResult``, served by
``find_joint`` / ``get_joint`` / ``plot_joint_mcmc``) are currently
in-session only: archives cannot reconstruct them until schema 7.
The ``File.get_*`` / ``File.plot_*`` / ``File.compare_models`` methods
are thin delegates into this class.

.. autoclass:: trspecfit.fit_results.FitResults
   :members:
   :show-inheritance:

.. autoclass:: trspecfit.utils.fit_io.JointFitResult
   :members:
   :show-inheritance:

.. autoclass:: trspecfit.utils.fit_io.JointFitProjection
   :members:
   :show-inheritance:

.. autoclass:: trspecfit.utils.lmfit.MCMCResult
   :members:
   :show-inheritance:
