"""
Declared measurement noise: residuals and Jacobian factors in likelihood units.

``NoiseModel`` owns one file's declared noise (``unknown`` / ``gaussian`` /
``poisson``), its reduction to a fitted data view (:meth:`NoiseModel.for_view`),
the residual the optimizer minimizes (:meth:`NoiseModel.apply`) and the chain
factor ``dr/dm`` that turns a model Jacobian into a residual Jacobian
(:meth:`NoiseModel.jacobian_factor`). ``SegmentedNoise`` carries one model per
segment for joint fits, whose residual is the concatenation of per-file
residuals.

Residual per point:

===========  ==========================================================
``unknown``  ``d - m`` (unweighted, the historical behavior)
``gaussian`` ``(d - m) / sigma``
``poisson``  ``sign(d - m) * sqrt(2 * scale * kl_div(d, m))`` (deviance)
===========  ==========================================================

``scale`` is counts per data unit, so ``scale * m`` is the expected count in a
bin. The deviance is the exact Poisson likelihood; zero-count bins need no
special case, only a floor on the *model* keeps the optimizer away from the
singularity at ``m = 0`` (see :data:`EPS_COUNTS`).

The module depends on numpy and ``scipy.special`` only — no trspecfit import —
so the authoring layer (``File``) and the fitting layer (``fitlib``) can both
use it without an import cycle.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import kl_div

NoiseKind = Literal["unknown", "gaussian", "poisson"]

NOISE_TYPE_UNKNOWN = "unknown"
NOISE_TYPE_GAUSSIAN = "gaussian"
NOISE_TYPE_POISSON = "poisson"
NOISE_TYPES = (NOISE_TYPE_UNKNOWN, NOISE_TYPE_GAUSSIAN, NOISE_TYPE_POISSON)

EPS_COUNTS = 1e-3
"""Model floor for the Poisson residual, in counts.

The deviance and its derivative both diverge as the model approaches zero
while the data are positive, so the model is floored at ``EPS_COUNTS / scale``
data units. A thousandth of a count is far below any measurable signal (a
single count is the smallest observable increment), so no realistic fit window
reaches it, while it keeps ``sqrt(D)`` and ``1/m`` bounded for any parameter
set the optimizer proposes mid-search.
"""

LIMIT_REL_TOL = 1e-6
"""Relative width of the ``d ~ m`` branch of the Poisson Jacobian factor.

The closed form ``|m - d| / (m * sqrt(D))`` is 0/0 at ``d = m`` and loses
digits near it, because ``D`` is quadratic in ``m - d``. Inside
``|m - d| <= LIMIT_REL_TOL * m`` the analytic limit ``-sqrt(scale / m)`` is
used instead; at that width the two agree to about one part in ``1e-12``.
"""


#
#
@dataclass(frozen=True, eq=False)
class NoiseModel:
    """
    Declared noise of one data view.

    Instances are immutable: ``for_view`` returns a new model aligned with the
    fitted view, so ``apply`` and ``jacobian_factor`` never need to know about
    fit windows or slice selections.

    Parameters
    ----------
    kind : {'unknown', 'gaussian', 'poisson'}
        Noise model. ``'unknown'`` reproduces the unweighted residual and
        rejects both ``sigma`` and ``scale``.
    sigma : float or ndarray, optional
        Gaussian standard deviation, scalar (constant σ) or an array
        broadcastable to the data view (per-point σ). Required for
        ``'gaussian'``, rejected otherwise. Arrays are copied and frozen.
    scale : float, optional
        Poisson counts per data unit (data are counts for ``scale = 1``).
        Defaults to 1.0 for ``'poisson'``, rejected otherwise.

    Attributes
    ----------
    is_weighted : bool
        Whether the residual carries likelihood units, i.e. ``kind`` is not
        ``'unknown'``.
    """

    kind: NoiseKind
    sigma: float | np.ndarray | None = None
    scale: float | None = None

    #
    def __post_init__(self) -> None:
        if self.kind not in NOISE_TYPES:
            raise ValueError(
                f"noise kind must be one of {NOISE_TYPES}; got {self.kind!r}"
            )
        if self.kind == NOISE_TYPE_GAUSSIAN:
            if self.scale is not None:
                raise ValueError(
                    "scale belongs to poisson noise; gaussian noise takes sigma"
                )
            if self.sigma is None:
                raise ValueError("gaussian noise requires sigma")
            object.__setattr__(self, "sigma", _freeze_sigma(self.sigma))
        elif self.kind == NOISE_TYPE_POISSON:
            if self.sigma is not None:
                raise ValueError(
                    "sigma belongs to gaussian noise; poisson noise takes scale"
                )
            scale = 1.0 if self.scale is None else float(self.scale)
            if not (np.isfinite(scale) and scale > 0):
                raise ValueError(
                    f"poisson scale must be a finite positive number; got {scale!r}"
                )
            object.__setattr__(self, "scale", scale)
        elif self.sigma is not None or self.scale is not None:
            raise ValueError(
                "unknown noise takes neither sigma nor scale; declare "
                "kind='gaussian' or kind='poisson' to weight the residual"
            )

    #
    @property
    def is_weighted(self) -> bool:
        """Whether the residual is in likelihood units."""

        return self.kind != NOISE_TYPE_UNKNOWN

    #
    def validate_data(self, data: ArrayLike) -> None:
        """
        Check a data array against the declared model's domain.

        Only ``'poisson'`` restricts the data: counts cannot be negative.
        ``NaN`` passes — non-finite values outside the fit window are legal
        and ``fit_wrapper`` rejects them inside the window — but ``±inf``
        does not.

        Raises
        ------
        ValueError
            If ``'poisson'`` is declared and *data* holds negative or
            infinite values.
        """

        if self.kind != NOISE_TYPE_POISSON:
            return
        arr = np.asarray(data, dtype=float)
        bad = int(np.count_nonzero((arr < 0) | np.isinf(arr)))
        if bad > 0:
            raise ValueError(
                f"poisson noise requires non-negative data; {bad} value(s) are "
                "negative or infinite. Use gaussian for dark-subtracted or "
                "difference data."
            )

    #
    def for_view(
        self,
        *,
        rows: int | slice | Sequence[int] | np.ndarray | None = None,
        e_window: slice | None = None,
        average: int | None = None,
    ) -> "NoiseModel":
        """
        Reduce the model to the data view a fit actually sees.

        Reductions compose in the order *rows* → *average* → *e_window*.

        Parameters
        ----------
        rows : int, slice, sequence of int or ndarray, optional
            Selection along the time axis (axis 0) of a per-point σ array.
            An ``int`` picks a single slice and drops the axis.
        e_window : slice, optional
            Fit window along the energy axis (the last axis of σ).
        average : int, optional
            Number of time slices averaged into the fitted view (baseline
            fits, ``time_range`` means). ``None`` or 1 means no averaging.

        Returns
        -------
        NoiseModel
            Model aligned with the view. Gaussian: constant σ becomes
            ``σ / √average``; a per-point σ keeps the shape left by the
            selection (2D for a row slice, 1D for a row index or after
            averaging), reduced pixel-wise to ``sqrt(Σ_t σ_t²) / average``
            when averaging. Poisson: ``scale`` becomes ``scale * average``
            (the sum of *average* Poisson draws is Poisson), σ stays absent.
            ``'unknown'`` is returned unchanged.

        Raises
        ------
        ValueError
            If *average* is below 1, or if the rows selected from a per-point
            σ do not match the number of averaged slices.
        """

        n_avg = 1 if average is None else int(average)
        if n_avg < 1:
            raise ValueError(f"average must be a positive slice count; got {average!r}")

        if self.kind == NOISE_TYPE_UNKNOWN:
            return self
        if self.kind == NOISE_TYPE_POISSON:
            if n_avg == 1:
                return self
            assert self.scale is not None  # type guard
            return NoiseModel(kind=self.kind, scale=self.scale * n_avg)

        assert self.sigma is not None  # type guard
        if not isinstance(self.sigma, np.ndarray):
            sigma_const = float(self.sigma)
            if n_avg > 1:
                sigma_const = sigma_const / float(np.sqrt(n_avg))
            return NoiseModel(kind=self.kind, sigma=sigma_const)

        sigma_view = self.sigma if rows is None else np.asarray(self.sigma[rows])
        if n_avg > 1:
            if sigma_view.ndim < 2 or sigma_view.shape[0] != n_avg:
                raise ValueError(
                    f"averaging {n_avg} slices needs a per-point sigma with "
                    f"{n_avg} rows; the selected view has shape "
                    f"{sigma_view.shape}"
                )
            sigma_view = np.sqrt(np.sum(sigma_view**2, axis=0)) / n_avg
        if e_window is not None:
            sigma_view = sigma_view[..., e_window]
        return NoiseModel(kind=self.kind, sigma=sigma_view)

    #
    def apply(self, data: ArrayLike, model: ArrayLike) -> np.ndarray:
        """
        Residual in likelihood units for one view.

        Parameters
        ----------
        data : array_like
            Observed data on the fitted view.
        model : array_like
            Model prediction on the same view.

        Returns
        -------
        ndarray
            Residual, broadcast to the common shape of *data* and *model*.
            Nothing raises and no warning is emitted for finite input: the
            Poisson branch floors the model at ``EPS_COUNTS / scale`` and
            extends the residual linearly below it (``d > 0``), while
            ``d = 0`` scores ``m <= 0`` exactly like ``m = 0``.
        """

        d = np.asarray(data, dtype=float)
        m = np.asarray(model, dtype=float)
        if self.kind == NOISE_TYPE_GAUSSIAN:
            return np.asarray((d - m) / self.sigma)
        if self.kind == NOISE_TYPE_POISSON:
            assert self.scale is not None  # type guard
            return _poisson_residual(d, m, self.scale)
        unweighted: np.ndarray = d - m
        return unweighted

    #
    def jacobian_factor(self, data: ArrayLike, model: ArrayLike) -> np.ndarray:
        """
        Chain factor ``dr/dm`` of the residual with respect to the model.

        Multiply a model Jacobian ``dm/dtheta`` by this factor to obtain the
        residual Jacobian. The factor is negative everywhere.

        Parameters
        ----------
        data : array_like
            Observed data on the fitted view.
        model : array_like
            Model prediction on the same view.

        Returns
        -------
        ndarray
            ``-1`` (``'unknown'``), ``-1 / sigma`` (``'gaussian'``) or
            ``-sqrt(scale / 2) * |m - d| / (m * sqrt(D))`` with
            ``D = kl_div(d, m)`` (``'poisson'``), the latter replaced by its
            analytic limit ``-sqrt(scale / m)`` on the ``|m - d| <=
            LIMIT_REL_TOL * m`` branch and evaluated at the floor
            ``EPS_COUNTS / scale`` below it. Broadcast to the common shape of
            *data* and *model*.
        """

        d = np.asarray(data, dtype=float)
        m = np.asarray(model, dtype=float)
        if self.kind == NOISE_TYPE_GAUSSIAN:
            factor = -1.0 / np.asarray(self.sigma, dtype=float)
            shape = np.broadcast_shapes(d.shape, m.shape, factor.shape)
            return np.broadcast_to(factor, shape)
        if self.kind == NOISE_TYPE_POISSON:
            assert self.scale is not None  # type guard
            m_floor = EPS_COUNTS / self.scale
            return _poisson_slope(d, np.maximum(m, m_floor), self.scale)
        return np.broadcast_to(-1.0, np.broadcast_shapes(d.shape, m.shape))


#
#
@dataclass(frozen=True, eq=False)
class SegmentedNoise:
    """
    One noise model per segment of a concatenated residual (joint fits).

    Parameters
    ----------
    models : sequence of NoiseModel
        One model per segment, σ already reduced to that segment's view.
    lengths : sequence of int
        Number of residual points per segment, in the same order. They must
        sum to the length of the concatenated data vector.

    Raises
    ------
    ValueError
        If the two sequences disagree in length, if a segment length is not
        positive, or if ``'unknown'`` is mixed with a weighted segment (the
        concatenated residual would carry two different units).
    """

    models: tuple[NoiseModel, ...]
    lengths: tuple[int, ...]

    #
    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "lengths", tuple(int(n) for n in self.lengths))
        if len(self.models) != len(self.lengths):
            raise ValueError(
                f"got {len(self.models)} noise models for {len(self.lengths)} segments"
            )
        if any(n <= 0 for n in self.lengths):
            raise ValueError(f"segment lengths must be positive; got {self.lengths}")
        _check_segment_mix(self.models)

    #
    @property
    def is_weighted(self) -> bool:
        """Whether the concatenated residual is in likelihood units."""

        return any(noise.is_weighted for noise in self.models)

    #
    def apply(self, data: ArrayLike, model: ArrayLike) -> np.ndarray:
        """Concatenated residual, each segment in its own likelihood units."""

        return apply_segments(self.models, data, model, lengths=self.lengths)

    #
    def jacobian_factor(self, data: ArrayLike, model: ArrayLike) -> np.ndarray:
        """Concatenated chain factor ``dr/dm``, one segment at a time."""

        return jacobian_factor_segments(self.models, data, model, lengths=self.lengths)


#
def apply_segments(
    models: Sequence[NoiseModel],
    data: ArrayLike,
    model: ArrayLike,
    *,
    lengths: Sequence[int],
) -> np.ndarray:
    """
    Residual of a concatenated multi-segment fit.

    Parameters
    ----------
    models : sequence of NoiseModel
        One model per segment, σ already reduced to that segment's view.
    data, model : array_like
        Concatenated (flat) observed data and model prediction.
    lengths : sequence of int
        Points per segment, summing to the length of *data*.

    Returns
    -------
    ndarray
        Flat residual, segment by segment.
    """

    d, m, bounds = _segment_views(models, data, model, lengths)
    return np.concatenate(
        [
            _segment_noise(noise).apply(d[start:stop], m[start:stop])
            for noise, (start, stop) in zip(models, bounds, strict=True)
        ]
    )


#
def jacobian_factor_segments(
    models: Sequence[NoiseModel],
    data: ArrayLike,
    model: ArrayLike,
    *,
    lengths: Sequence[int],
) -> np.ndarray:
    """
    Chain factor ``dr/dm`` of a concatenated multi-segment fit.

    Parameters
    ----------
    models : sequence of NoiseModel
        One model per segment, σ already reduced to that segment's view.
    data, model : array_like
        Concatenated (flat) observed data and model prediction.
    lengths : sequence of int
        Points per segment, summing to the length of *data*.

    Returns
    -------
    ndarray
        Flat factor array, segment by segment.
    """

    d, m, bounds = _segment_views(models, data, model, lengths)
    return np.concatenate(
        [
            np.broadcast_to(
                _segment_noise(noise).jacobian_factor(d[start:stop], m[start:stop]),
                (stop - start,),
            )
            for noise, (start, stop) in zip(models, bounds, strict=True)
        ]
    )


#
def _freeze_sigma(sigma: float | np.ndarray) -> float | np.ndarray:
    """Validate σ and return it as a float or a read-only float array."""

    try:
        arr = np.asarray(sigma, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sigma must be numeric; got {sigma!r}") from exc
    if not bool(np.all(np.isfinite(arr) & (arr > 0))):
        shown = repr(sigma) if arr.ndim == 0 else f"an array of shape {arr.shape}"
        raise ValueError(f"sigma must be finite and positive everywhere; got {shown}")
    if arr.ndim == 0:
        return float(arr)
    frozen = np.array(arr, dtype=float, copy=True)
    frozen.flags.writeable = False
    return frozen


#
def _poisson_residual(d: np.ndarray, m: np.ndarray, scale: float) -> np.ndarray:
    """Signed deviance residual with the model floored at ``EPS_COUNTS``."""

    m_floor = EPS_COUNTS / scale
    m_eval = np.maximum(m, m_floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = np.maximum(kl_div(d, m_eval), 0.0)
        # value and slope at the floor: the linear extension below it keeps
        # the gradient nonzero, so leastsq is pushed back up instead of
        # stalling on a clamped residual with a zero analytic Jacobian
        res_eval = np.sign(d - m_eval) * np.sqrt(2.0 * scale * deviance)
        res_below = res_eval + _poisson_slope(d, m_eval, scale) * (m - m_floor)
        # d = 0 has no singularity: the deviance is 2*scale*m, a plateau at
        # zero for m <= 0, which is exactly what the likelihood says
        res_zero = -np.sqrt(2.0 * scale * np.maximum(m, 0.0))
        # test d == 0 (not d > 0) so NaN data propagate through kl_div
        residual: np.ndarray = np.where(
            d == 0.0, res_zero, np.where(m < m_floor, res_below, res_eval)
        )
        return residual


#
def _poisson_slope(d: np.ndarray, m: np.ndarray, scale: float) -> np.ndarray:
    """``dr/dm`` of the deviance residual for strictly positive *m*."""

    diff = m - d
    with np.errstate(divide="ignore", invalid="ignore"):
        deviance = np.maximum(kl_div(d, m), 0.0)
        near = np.abs(diff) <= LIMIT_REL_TOL * m
        denominator = np.where(near, 1.0, m * np.sqrt(deviance))
        general = -np.sqrt(scale / 2.0) * np.abs(diff) / denominator
        return np.where(near, -np.sqrt(scale / m), general)


#
def _check_segment_mix(models: Sequence[NoiseModel]) -> None:
    """Reject a segment list that mixes 'unknown' with a weighted model."""

    weighted = [noise.is_weighted for noise in models]
    if any(weighted) and not all(weighted):
        raise ValueError(
            "a joint fit cannot mix 'unknown' with a weighted noise model: "
            "the concatenated residual would carry two different units. "
            "Declare a noise model on every file or on none."
        )


#
def _segment_noise(noise: NoiseModel) -> NoiseModel:
    """Return *noise* with a per-point σ flattened to the segment vector."""

    if isinstance(noise.sigma, np.ndarray) and noise.sigma.ndim > 1:
        return NoiseModel(kind=noise.kind, sigma=noise.sigma.reshape(-1))
    return noise


#
def _segment_views(
    models: Sequence[NoiseModel],
    data: ArrayLike,
    model: ArrayLike,
    lengths: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Flatten the concatenated arrays and resolve the segment bounds."""

    _check_segment_mix(models)
    d = np.asarray(data, dtype=float).reshape(-1)
    m = np.asarray(model, dtype=float).reshape(-1)
    if len(models) != len(lengths):
        raise ValueError(f"got {len(models)} noise models for {len(lengths)} segments")
    total = int(sum(int(n) for n in lengths))
    if total != d.size:
        raise ValueError(
            f"segment lengths sum to {total} points but the concatenated "
            f"data has {d.size}"
        )
    bounds: list[tuple[int, int]] = []
    start = 0
    for n_points in lengths:
        stop = start + int(n_points)
        bounds.append((start, stop))
        start = stop
    return d, m, bounds
