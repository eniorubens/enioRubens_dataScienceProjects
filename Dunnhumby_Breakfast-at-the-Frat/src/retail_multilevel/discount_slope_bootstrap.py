"""Parametric null bootstrap for the joint discount-slope likelihood-ratio test.

Each draw keeps the fitted design and simulates a new response, including freshly
drawn random intercepts and residuals under C0. The calibration uses the C0
distribution and therefore inherits its residual independence assumption.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pandas as pd
from scipy.stats import beta

from retail_multilevel.multilevel import LmerFit
from retail_multilevel.rbridge import get_robjects

BOOTSTRAP_COLUMNS = ("replicate", "seed", "lr", "valid", "singular_c0", "singular_c1", "seconds", "message")
LR_NEGATIVE_TOL = 1e-5

_REFIT_FUNCTION = r"""
function(c0, c1, seed, lr_tolerance) {
  checked_refit <- function(model, response, control) {
    notes <- character(0)
    capture <- function(condition) {
      notes <<- c(notes, conditionMessage(condition))
      invokeRestart(if (inherits(condition, "warning")) "muffleWarning" else "muffleMessage")
    }
    fit <- tryCatch(
      withCallingHandlers(lme4::refit(model, newresp = response, control = control),
                          warning = capture, message = capture),
      error = identity
    )
    if (inherits(fit, "error")) {
      return(list(valid = FALSE, lr = NA_real_, singular = NA, message = conditionMessage(fit)))
    }
    codes <- fit@optinfo$conv$opt
    if (length(codes) && any(is.na(codes) | codes != 0L)) {
      notes <- c(notes, paste("optimizer return code", paste(codes, collapse = ",")))
    }
    notes <- unique(c(notes, fit@optinfo$conv$lme4$messages))
    notes <- notes[!grepl("^boundary \\(singular\\) fit:|^singular fit$", notes)]
    log_likelihood <- as.numeric(stats::logLik(fit))
    if (!is.finite(log_likelihood)) notes <- c(notes, "non-finite log likelihood")
    list(valid = !length(notes), lr = log_likelihood, singular = lme4::isSingular(fit),
         message = paste(notes, collapse = " | "))
  }

  make_draw <- function(response, optimizer) {
    control <- lme4::lmerControl(optimizer = optimizer)
    null <- checked_refit(c0, response, control)
    alternative <- checked_refit(c1, response, control)
    statistic <- 2 * (alternative$lr - null$lr)
    notes <- character(0)
    if (nzchar(null$message)) notes <- c(notes, paste0("C0: ", null$message))
    if (nzchar(alternative$message)) notes <- c(notes, paste0("C1: ", alternative$message))
    if (is.finite(statistic) && statistic < -lr_tolerance) {
      notes <- c(notes, sprintf("materially negative likelihood ratio (%.8g)", statistic))
    }
    valid <- null$valid && alternative$valid && is.finite(statistic) && statistic >= -lr_tolerance
    list(lr = if (valid) max(0, statistic) else NA_real_, valid = valid,
         singular_c0 = null$singular, singular_c1 = alternative$singular,
         message = paste(notes, collapse = " | "))
  }

  response <- tryCatch(stats::simulate(c0, nsim = 1L, seed = seed, re.form = NA)[[1L]], error = identity)
  if (inherits(response, "error")) {
    return(list(lr = NA_real_, valid = FALSE, singular_c0 = NA, singular_c1 = NA,
                message = paste("simulation:", conditionMessage(response))))
  }
  first <- make_draw(response, "bobyqa")
  if (first$valid) return(first)
  second <- make_draw(response, "nloptwrap")
  first_note <- paste("bobyqa:", first$message)
  retry_note <- if (second$valid) "nloptwrap: recovered" else paste("nloptwrap:", second$message)
  second$message <- paste(first_note, retry_note, sep = " | ")
  second
}
"""


def bootstrap(
    c0: LmerFit,
    c1: LmerFit,
    seeds: list[tuple[int, int]],
    *,
    on_result: Callable[[dict], None] | None = None,
    calc_derivs: bool = True,
) -> pd.DataFrame:
    """Simulate the null LR distribution from two ML fits on the same design.

    Parameters
    ----------
    c0, c1 : LmerFit
        Nested null and alternative models estimated by maximum likelihood.
    seeds : list of tuple
        Pairs of unique replicate index and integer R seed, in execution order.
    on_result : callable, optional
        Receive a serialisable row after each completed attempt.
    calc_derivs : bool, default True
        Request derivative-based convergence checks in the refits. If disabled,
        optimizer return codes, warnings and log-likelihood checks still apply.

    Returns
    -------
    pandas.DataFrame
        One row per attempted replicate. A singular simulation fit can be valid
        because a zero random-effect variance is expected under the null.
    """
    if not seeds:
        return pd.DataFrame(columns=BOOTSTRAP_COLUMNS)
    if c0.reml or c1.reml or c0.nobs != c1.nobs or c1.npar != c0.npar + 2:
        raise ValueError("bootstrap requires aligned ML fits differing by two parameters")
    ids, r_seeds = zip(*seeds, strict=True)
    if len(set(ids)) != len(ids) or len(set(r_seeds)) != len(r_seeds):
        raise ValueError("replicate indices and R seeds must be unique")
    if any(int(seed) != seed or not 0 <= seed < 2**31 for seed in r_seeds):
        raise ValueError("R seeds must be nonnegative 32-bit signed integers")

    ro = get_robjects()
    prepare = ro.r(
        """function(c0, c1, calc_derivs) {
          if (lme4::isREML(c0) || lme4::isREML(c1)) stop("models must be fitted by ML")
          if (lme4::isSingular(c0)) stop("observed C0 is singular")
          base_frame <- stats::model.frame(c0)
          full_frame <- stats::model.frame(c1)
          same_rows <- identical(rownames(base_frame), rownames(full_frame)) &&
                       setequal(names(base_frame), names(full_frame)) &&
                       all(vapply(names(base_frame), function(name) {
                         identical(base_frame[[name]], full_frame[[name]])
                       }, logical(1)))
          if (!same_rows ||
              !identical(lme4::getME(c0, "X"), lme4::getME(c1, "X"))) {
            stop("observed models must use the same rows and fixed design")
          }
          # refit inherits the derivative policy from the template fit's optinfo.
          # An empty list requests derivatives without attaching stale values.
          derivative_template <- if (calc_derivs) list() else NULL
          c0@optinfo$derivs <- derivative_template
          c1@optinfo$derivs <- derivative_template
          list(c0 = c0, c1 = c1)
        }"""
    )
    templates = prepare(c0.r_model, c1.r_model, calc_derivs)
    null, alternative = templates.rx2("c0"), templates.rx2("c1")
    fit_draw = ro.r(_REFIT_FUNCTION)
    rows = []
    for index, seed in seeds:
        start = time.perf_counter()
        result = fit_draw(null, alternative, int(seed), LR_NEGATIVE_TOL)
        valid = bool(result.rx2("valid")[0])
        row = {
            "replicate": int(index),
            "seed": int(seed),
            "lr": float(result.rx2("lr")[0]) if valid else None,
            "valid": valid,
            "singular_c0": _optional_flag(result.rx2("singular_c0")[0], ro.NA_Logical),
            "singular_c1": _optional_flag(result.rx2("singular_c1")[0], ro.NA_Logical),
            "seconds": time.perf_counter() - start,
            "message": str(result.rx2("message")[0]),
        }
        rows.append(row)
        if on_result is not None:
            on_result(row)
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def _optional_flag(value, missing) -> bool | None:
    if isinstance(value, type(missing)):
        return None
    return bool(value)


def bootstrap_summary(observed_lr: float, draws: pd.DataFrame, planned: int) -> dict:
    """Summarise Monte Carlo precision and publish a p-value after all valid draws.

    The Clopper–Pearson interval estimates the simulation exceedance fraction;
    it quantifies Monte Carlo error under the fitted null distribution.
    """
    if not math.isfinite(observed_lr) or observed_lr < 0:
        raise ValueError("observed LR must be finite and nonnegative")
    attempted = len(draws)
    if planned < 0 or attempted > planned:
        raise ValueError("attempted replicas must not exceed the planned count")
    if attempted and draws["replicate"].duplicated().any():
        raise ValueError("replicate indices must be unique")
    simulated = draws.loc[draws["valid"].eq(True), "lr"] if attempted else pd.Series(dtype=float)
    if not simulated.map(math.isfinite).all() or (simulated < 0).any():
        raise ValueError("valid simulated LR values must be finite and nonnegative")
    valid = len(simulated)
    exceedances = int(simulated.ge(observed_lr).sum())
    complete = planned > 0 and attempted == valid == planned
    lower = upper = None
    if valid:
        lower = float(beta.ppf(0.025, exceedances, valid - exceedances + 1)) if exceedances else 0.0
        upper = float(beta.ppf(0.975, exceedances + 1, valid - exceedances)) if exceedances < valid else 1.0

    if complete:
        status = "complete"
    elif attempted > valid:
        status = "invalid_replicates"
    else:
        status = "pilot" if attempted else "observed_only"
    return {
        "status": status,
        "observed_lr": float(observed_lr),
        "planned": int(planned),
        "attempted": attempted,
        "valid": valid,
        "exceedances": exceedances,
        "complete": complete,
        "p_value": (exceedances + 1) / (valid + 1) if complete else None,
        "null_tail_ci_low": lower,
        "null_tail_ci_high": upper,
    }
