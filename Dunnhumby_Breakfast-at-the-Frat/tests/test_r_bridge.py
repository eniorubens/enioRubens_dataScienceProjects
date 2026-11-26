"""Smoke test: lme4 reachable from Python and recovers a known crossed random-effects variance."""

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.r

pytest.importorskip("rpy2")

from retail_multilevel.rbridge import env_r_home, get_robjects  # noqa: E402

ro = get_robjects()


def test_rpy2_uses_the_env_r_not_a_system_r():
    import os

    r_home = ro.r("normalizePath(R.home())")[0]
    assert os.path.samefile(r_home, env_r_home())


def test_lmer_crossed_random_intercepts_recover_variances():
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    lme4 = importr("lme4")
    rng = np.random.default_rng(0)
    n_store, n_prod, n_week = 30, 20, 10
    u_store = rng.normal(0, 0.5, n_store)
    u_prod = rng.normal(0, 1.0, n_prod)
    rows = [
        (s, p, 1.0 + u_store[s] + u_prod[p] + rng.normal(0, 0.3))
        for s in range(n_store)
        for p in range(n_prod)
        for _ in range(n_week)
    ]
    df = pd.DataFrame(rows, columns=["store", "prod", "y"]).astype({"store": str, "prod": str})

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.get_conversion().py2rpy(df)
    fit = lme4.lmer(ro.Formula("y ~ 1 + (1 | store) + (1 | prod)"), data=r_df, REML=True)
    vc = ro.r("function(m) { v <- as.data.frame(VarCorr(m)); setNames(v$sdcor, v$grp) }")(fit)
    sd = dict(zip(vc.names, list(vc), strict=True))

    assert sd["store"] == pytest.approx(0.5, abs=0.2)
    assert sd["prod"] == pytest.approx(1.0, abs=0.4)
    assert sd["Residual"] == pytest.approx(0.3, abs=0.03)
