"""Tables and messages of the experiment section, in either language.

They turn the numbers of the experiment design, the Monte Carlo power
analysis and the sample check into the tables and text the notebooks
show, so that both notebooks run the same code.
"""
from __future__ import annotations

import math
import warnings

import pandas as pd
from statsmodels.stats.proportion import proportion_effectsize

from marketing_campaign.config import ALPHA, MINIMUM_LIFT_PP, POWER
from marketing_campaign.i18n import say, t


def design_table(
    baseline: float,
    required_per_group: int,
    runtime_days: int,
    lang: str = "en",
    minimum_lift_pp: float = MINIMUM_LIFT_PP,
    alpha: float = ALPHA,
    power: float = POWER,
) -> pd.DataFrame:
    """Describe the A/B experiment: hypotheses, rates, error levels, size.

    Parameters
    ----------
    baseline : float
        Control conversion rate, as a fraction.
    required_per_group : int
        Users needed per group.
    runtime_days : int
        Estimated runtime of the experiment.
    lang : {"en", "pt"}, default="en"
        Language of the table.
    minimum_lift_pp : float, default=2.0
        Smallest lift worth detecting, in percentage points.
    alpha : float, default=0.05
        Type I error.
    power : float, default=0.80
        Desired statistical power.

    Returns
    -------
    pd.DataFrame
        Two columns, parameter and value, with eleven rows.
    """
    treatment = min(baseline + minimum_lift_pp / 100, 0.999)
    cohens_h = proportion_effectsize(baseline, treatment)
    rows = [
        ("design.h0_label", t("design.h0", lang)),
        ("design.h1_label", t("design.h1", lang)),
        ("design.lift_label", f"{minimum_lift_pp:.1f} p.p."),
        ("design.baseline_label", f"{baseline * 100:.2f}%"),
        ("design.treatment_label", f"{treatment * 100:.2f}%"),
        ("design.cohen_label", f"{cohens_h:.4f}"),
        ("design.alpha_label", f"{alpha}"),
        ("design.power_label", f"{power}"),
        ("design.sample_label", t("design.users", lang, n=required_per_group)),
        ("design.runtime_label", t("design.runtime", lang, days=runtime_days)),
        ("design.guardrail_label", t("design.guardrail", lang)),
    ]
    return pd.DataFrame(
        {
            t("table.parameter", lang): [t(key, lang) for key, _ in rows],
            t("table.value", lang): [value for _, value in rows],
        }
    )


def adequacy_message(
    available: int, required_per_group: int, lang: str = "en"
) -> str:
    """Say whether the base is large enough for both groups.

    Parameters
    ----------
    available : int
        Users available.
    required_per_group : int
        Users needed per group.
    lang : {"en", "pt"}, default="en"
        Language of the message.

    Returns
    -------
    str
        The sufficient or insufficient sample message.
    """
    required = required_per_group * 2
    key = "design.adequate" if available >= required else "design.inadequate"
    return t(key, lang, available=available, required=required)


def power_summary_table(
    p_control: float,
    p_treatment: float,
    sample_per_group: int,
    n_simulations: int,
    empirical_power: float,
    lang: str = "en",
    minimum_lift_pp: float = MINIMUM_LIFT_PP,
) -> pd.DataFrame:
    """Summarise the Monte Carlo power simulation.

    Parameters
    ----------
    p_control, p_treatment : float
        Conversion rates of the control and the treatment.
    sample_per_group : int
        Users per group in each simulated experiment.
    n_simulations : int
        Number of simulated experiments.
    empirical_power : float
        Share of simulations that rejected the null hypothesis.
    lang : {"en", "pt"}, default="en"
        Language of the table.
    minimum_lift_pp : float, default=2.0
        Simulated lift, in percentage points.

    Returns
    -------
    pd.DataFrame
        Two columns, parameter and value, with seven rows.
    """
    verdict = "power.adequate" if empirical_power >= 0.80 else (
        "power.inadequate"
    )
    rows = [
        ("power.baseline_label", f"{p_control * 100:.2f}%"),
        ("power.expected_label", f"{p_treatment * 100:.2f}%"),
        ("power.lift_label", f"{minimum_lift_pp:.1f} p.p."),
        ("power.sample_label", f"{sample_per_group:,}"),
        ("power.simulations_label", f"{n_simulations:,}"),
        ("power.power_label", f"{empirical_power:.1%}"),
        ("power.interpretation_label", t(verdict, lang)),
    ]
    return pd.DataFrame(
        {
            t("table.parameter", lang): [t(key, lang) for key, _ in rows],
            t("table.value", lang): [value for _, value in rows],
        }
    )


def adequacy_table(
    variant_counts: pd.Series,
    required_per_group: int,
    empirical_power: float,
    lang: str = "en",
) -> pd.DataFrame:
    """Check the sample of every variant against the power analysis.

    Parameters
    ----------
    variant_counts : pd.Series
        Users per variant.
    required_per_group : int
        Users needed per group (0 when the power analysis was not run).
    empirical_power : float
        Simulated power with the required sample.
    lang : {"en", "pt"}, default="en"
        Language of the table.

    Returns
    -------
    pd.DataFrame
        One row per variant with the users available and required, whether
        they suffice, the estimated power and the gap.
    """
    unknown = t("adequacy.na", lang)
    rows = []
    for variant, count in variant_counts.items():
        users = int(count)
        if required_per_group > 0:
            fraction = min(users / required_per_group, 1.0)
            estimated = empirical_power * fraction
            power_text = (
                unknown if math.isnan(estimated) else f"{estimated:.1%}"
            )
            required = required_per_group
            gap = max(required_per_group - users, 0)
        else:
            power_text, required, gap = unknown, unknown, unknown
        rows.append(
            {
                t("adequacy.variant", lang): str(variant),
                t("adequacy.available", lang): users,
                t("adequacy.required", lang): required,
                t("adequacy.sufficient", lang): t(
                    "adequacy.yes"
                    if users >= required_per_group
                    else "adequacy.no",
                    lang,
                ),
                t("adequacy.power", lang): power_text,
                t("adequacy.gap", lang): gap,
            }
        )
    return pd.DataFrame(rows)


def pretest_decision(
    variant_counts: pd.Series,
    required_per_group: int,
    empirical_power: float,
    runtime_days: int,
    lang: str = "en",
) -> None:
    """Print whether the real sample supports the planned A/B test.

    When the smallest variant is below the requirement a warning is also
    raised, because the real power is lower than the simulated one.

    Parameters
    ----------
    variant_counts : pd.Series
        Users per variant.
    required_per_group : int
        Users needed per group (0 when the power analysis was not run).
    empirical_power : float
        Simulated power with the required sample.
    runtime_days : int
        Estimated days to reach the required sample.
    lang : {"en", "pt"}, default="en"
        Language of the messages.
    """
    smallest = int(variant_counts.min())
    required = required_per_group
    known_power = not math.isnan(empirical_power)

    print("\n" + "=" * 65)
    say("pretest.title", lang)
    print("=" * 65)

    if required == 0:
        say("pretest.not_run", lang)
        say("pretest.run_cell", lang)
    elif smallest >= required:
        say("pretest.sufficient", lang)
        say("pretest.smallest_ok", lang, smallest=smallest, required=required)
        if known_power:
            say("pretest.power", lang, power=empirical_power)
        say("pretest.proceed", lang)
    else:
        say("pretest.insufficient", lang)
        say(
            "pretest.smallest_bad", lang,
            smallest=smallest, required=required, deficit=required - smallest,
        )
        if known_power:
            say(
                "pretest.estimated_power", lang,
                available=empirical_power * (smallest / required),
                target=empirical_power,
            )
        if runtime_days > 0:
            say("pretest.runtime", lang, days=runtime_days)
        print()
        for key in ("options", "option_1", "option_2", "option_3"):
            say(f"pretest.{key}", lang)
        warnings.warn(
            t(
                "pretest.warning", lang,
                smallest=smallest, required=required, power=empirical_power,
            ),
            UserWarning,
            stacklevel=1,
        )
    print("=" * 65)
