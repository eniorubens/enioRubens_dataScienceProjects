"""Vetted bibliography of the project and its rendering as a Markdown references section.

The single source is ``docs/REFERENCIAS.md``; this module mirrors it (keys, full APA entries and
verification status) so the notebook builder can cite and list references consistently.
"""

from __future__ import annotations

REFERENCES: dict[str, dict[str, str]] = {
    "akiba2019": {
        "cite_pt": "Akiba et al., 2019",
        "cite_en": "Akiba et al., 2019",
        "entry": (
            "Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation "
            "hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International "
            "Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). "
            "https://doi.org/10.1145/3292500.3330701"
        ),
        "status": "doi",
    },
    "baayen2008": {
        "cite_pt": "Baayen et al., 2008",
        "cite_en": "Baayen et al., 2008",
        "entry": (
            "Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed "
            "random effects for subjects and items. *Journal of Memory and Language, 59*(4), 390–412. "
            "https://doi.org/10.1016/j.jml.2007.12.005"
        ),
        "status": "doi",
    },
    "bates2015": {
        "cite_pt": "Bates et al., 2015",
        "cite_en": "Bates et al., 2015",
        "entry": (
            "Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting linear mixed-effects "
            "models using lme4. *Journal of Statistical Software, 67*(1), 1–48. "
            "https://doi.org/10.18637/jss.v067.i01"
        ),
        "status": "doi",
    },
    "bergstra2011": {
        "cite_pt": "Bergstra et al., 2011",
        "cite_en": "Bergstra et al., 2011",
        "entry": (
            "Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter "
            "optimization. In *Advances in Neural Information Processing Systems 24*."
        ),
        "status": "page",
    },
    "bijmolt2005": {
        "cite_pt": "Bijmolt et al., 2005",
        "cite_en": "Bijmolt et al., 2005",
        "entry": (
            "Bijmolt, T. H. A., van Heerde, H. J., & Pieters, R. G. M. (2005). New empirical "
            "generalizations on the determinants of price elasticity. *Journal of Marketing Research, "
            "42*(2), 141–156. https://doi.org/10.1509/jmkr.42.2.141.62296"
        ),
        "status": "doi",
    },
    "blattberg1989": {
        "cite_pt": "Blattberg & Neslin, 1989",
        "cite_en": "Blattberg & Neslin, 1989",
        "entry": (
            "Blattberg, R. C., & Neslin, S. A. (1989). Sales promotion: The long and the short of it. "
            "*Marketing Letters, 1*(1), 81–97. https://doi.org/10.1007/BF00436151"
        ),
        "status": "doi",
    },
    "cameron2011": {
        "cite_pt": "Cameron et al., 2011",
        "cite_en": "Cameron et al., 2011",
        "entry": (
            "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway "
            "clustering. *Journal of Business & Economic Statistics, 29*(2), 238–249. "
            "https://doi.org/10.1198/jbes.2010.07136"
        ),
        "status": "doi",
    },
    "cameron2015": {
        "cite_pt": "Cameron & Miller, 2015",
        "cite_en": "Cameron & Miller, 2015",
        "entry": (
            "Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. "
            "*Journal of Human Resources, 50*(2), 317–372. https://doi.org/10.3368/jhr.50.2.317"
        ),
        "status": "doi",
    },
    "gelman2007": {
        "cite_pt": "Gelman & Hill, 2007",
        "cite_en": "Gelman & Hill, 2007",
        "entry": (
            "Gelman, A., & Hill, J. (2007). *Data analysis using regression and multilevel/hierarchical "
            "models*. Cambridge University Press. https://doi.org/10.1017/CBO9780511790942"
        ),
        "status": "doi",
    },
    "halekoh2014": {
        "cite_pt": "Halekoh & Højsgaard, 2014",
        "cite_en": "Halekoh & Højsgaard, 2014",
        "entry": (
            "Halekoh, U., & Højsgaard, S. (2014). A Kenward-Roger approximation and parametric "
            "bootstrap methods for tests in linear mixed models – The R package pbkrtest. "
            "*Journal of Statistical Software, 59*(9), 1–32. https://doi.org/10.18637/jss.v059.i09"
        ),
        "status": "page",
    },
    "henderson1975": {
        "cite_pt": "Henderson, 1975",
        "cite_en": "Henderson, 1975",
        "entry": (
            "Henderson, C. R. (1975). Best linear unbiased estimation and prediction under a selection "
            "model. *Biometrics, 31*(2), 423–447. https://doi.org/10.2307/2529430"
        ),
        "status": "doi",
    },
    "holm1979": {
        "cite_pt": "Holm, 1979",
        "cite_en": "Holm, 1979",
        "entry": (
            "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian "
            "Journal of Statistics, 6*(2), 65–70."
        ),
        "status": "partial",
    },
    "hyndman2021": {
        "cite_pt": "Hyndman & Athanasopoulos, 2021",
        "cite_en": "Hyndman & Athanasopoulos, 2021",
        "entry": (
            "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd "
            "ed.). OTexts. https://otexts.com/fpp3/"
        ),
        "status": "page",
    },
    "mundlak1978": {
        "cite_pt": "Mundlak, 1978",
        "cite_en": "Mundlak, 1978",
        "entry": (
            "Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica, "
            "46*(1), 69–85. https://doi.org/10.2307/1913646"
        ),
        "status": "doi",
    },
    "nosek2018": {
        "cite_pt": "Nosek et al., 2018",
        "cite_en": "Nosek et al., 2018",
        "entry": (
            "Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration "
            "revolution. *Proceedings of the National Academy of Sciences, 115*(11), 2600–2606. "
            "https://doi.org/10.1073/pnas.1708274114"
        ),
        "status": "doi",
    },
    "ott2014": {
        "cite_pt": "Ott, 2014",
        "cite_en": "Ott, 2014",
        "entry": (
            "Ott, P. K. (2014). COVTEST: A SAS macro for hypothesis testing in linear mixed effects "
            "models via parametric bootstrap. In *SAS Global Forum 2014 Proceedings* (Paper 1558-2014). "
            "https://support.sas.com/resources/papers/proceedings14/1558-2014.pdf"
        ),
        "status": "page",
    },
    "pinheiro2000": {
        "cite_pt": "Pinheiro & Bates, 2000",
        "cite_en": "Pinheiro & Bates, 2000",
        "entry": (
            "Pinheiro, J. C., & Bates, D. M. (2000). *Mixed-effects models in S and S-PLUS*. Springer. "
            "https://doi.org/10.1007/b98882"
        ),
        "status": "doi",
    },
    "raudenbush2002": {
        "cite_pt": "Raudenbush & Bryk, 2002",
        "cite_en": "Raudenbush & Bryk, 2002",
        "entry": (
            "Raudenbush, S. W., & Bryk, A. S. (2002). *Hierarchical linear models: Applications and data "
            "analysis methods* (2nd ed.). Sage."
        ),
        "status": "partial",
    },
    "robinson1991": {
        "cite_pt": "Robinson, 1991",
        "cite_en": "Robinson, 1991",
        "entry": (
            "Robinson, G. K. (1991). That BLUP is a good thing: The estimation of random effects. "
            "*Statistical Science, 6*(1), 15–32. https://doi.org/10.1214/ss/1177011926"
        ),
        "status": "doi",
    },
    "seabold2010": {
        "cite_pt": "Seabold & Perktold, 2010",
        "cite_en": "Seabold & Perktold, 2010",
        "entry": (
            "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with "
            "Python. In *Proceedings of the 9th Python in Science Conference* (pp. 92–96). "
            "https://doi.org/10.25080/Majora-92bf1922-011"
        ),
        "status": "doi",
    },
    "self1987": {
        "cite_pt": "Self & Liang, 1987",
        "cite_en": "Self & Liang, 1987",
        "entry": (
            "Self, S. G., & Liang, K.-Y. (1987). Asymptotic properties of maximum likelihood estimators "
            "and likelihood ratio tests under nonstandard conditions. *Journal of the American "
            "Statistical Association, 82*(398), 605–610. https://doi.org/10.1080/01621459.1987.10478472"
        ),
        "status": "doi",
    },
    "sigrist2022": {
        "cite_pt": "Sigrist, 2022",
        "cite_en": "Sigrist, 2022",
        "entry": ("Sigrist, F. (2022). Gaussian process boosting. *Journal of Machine Learning Research, 23*, 1–46."),
        "status": "page",
    },
    "stram1994": {
        "cite_pt": "Stram & Lee, 1994",
        "cite_en": "Stram & Lee, 1994",
        "entry": (
            "Stram, D. O., & Lee, J. W. (1994). Variance components testing in the longitudinal mixed "
            "effects model. *Biometrics, 50*(4), 1171–1177. https://doi.org/10.2307/2533455"
        ),
        "status": "doi",
    },
    "tashman2000": {
        "cite_pt": "Tashman, 2000",
        "cite_en": "Tashman, 2000",
        "entry": (
            "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. "
            "*International Journal of Forecasting, 16*(4), 437–450. "
            "https://doi.org/10.1016/S0169-2070(00)00065-0"
        ),
        "status": "doi",
    },
    "tellis1988": {
        "cite_pt": "Tellis, 1988",
        "cite_en": "Tellis, 1988",
        "entry": (
            "Tellis, G. J. (1988). The price elasticity of selective demand: A meta-analysis of "
            "econometric models of sales. *Journal of Marketing Research, 25*(4), 331–341. "
            "https://doi.org/10.1177/002224378802500401"
        ),
        "status": "doi",
    },
    "villasboas1999": {
        "cite_pt": "Villas-Boas & Winer, 1999",
        "cite_en": "Villas-Boas & Winer, 1999",
        "entry": (
            "Villas-Boas, J. M., & Winer, R. S. (1999). Endogeneity in brand choice models. *Management "
            "Science, 45*(10), 1324–1338. https://doi.org/10.1287/mnsc.45.10.1324"
        ),
        "status": "doi",
    },
    "weston2019": {
        "cite_pt": "Weston et al., 2019",
        "cite_en": "Weston et al., 2019",
        "entry": (
            "Weston, S. J., Ritchie, S. J., Rohrer, J. M., & Przybylski, A. K. (2019). Recommendations "
            "for increasing the transparency of analysis of preexisting data sets. *Advances in Methods "
            "and Practices in Psychological Science, 2*(3), 214–227. "
            "https://doi.org/10.1177/2515245919848684"
        ),
        "status": "doi",
    },
    "wooldridge2010": {
        "cite_pt": "Wooldridge, 2010",
        "cite_en": "Wooldridge, 2010",
        "entry": (
            "Wooldridge, J. M. (2010). *Econometric analysis of cross section and panel data* (2nd ed.). MIT Press."
        ),
        "status": "partial",
    },
}

HEADINGS = {"pt": "## Referências", "en": "## References"}
_NOTE_DEFAULT = {
    "pt": "(verificação parcial: ano da edição não conferido em fonte primária)",
    "en": "(partial verification: edition year not checked against a primary source)",
}
_NOTE_OVERRIDES = {
    "holm1979": {
        "pt": "(verificação parcial: fonte não acessível para conferência automática)",
        "en": "(partial verification: source not accessible for automated checking)",
    },
}


def cite(key: str, lang: str) -> str:
    """Inline citation text (without parentheses) of ``key`` in ``lang`` ('pt' or 'en')."""
    return REFERENCES[key][f"cite_{lang}"]


def render_references(keys: list[str] | set[str] | tuple[str, ...], lang: str) -> str:
    """Markdown references section: one paragraph per entry, sorted by first author, no list markers."""
    if lang not in HEADINGS:
        raise ValueError(f"unknown language {lang!r}")
    paragraphs = [HEADINGS[lang]]
    for key in sorted(set(keys), key=lambda k: REFERENCES[k]["entry"].casefold()):
        ref = REFERENCES[key]
        text = ref["entry"]
        if ref["status"] == "partial":
            text += " " + _NOTE_OVERRIDES.get(key, _NOTE_DEFAULT)[lang]
        paragraphs.append(text)
    return "\n\n".join(paragraphs)
