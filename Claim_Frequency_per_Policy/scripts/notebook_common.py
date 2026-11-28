"""Utilitarios do construtor de notebooks (formatacao e celulas)."""

from __future__ import annotations

import nbformat

PT, EN = "pt_br", "en_us"
MATPLOTLIB_INLINE_MAGIC = "%matplotlib inline"
RETINA_SETUP = (
    "from matplotlib_inline.backend_inline import set_matplotlib_formats\n"
    'set_matplotlib_formats("retina")'
)
REFERENCES = {
    "cameron1990": (
        "Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests "
        "for overdispersion in the Poisson model. *Journal of "
        "Econometrics, 46*(3), 347–364. "
        "https://doi.org/10.1016/0304-4076(90)90014-K"),
    "cameron2013": (
        "Cameron, A. C., & Trivedi, P. K. (2013). *Regression Analysis of "
        "Count Data* (2nd ed.). Cambridge University Press. "
        "https://doi.org/10.1017/CBO9781139013567"),
    "dutang2026": (
        "Dutang, C., & Charpentier, A. (2026). *CASdatasets: Insurance "
        "datasets* (R package version 1.2-1; dicionário de freMTPL2freq em "
        "https://dutangc.github.io/CASdatasets/reference/freMTPL.html). "
        "https://doi.org/10.57745/P0KHAG"),
    "dwork2015": (
        "Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & "
        "Roth, A. (2015). [Generalization in adaptive data analysis and "
        "holdout reuse](https://proceedings.neurips.cc/paper_files/paper/2015/"
        "hash/bad5f33780c42f2588878a9d07405083-Abstract.html). "
        "*Advances in Neural Information Processing Systems, 28*."),
    "favero2017": (
        "Fávero, L. P., & Belfiore, P. (2017). *Manual de Análise de "
        "Dados: Estatística e Modelagem Multivariada com Excel, SPSS e "
        "Stata*. Rio de Janeiro: Elsevier. ISBN 978-85-352-7087-7."),
    "wickham2017": (
        "Wickham, H., & Grolemund, G. (2017). "
        "[*R for Data Science*](https://r4ds.had.co.nz/model-intro.html) "
        "(1st ed., ch. 22). O’Reilly Media."),
    "wilson2015": (
        "Wilson, P. (2015). The misuse of the Vuong test for non-nested "
        "models to test for zero-inflation. *Economics Letters, 127*, "
        "51–53. https://doi.org/10.1016/j.econlet.2014.12.029"),
    "noll2020": (
        "Noll, A., Salzmann, R., & Wüthrich, M. V. (2020). Case study: "
        "French motor third-party liability claims (revised version; "
        "first posted in 2018). *SSRN*. "
        "https://doi.org/10.2139/ssrn.3164764"),
}


def num(value: float, decimals: int = 2, lang: str = PT) -> str:
    """Formate um numero com separadores do idioma."""
    text = f"{value:,.{decimals}f}"
    if lang == PT:
        text = text.replace(",", "X").replace(".", ",").replace("X", ".")
    return text


def pct(value: float, decimals: int = 1, lang: str = PT) -> str:
    """Formate uma proporcao como percentual."""
    return num(100 * value, decimals, lang) + "%"


class Builder:
    """Acumula celulas em dois idiomas com codigo compartilhado."""

    def __init__(self) -> None:
        """Inicie as listas de celulas."""
        self.cells: list[tuple] = []

    def md(self, pt: str, en: str) -> None:
        """Adicione uma celula markdown bilingue."""
        self.cells.append(("md", pt.strip(), en.strip()))

    def code(self, source: str) -> None:
        """Adicione uma celula de codigo igual nos dois idiomas."""
        self.cells.append(("code", source.strip("\n")))

    def build(self, lang: str, references: list[str]) -> nbformat.NotebookNode:
        """Monte o notebook de um idioma, com a secao final de referencias."""
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"display_name": "Python 3",
                                     "language": "python",
                                     "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        for cell in self.cells:
            if cell[0] == "md":
                text = cell[1] if lang == PT else cell[2]
                nb.cells.append(nbformat.v4.new_markdown_cell(text))
            else:
                source = cell[1].replace("__LANG__", lang)
                if (MATPLOTLIB_INLINE_MAGIC in source
                        and RETINA_SETUP not in source):
                    source = source.replace(
                        MATPLOTLIB_INLINE_MAGIC,
                        f"{MATPLOTLIB_INLINE_MAGIC}\n{RETINA_SETUP}",
                        1,
                    )
                nb.cells.append(nbformat.v4.new_code_cell(source))
        title = "Referências" if lang == PT else "References"
        body = "\n".join(f"- {REFERENCES[key]}" for key in sorted(
            references, key=lambda k: REFERENCES[k]))
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{body}"))
        return nb
