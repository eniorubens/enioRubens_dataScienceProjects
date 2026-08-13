"""Shims de compatibilidade entre versões de dependências do stack causal."""


def patch_sklearn_matplotlib_support():
    """Restaura `sklearn.utils.check_matplotlib_support` para o scikit-uplift.

    scikit-uplift 0.5.1 importa essa função de `sklearn.utils`, mas ela não é
    reexportada ali em scikit-learn >= 1.5 (verificado empiricamente em 1.5.2
    e 1.6.1 — não é específico de 1.6+). Sem o shim, `from sklift.viz import
    ...` falha com ImportError.
    """
    import sklearn.utils as sklearn_utils

    if not hasattr(sklearn_utils, "check_matplotlib_support"):
        def _check_matplotlib_support(*args, **kwargs):
            return None

        sklearn_utils.check_matplotlib_support = _check_matplotlib_support
