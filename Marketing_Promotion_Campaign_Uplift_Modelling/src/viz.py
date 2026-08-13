"""Funções de plotagem para S1-S3 (estilo, distribuições, balance, ATE, splits).

Textos exibidos nas figuras (títulos, labels, eixos) são sempre recebidos
como parâmetro — nunca hardcoded aqui — para que o mesmo código sirva a
uma futura edição do notebook em outro idioma.
"""
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def apply_plot_style():
    """Aplica o estilo de gráficos padrão do portfólio (rcParams, sem texto)."""
    sns.set_theme(style="ticks", context="notebook")
    mpl.rcParams["figure.figsize"] = (16, 6)
    mpl.rcParams["figure.titlesize"] = 16
    mpl.rcParams["figure.titleweight"] = "bold"
    mpl.rcParams["axes.titlesize"] = 16
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.titlepad"] = 16
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["axes.labelweight"] = "bold"
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["axes.grid.axis"] = "y"
    mpl.rcParams["ytick.left"] = False
    mpl.rcParams["axes.formatter.useoffset"] = True
    mpl.rcParams["legend.facecolor"] = "white"
    mpl.rcParams["legend.title_fontsize"] = 14
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["legend.framealpha"] = 1
    mpl.rcParams["legend.fancybox"] = True
    mpl.rcParams["legend.edgecolor"] = "black"
    mpl.rcParams["legend.borderpad"] = 0.8
    mpl.rcParams["lines.linewidth"] = 3
    mpl.rcParams["lines.markersize"] = 10
    try:
        get_ipython().run_line_magic("config", "InlineBackend.figure_format = 'retina'")
    except NameError:
        pass


def add_chart_header(fig, title, subtitle=None, x=0.015, y=0.98,
                      title_offset=0.055, subtitle_offset=0.095,
                      title_size=16, subtitle_size=12):
    """Cabeçalho editorial: título em negrito+itálico, subtítulo em itálico, ambos à esquerda.

    `subtitle` funciona como comentário analítico curto (o que o gráfico
    mostra de relevante), não como descrição genérica — a leitura completa
    continua na célula markdown logo abaixo. Opera em qualquer `fig`
    (inclusive de bibliotecas externas, ex.: `UpliftCurveDisplay.figure_`).
    """
    fig.text(x=x, y=y - title_offset, s=title, ha="left", va="bottom",
              fontsize=title_size, fontweight="bold", style="italic")
    if subtitle:
        fig.text(x=x, y=y - subtitle_offset, s=subtitle, ha="left", va="bottom",
                  fontsize=subtitle_size, style="italic")


def add_chart_footer(fig, text=None, data_source=None, method=None,
                      x=0.015, y=0.01, fontsize=9):
    """Rodapé opcional (fonte/método/nota extra) — exceção, só quando o subtítulo não basta."""
    parts = [p for p in (
        f"Fonte: {data_source}" if data_source else None,
        f"Método: {method}" if method else None,
        text,
    ) if p]
    if parts:
        fig.text(x=x, y=y, s="  |  ".join(parts), ha="left", va="bottom",
                  fontsize=fontsize, style="italic", color="#888888")


_PANEL_TITLE_KWARGS = {"fontsize": 13, "fontweight": "normal"}


def plot_univariate_continuous(df, cont_vars, titles, title, subtitle=None, bins=40, figsize=(12, 4)):
    """Histogramas lado a lado para variáveis contínuas."""
    fig, axes = plt.subplots(1, len(cont_vars), figsize=figsize)
    for ax, v, panel_title in zip(np.atleast_1d(axes), cont_vars, titles):
        sns.histplot(df[v], bins=bins, kde=False, ax=ax)
        ax.set_title(panel_title, **_PANEL_TITLE_KWARGS)
    fig.subplots_adjust(top=0.72)
    add_chart_header(fig, title, subtitle)
    return fig, axes


def plot_univariate_categorical(df, bin_vars, cat_vars, title, subtitle=None, titles=None, rotate_vars=()):
    """Barplots de contagem em grade 2x3 para variáveis binárias e categóricas.

    Variáveis em `bin_vars` são ordenadas por índice (0/1); as demais, por
    frequência (ordem padrão de `value_counts`).
    """
    all_vars = list(bin_vars) + list(cat_vars)
    titles = titles if titles is not None else all_vars
    ncols = 3
    nrows = int(np.ceil(len(all_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    for ax, v, panel_title in zip(np.atleast_1d(axes).flat, all_vars, titles):
        counts = df[v].value_counts().sort_index() if v in bin_vars else df[v].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        ax.set_title(panel_title, **_PANEL_TITLE_KWARGS)
        ax.set_xlabel("")
        ax.set_ylabel("count")
        if v in rotate_vars:
            ax.tick_params(axis="x", rotation=30)
    # Reserva ~1.4in absolutos para o cabeçalho, independente da altura total
    # (que cresce com nrows) — uma fração fixa sobrava pouco espaço com 2+ linhas.
    fig.subplots_adjust(top=1 - (1.4 / (3.5 * nrows)))
    add_chart_header(fig, title, subtitle)
    return fig, axes


def plot_love_plot(smd_tables, title, xlabel, subtitle=None, threshold_ok=0.05, threshold_strong=0.10, figsize=(9, 7)):
    """Love plot: dispersão de SMD por variável, uma série por comparação.

    `smd_tables`: dict {rótulo_da_série: DataFrame com colunas 'variable' e
    'smd'}. Todas as séries devem compartilhar a mesma lista/ordem de
    variáveis (garantido por `build_smd_table` quando chamado sobre o mesmo
    `df`).
    """
    fig, ax = plt.subplots(figsize=figsize)
    variables = next(iter(smd_tables.values()))["variable"]
    y = np.arange(len(variables))
    markers = ["o", "s", "^", "D"]

    for (label, table), marker in zip(smd_tables.items(), markers):
        ax.scatter(table["smd"], y, label=label, s=60, marker=marker)

    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(threshold_strong, color="red", ls="--", lw=0.8, label=f"Threshold |SMD|={threshold_strong}")
    ax.axvline(-threshold_strong, color="red", ls="--", lw=0.8)
    ax.axvline(threshold_ok, color="orange", ls=":", lw=0.8, label=f"Threshold |SMD|={threshold_ok}")
    ax.axvline(-threshold_ok, color="orange", ls=":", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(variables)
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.subplots_adjust(top=0.80)
    add_chart_header(fig, title, subtitle)
    return fig, ax


def plot_outcomes_by_arm(df, treatment_col, arms, outcomes, ylabels, titles, title, subtitle=None, figsize=(15, 4)):
    """Barplot da média de cada outcome por braço, com rótulo de valor sobre a barra."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=figsize)
    for ax, outcome, ylabel, panel_title in zip(np.atleast_1d(axes), outcomes, ylabels, titles):
        means = df.groupby(treatment_col, observed=True)[outcome].mean().reindex(arms)
        sns.barplot(x=means.index, y=means.values, ax=ax)
        ax.set_title(panel_title, **_PANEL_TITLE_KWARGS)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        for i, v in enumerate(means.values):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.subplots_adjust(top=0.72)
    add_chart_header(fig, title, subtitle)
    return fig, axes


def plot_ate_forest(ate_df, outcomes, xlabel_prefix, title_prefix, title, subtitle=None, figsize=(15, 3.5)):
    """Forest plot do ATE (com IC) por braço tratado, um painel por outcome."""
    fig, axes = plt.subplots(1, len(outcomes), figsize=figsize)
    for ax, outcome in zip(np.atleast_1d(axes), outcomes):
        sub = ate_df[ate_df["outcome"] == outcome]
        y = np.arange(len(sub))
        ax.errorbar(
            sub["ate"], y,
            xerr=[sub["ate"] - sub["ci_low"], sub["ci_high"] - sub["ate"]],
            fmt="o", capsize=4, markersize=8,
        )
        ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["treatment"])
        ax.set_xlabel(f"{xlabel_prefix}{outcome}")
        ax.set_title(f"{title_prefix}{outcome}", **_PANEL_TITLE_KWARGS)
        ax.grid(axis="x", alpha=0.3)
    fig.subplots_adjust(top=0.68)
    add_chart_header(fig, title, subtitle)
    return fig, axes


def plot_split_overview(train_df, val_df, n_test, treatment_col, arms, outcome_col, labels,
                         title, subtitle=None, figsize=(15, 3.2)):
    """Visualiza a composição do split: partição, braço de tratamento e outcome.

    Treino e validação são ordenados por (braço, outcome) só para tornar
    visíveis, em blocos, as proporções relativas de cada combinação — a
    ordem não tem significado temporal. O teste selado aparece apenas como
    bloco proporcional ao seu tamanho: nenhuma composição de braço/outcome é
    revelada, por disciplina do protocolo (o teste só é aberto em S6).

    `labels`: dict com chaves 'partition_row', 'arm_row', 'outcome_row',
    'train', 'val', 'sealed', 'outcome_neg' (legenda do outcome = 0) e
    'outcome_pos' (legenda do outcome = 1).
    """
    train_sorted = train_df.sort_values([treatment_col, outcome_col])
    val_sorted = val_df.sort_values([treatment_col, outcome_col])
    n_train, n_val = len(train_sorted), len(val_sorted)
    n_shown = n_train + n_val
    n_total = n_shown + n_test

    arm_codes = {arm: i for i, arm in enumerate(arms)}
    combined = pd.concat([train_sorted, val_sorted])

    partition_row = np.array([0] * n_train + [1] * n_val + [2] * n_test, dtype=float)
    arm_row = np.full(n_total, np.nan)
    arm_row[:n_shown] = combined[treatment_col].map(arm_codes).to_numpy()
    outcome_row = np.full(n_total, np.nan)
    outcome_row[:n_shown] = combined[outcome_col].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    partition_colors = ["#4C72B0", "#DD8452", "#B0B0B0"]
    ax.imshow(
        partition_row[np.newaxis, :], aspect="auto", extent=(0, n_total, 2.5, 3.5),
        cmap=mpl.colors.ListedColormap(partition_colors), vmin=0, vmax=2,
    )

    arm_colors = ["#55A868", "#C44E52", "#8172B2", "#CCB974"][: len(arms)]
    arm_cmap = mpl.colors.ListedColormap(arm_colors).with_extremes(bad="#E0E0E0")
    ax.imshow(
        np.ma.masked_invalid(arm_row)[np.newaxis, :], aspect="auto", extent=(0, n_total, 1.5, 2.5),
        cmap=arm_cmap, vmin=0, vmax=len(arms) - 1,
    )

    outcome_colors = ["#DA8BC3", "#64B5CD"]
    outcome_cmap = mpl.colors.ListedColormap(outcome_colors).with_extremes(bad="#E0E0E0")
    ax.imshow(
        np.ma.masked_invalid(outcome_row)[np.newaxis, :], aspect="auto", extent=(0, n_total, 0.5, 1.5),
        cmap=outcome_cmap, vmin=0, vmax=1,
    )

    ax.set_xlim(0, n_total)
    ax.set_ylim(0.5, 3.5)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels([labels["outcome_row"], labels["arm_row"], labels["partition_row"]])
    ax.set_xlabel("Sample index")

    partition_handles = [
        mpatches.Patch(color=partition_colors[0], label=labels["train"]),
        mpatches.Patch(color=partition_colors[1], label=labels["val"]),
        mpatches.Patch(color=partition_colors[2], label=labels["sealed"]),
    ]
    arm_handles = [mpatches.Patch(color=c, label=a) for c, a in zip(arm_colors, arms)]
    outcome_handles = [
        mpatches.Patch(color=outcome_colors[0], label=labels["outcome_neg"]),
        mpatches.Patch(color=outcome_colors[1], label=labels["outcome_pos"]),
    ]

    # fig.legend() (não ax.legend com bbox_to_anchor > 1) + subplots_adjust:
    # âncora em coordenadas de figura é o padrão robusto para "legenda fora do
    # gráfico" — com ax.legend(), o bbox 'tight' do backend inline do Jupyter
    # cortava a legenda mais longa (`sealed`) de forma inconsistente.
    fig.subplots_adjust(left=0.09, right=0.62, top=0.72, bottom=0.2)
    fig.legend(handles=partition_handles, loc="upper left", bbox_to_anchor=(0.64, 0.72), fontsize=8)
    fig.legend(handles=arm_handles, loc="upper left", bbox_to_anchor=(0.64, 0.46), fontsize=8)
    fig.legend(handles=outcome_handles, loc="upper left", bbox_to_anchor=(0.64, 0.22), fontsize=8)
    add_chart_header(fig, title, subtitle)
    return fig, ax


def plot_permutation_noise_floor(permuted_stds, real_stds, title, subtitle=None,
                                  xlabel="Desvio-padrão do CATE (validação)", ylabel="Frequência",
                                  figsize=(9, 5)):
    """Histograma do chão de ruído por permutação (S4.4), com marcadores verticais
    para o desvio-padrão real de cada meta-learner.

    `permuted_stds`: array de `std(CATE)` sob tratamento embaralhado (a
    distribuição nula). `real_stds`: dict {nome_do_learner: std real}, já
    medido em S4.2.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(permuted_stds, bins=15, color="#B0B0B0", alpha=0.7, label="Ruído (tratamento embaralhado)")
    colors = {"S": "#4C72B0", "T": "#DD8452", "X": "#55A868", "R": "#C44E52"}
    for name, std in real_stds.items():
        ax.axvline(std, color=colors.get(name, "black"), lw=2.5, label=f"{name}-learner (real)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    fig.subplots_adjust(top=0.78)
    add_chart_header(fig, title, subtitle)
    return fig, ax


def plot_gates_bars(gates_df, title, subtitle=None, xlabel="Quintil de CATE estimado (0=menor, 4=maior)",
                     ylabel="ATE dentro do grupo (IC 95%)", figsize=(9, 5)):
    """Barras de ATE por quintil de CATE (GATES-lite, S4.4), com IC 95%.

    `gates_df`: saída de `evaluation.gates_by_cate_quintile` (colunas 'group',
    'ate', 'ci_low', 'ci_high').
    """
    fig, ax = plt.subplots(figsize=figsize)
    yerr = [gates_df["ate"] - gates_df["ci_low"], gates_df["ci_high"] - gates_df["ate"]]
    ax.bar(gates_df["group"].astype(str), gates_df["ate"], yerr=yerr, capsize=4, color="#4C72B0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.subplots_adjust(top=0.78)
    add_chart_header(fig, title, subtitle)
    return fig, ax


def plot_uplift_distributions(cate_dict, title, subtitle=None, xlabel="CATE estimado (uplift)",
                               ylabel="Densidade", bins=40, figsize=(10, 5)):
    """Histogramas sobrepostos (contorno, sem preenchimento) do CATE estimado por learner.

    `cate_dict`: dict {nome_do_learner: array de CATE}.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, values in cate_dict.items():
        sns.histplot(values, bins=bins, stat="density", element="step", fill=False,
                      linewidth=2, ax=ax, label=name)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="Learner")
    fig.subplots_adjust(top=0.78)
    add_chart_header(fig, title, subtitle)
    return fig, ax
