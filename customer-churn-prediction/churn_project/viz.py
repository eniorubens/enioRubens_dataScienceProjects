from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd
import phik
from phik.report import plot_correlation_matrix
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, recall_score

RANDOM_SEED = 738

# ---------------------------------------------------------------------------
# Module-level LangMap support
# ---------------------------------------------------------------------------
_LANG = None
_UNSET = object()


def set_lang(lang) -> None:
    """Configure module-level translator, e.g. LangMap(source='en', target='pt').

    Call once from the notebook/script before any plotting.
    """
    global _LANG
    _LANG = lang


def _t(text: str | None) -> str | None:
    """Translate *text* via the module-level LangMap; passthrough if none set."""
    if _LANG is None or text is None:
        return text
    return _LANG({text: text})[text]


recall_macro_scorer = make_scorer(
    recall_score,
    average="macro",
    zero_division=0,
)


# ---------------------------------------------------------------------------
# Corporate theme
# ---------------------------------------------------------------------------

def set_corporate_theme():
    """
    Corporate plotting theme inspired by editorial / financial charts,
    preserving strong readability.
    """
    sns.set_theme(style="ticks")

    mpl.rcParams['figure.dpi'] = 96
    mpl.rcParams['figure.figsize'] = (13.33, 6.5)

    # Titles
    mpl.rcParams['figure.titlesize'] = 22
    mpl.rcParams['figure.titleweight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = 22
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.titlepad'] = 20

    # Axis labels
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['axes.labelweight'] = 'bold'

    # Axes / grid
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'y'
    mpl.rcParams['grid.alpha'] = 0.45
    mpl.rcParams['grid.linewidth'] = 1.0
    mpl.rcParams['ytick.left'] = False

    # Legend
    mpl.rcParams['legend.title_fontsize'] = 14
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 1
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.facecolor'] = 'white'
    mpl.rcParams['legend.edgecolor'] = 'gray'
    mpl.rcParams['legend.borderpad'] = 0.6

    # Lines / markers
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 10

    # Pandas display
    pd.set_option('display.max_rows', 3000)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 2000)
    pd.options.display.max_colwidth = 1000


def add_corporate_header(
    fig,
    title,
    subtitle=None,
    x=0.015,
    y_title=0.93,
    y_subtitle=0.885,
    title_size=22,
    subtitle_size=16
):
    """
    Add a clean editorial-style title/subtitle without decorative line.
    """
    fig.text(
        x=x,
        y=y_title,
        s=title,
        ha='left',
        va='bottom',
        fontsize=title_size,
        fontweight='bold',
        color='#4a4a4a'
    )

    if subtitle:
        fig.text(
            x=x,
            y=y_subtitle,
            s=subtitle,
            ha='left',
            va='bottom',
            fontsize=subtitle_size,
            color='#4a4a4a'
        )


def add_corporate_footer(
    fig,
    text=None,
    enabled=True,
    x=0.015,
    y=0.01,
    fontsize=9,
    color="#888888",
    data_source=None,
    method=None,
):
    """
    Standard chart footer with optional data source and method annotation.
    """
    parts = []
    if data_source:
        parts.append(f"Source: {data_source}")
    if method:
        parts.append(f"Method: {method}")
    if text:
        parts.append(text)
    footer_text = "  |  ".join(parts) if parts else None
    if footer_text and enabled:
        fig.text(x=x, y=y, s=footer_text, ha='left', va='bottom',
                 fontsize=fontsize, color=color, style='italic')
        fig.subplots_adjust(bottom=0.08)


def format_corporate_axes(
    ax,
    hide_y_spine_only=False,
    hide_y_entirely=False,
    grid=True,
    grid_axis='y'
):
    """
    Axis formatter with three useful modes:
    - default: normal axes
    - hide_y_spine_only=True: hides left spine/tick marks, keeps y labels
    - hide_y_entirely=True: hides y axis completely
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if hide_y_entirely:
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
    elif hide_y_spine_only:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', left=False, labelleft=True)

    ax.tick_params(axis='x', labelsize=11, length=6, width=1.0, color="#2b2b2b")
    ax.tick_params(axis='y', labelsize=11, length=0, width=0.0, color="#2b2b2b")

    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.45, linewidth=1.0, color="#bdbdbd")
    else:
        ax.grid(False)


# ---------------------------------------------------------------------------
# Finance header style
# ---------------------------------------------------------------------------

def add_finance_header(
    fig,
    title,
    subtitle=None,
    x_line=0.0,
    y_line=0.98,
    line_width=0.9,
    rect_width=0.04,
    text_pad=0.012,
    title_offset=0.055,
    subtitle_offset=0.095,
    color='#E3120B'
):
    ax0 = fig.axes[0]

    x_text = x_line + text_pad

    fig.text(
        x=x_text,
        y=y_line - title_offset,
        s=title,
        ha='left',
        va='bottom',
        fontsize=16,
        fontweight='bold',
        style='italic'
    )

    if subtitle:
        fig.text(
            x=x_text,
            y=y_line - subtitle_offset,
            s=subtitle,
            ha='left',
            va='bottom',
            fontsize=12,
            style='italic'
        )


# ---------------------------------------------------------------------------
# Business / impact charts
# ---------------------------------------------------------------------------

def plot_annual_churn_impact(
    annual_revenue=50_000_000,
    churn_rate=0.02,
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    line_color='red'
):
    """
    Plot cumulative annual revenue loss under a fixed monthly churn rate.
    """
    if title is _UNSET:
        title = _t("Impact of Churn on Annual Revenue")
    if subtitle is _UNSET:
        subtitle = _t(
            f"A {churn_rate:.0%} monthly churn rate can significantly erode "
            f"revenue in a ${annual_revenue / 1e6:.0f}M ARR business"
        )

    def revenue_formatter(x, pos):
        return f"{x / 1e6:.1f} M"

    monthly_lost_revenue = annual_revenue * churn_rate
    cumulative_lost_revenue = [monthly_lost_revenue * i for i in range(1, 13)]

    months = list(range(1, 13))
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig, ax = plt.subplots(figsize=(12.33, 5.5), dpi=96)

    ax.plot(
        months,
        cumulative_lost_revenue,
        marker='.',
        color=line_color,
        linewidth=.6
    )

    ax.set_ylabel(_t('Cumulative Lost Revenue ($)'), fontsize=12, labelpad=10)
    ax.set_xlabel('')
    ax.set_xticks(months)
    ax.set_xticklabels(month_labels)

    ax.yaxis.set_major_formatter(FuncFormatter(revenue_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    format_corporate_axes(
        ax,
        hide_y_spine_only=True,
        hide_y_entirely=False,
        grid=True,
        grid_axis='y'
    )

    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


# ---------------------------------------------------------------------------
# EDA — distribution charts
# ---------------------------------------------------------------------------

def plot_churn_distribution(
    df,
    churn_col='Churn',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    palette=('coral', 'cornflowerblue')
):
    """
    Plot churn class distribution as percentage bars.
    """
    if title is _UNSET:
        title = _t("Number of Customers in Each Churn Class")
    if subtitle is _UNSET:
        subtitle = _t("Class distribution of the target variable")
    counts = df[churn_col].value_counts(normalize=True) * 100

    label_map = {
        'No': 'No Churn',
        'Yes': 'Churn',
        'No Churn': 'No Churn',
        'Churn': 'Churn',
        0: 'No Churn',
        1: 'Churn'
    }

    labels = [label_map.get(idx, str(idx)) for idx in counts.index]
    values = counts.values.tolist()

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values, width=0.5, color=list(palette))
    ax.bar_label(bars, fmt='%.2f%%', fontsize=12, color="#2b2b2b")

    format_corporate_axes(ax, hide_y_spine_only=False, hide_y_entirely=True, grid=False)
    ax.set_ylim(0, max(values) * 1.15)

    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_pairplot_corporate(
    df,
    rename_dict=None,
    hue='Contract',
    title=_UNSET,
    subtitle=None,
    palette='husl',
    height=2,
    show_footer=False,
    footer_text=None
):
    """
    Generate a Seaborn pairplot in corporate style.
    """
    if title is _UNSET:
        title = _t("Pairplot of Numerical Features")
    if rename_dict:
        data = df.rename(columns=rename_dict)
    else:
        data = df.copy()

    g = sns.pairplot(data, hue=hue, palette=palette, height=height, diag_kind='kde')

    fig = g.figure

    add_corporate_header(fig, title=title, subtitle=subtitle)

    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.82, hspace=0.2, wspace=0.2)

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)

    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_title(hue)
        g._legend.set_frame_on(False)

    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.show()
    return g


def plot_gender_distribution(
    df,
    gender_col='gender',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    palette=('coral', 'cornflowerblue')
):
    """
    Plot gender distribution as percentage bars.
    """
    if title is _UNSET:
        title = _t("Gender Distribution")
    if subtitle is _UNSET:
        subtitle = _t("Customer base is evenly distributed across genders")
    gender_counts = df[gender_col].value_counts(normalize=True) * 100

    labels = gender_counts.index.tolist()
    values = gender_counts.values.tolist()

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values, width=0.5, color=list(palette))
    ax.bar_label(bars, fmt='%.1f%%', fontsize=12, color="#2b2b2b")
    ax.set_ylim(0, max(values) * 1.15)

    format_corporate_axes(ax, hide_y_spine_only=False, hide_y_entirely=True, grid=False)
    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_senior_distribution(
    df,
    col='SeniorCitizen',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    palette=('coral', 'cornflowerblue')
):
    if title is _UNSET:
        title = _t("Customer Age Group Distribution")
    if subtitle is _UNSET:
        subtitle = _t("Customer base is concentrated in non-senior segment")
    dist = df[col].value_counts(normalize=True) * 100
    labels_map = {0: 'Young/Adult', 1: 'Senior'}

    labels = [labels_map[i] for i in dist.index]
    values = dist.values

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values, color=palette, width=0.5)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)

    format_corporate_axes(ax, hide_y_spine_only=False, hide_y_entirely=True, grid=False)
    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_household_composition(
    df,
    partner_col='Partner',
    dependents_col='Dependents',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    palette=('coral', 'cornflowerblue')
):
    """
    Plot household composition using stacked percentage bars.
    """
    if title is _UNSET:
        title = _t("Customers with Dependents and/or Partners")
    if subtitle is _UNSET:
        subtitle = _t("Customer distribution by household composition")
    data = df.copy()

    data['Partner_flag'] = data[partner_col].map({'Yes': 1, 'No': 0})
    data['Dependents_flag'] = data[dependents_col].map({'Yes': 1, 'No': 0})

    partner_dist = data['Partner_flag'].value_counts(normalize=True) * 100
    dependents_dist = data['Dependents_flag'].value_counts(normalize=True) * 100
    both = ((data['Partner_flag'] == 1) & (data['Dependents_flag'] == 1)).mean() * 100

    plot_df = pd.DataFrame({
        'Customer With': [
            dependents_dist[1],
            partner_dist[1],
            both
        ],
        'Customer Without': [
            dependents_dist[0],
            partner_dist[0],
            100 - both
        ]
    }, index=['Dependents', 'Partner', 'Partner & Dependent'])

    fig, ax = plt.subplots()

    bottom = np.zeros(len(plot_df))

    for i, col in enumerate(plot_df.columns):
        bars = ax.bar(plot_df.index, plot_df[col], bottom=bottom,
                      color=palette[i], label=col)
        ax.bar_label(bars, label_type='center', fmt='%.2f%%',
                     color='white', fontsize=12)
        bottom += plot_df[col].values

    format_corporate_axes(ax, hide_y_spine_only=False, hide_y_entirely=True, grid=False)
    add_corporate_header(fig, title=title, subtitle=subtitle)
    ax.legend(frameon=True, loc='upper right')
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_tenure_distribution(
    df,
    tenure_col='tenure',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    bins=36,
    color='darkblue'
):
    """
    Plot tenure distribution using histogram + KDE.
    """
    if title is _UNSET:
        title = _t("Number of Customers by Tenure")
    if subtitle is _UNSET:
        subtitle = _t("Customer retention duration distribution")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(df[tenure_col], bins=bins, color=color, edgecolor='black',
                 alpha=0.35, stat='density', ax=ax)
    sns.kdeplot(df[tenure_col], color=color, linewidth=3, ax=ax)

    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Density')

    format_corporate_axes(ax, hide_y_spine_only=True, hide_y_entirely=False,
                          grid=True, grid_axis='y')
    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_contract_distribution(
    df,
    contract_col='Contract',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    palette=('coral', 'cornflowerblue', 'silver')
):
    """
    Plot the number of customers in each contract type.
    """
    if title is _UNSET:
        title = _t("Number of Customers by Contract Type")
    if subtitle is _UNSET:
        subtitle = _t("Customer count by contract category")
    counts = df[contract_col].value_counts()

    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig, ax = plt.subplots()

    bars = ax.bar(labels, values, width=0.5, color=list(palette[:len(labels)]))
    ax.bar_label(bars, fmt='{:,.0f}', label_type='center', color='white', fontsize=12)

    format_corporate_axes(ax, hide_y_spine_only=False, hide_y_entirely=True, grid=False)
    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


def plot_tenure_by_contract(
    df,
    contract_col='Contract',
    tenure_col='tenure',
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    bins=36,
    palette=('coral', 'cornflowerblue', 'silver')
):
    """
    Plot tenure distributions separately for each contract type.
    """
    if title is _UNSET:
        title = _t("Tenure Distribution by Contract Type")
    if subtitle is _UNSET:
        subtitle = _t("Customer tenure segmented by contract structure")
    contract_order = ['Month-to-month', 'One year', 'Two year']
    subplot_titles = ['Month-to-Month Contract', 'One Year Contract', 'Two Year Contract']

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18, 8))

    for ax, contract_name, subplot_title, color in zip(axes, contract_order, subplot_titles, palette):
        subset = df[df[contract_col] == contract_name]

        sns.histplot(subset[tenure_col], bins=bins, kde=False, color=color,
                     edgecolor='black', alpha=0.35, ax=ax)

        ax.set_title(subplot_title)
        ax.set_xlabel('Tenure (months)')

        if ax is axes[0]:
            ax.set_ylabel('Number of Customers')
            format_corporate_axes(ax, hide_y_spine_only=True, hide_y_entirely=False,
                                  grid=True, grid_axis='y')
        else:
            ax.set_ylabel('')
            format_corporate_axes(ax, hide_y_spine_only=True, hide_y_entirely=False,
                                  grid=True, grid_axis='y')

    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.08, 1, 0.88])
    plt.show()


def plot_services_distribution(
    df,
    services=None,
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False
):
    """
    Plot service distributions in a 3x3 grid with consistent category colors.
    """
    if title is _UNSET:
        title = _t("Services Distribution")
    if subtitle is _UNSET:
        subtitle = _t("Customer adoption across subscribed services")
    data = df.copy()

    data = data.replace({
        'No internet service': 'No',
        'No phone service': 'No'
    })

    if services is None:
        services = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    color_map = {
        'No': 'coral',
        'Yes': 'cornflowerblue',
        'Fiber optic': 'silver',
        'DSL': 'cornflowerblue',
        'Month-to-month': 'coral',
        'One year': 'silver',
        'Two year': 'cornflowerblue'
    }

    fallback_color = 'silver'

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    axes = axes.flatten()

    for i, col in enumerate(services):
        ax = axes[i]

        counts = data[col].value_counts()
        labels = counts.index.tolist()
        values = counts.values.tolist()

        bar_colors = [color_map.get(label, fallback_color) for label in labels]
        ax.bar(labels, values, color=bar_colors, width=0.5)

        ax.set_title(col, fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')

        format_corporate_axes(ax, hide_y_spine_only=True, hide_y_entirely=False,
                               grid=True, grid_axis='y')
        ax.tick_params(axis='x', rotation=0)

    add_corporate_header(fig, title=title, subtitle=subtitle)
    add_corporate_footer(fig, text=footer_text, enabled=show_footer)

    plt.tight_layout(rect=[0, 0.06, 1, 0.88])
    plt.show()


# ---------------------------------------------------------------------------
# Correlation charts
# ---------------------------------------------------------------------------

def plot_pearson_correlation(
    df,
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    annot=True,
    figsize=(15, 10)
):
    """
    Plot Pearson correlation clustermap for numerical variables only.
    """
    if title is _UNSET:
        title = _t("Pearson Correlation")
    if subtitle is _UNSET:
        subtitle = _t("Linear relationships among numerical variables")
    numerical_features = df.select_dtypes(exclude=['category', 'object']).columns.tolist()
    corr_matrix = df[numerical_features].corr(method='pearson')

    cg = sns.clustermap(corr_matrix, annot=annot, annot_kws={"size": 10},
                        linewidths=0.4, figsize=figsize, cmap="rocket")

    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=40)

    fig = cg.figure

    fig.suptitle(title, x=0.02, y=0.98, ha='left', fontsize=22,
                 fontweight='bold', color='#2b2b2b')
    if subtitle:
        fig.text(0.02, 0.93, subtitle, ha='left', fontsize=14, color='#4a4a4a')

    if show_footer and footer_text:
        fig.text(0.02, 0.02, footer_text, ha='left', fontsize=10, color='#4a4a4a')

    fig.subplots_adjust(
        top=0.93 if not subtitle else 0.88,
        bottom=0.08 if not show_footer else 0.10
    )

    plt.show()
    return corr_matrix


def plot_phik_correlation(
    df,
    drop_cols=None,
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    figsize=(16, 10),
    color_map="Blues"
):
    """
    Plot PhiK correlation matrix for mixed-type variables.
    """
    if title is _UNSET:
        title = _t("Telecom PhiK Correlation")
    if subtitle is _UNSET:
        subtitle = _t("Association structure across mixed-type variables")
    data = df.copy(deep=True)

    if drop_cols is not None:
        data = data.drop(columns=drop_cols)

    phik_matrix = data.phik_matrix()

    plot_correlation_matrix(
        phik_matrix.values,
        x_labels=phik_matrix.columns,
        y_labels=phik_matrix.index,
        vmin=0,
        vmax=1,
        color_map=color_map,
        title="",
        fontsize_factor=1.5,
        figsize=figsize
    )

    fig = plt.gcf()

    fig.suptitle(title, x=0.02, y=0.98, ha='left', fontsize=22,
                 fontweight='bold', color='#2b2b2b')
    if subtitle:
        fig.text(0.02, 0.93, subtitle, ha='left', fontsize=14, color='#4a4a4a')

    if show_footer and footer_text:
        fig.text(0.02, 0.02, footer_text, ha='left', fontsize=10, color='#4a4a4a')

    plt.tight_layout(rect=[0, 0.05 if show_footer else 0.06, 1, 0.90])
    plt.show()

    return phik_matrix


def plot_phik_significance(
    df,
    drop_cols=None,
    title=_UNSET,
    subtitle=_UNSET,
    footer_text=None,
    show_footer=False,
    figsize=(16, 10)
):
    """
    Plot PhiK significance matrix.
    """
    if title is _UNSET:
        title = _t("Significance of the Coefficients")
    if subtitle is _UNSET:
        subtitle = _t("Statistical significance of observed associations")
    data = df.copy(deep=True)

    if drop_cols is not None:
        data = data.drop(columns=drop_cols)

    significance_overview = data.significance_matrix()

    plot_correlation_matrix(
        significance_overview.fillna(0).values,
        x_labels=significance_overview.columns,
        y_labels=significance_overview.index,
        vmin=-5,
        vmax=5,
        title="",
        usetex=False,
        fontsize_factor=1.5,
        figsize=figsize
    )

    fig = plt.gcf()

    fig.suptitle(title, x=0.02, y=0.98, ha='left', fontsize=22,
                 fontweight='bold', color='#2b2b2b')
    if subtitle:
        fig.text(0.02, 0.93, subtitle, ha='left', fontsize=14, color='#4a4a4a')

    if show_footer and footer_text:
        fig.text(0.02, 0.02, footer_text, ha='left', fontsize=10, color='#4a4a4a')

    plt.tight_layout(rect=[0, 0.05 if show_footer else 0.06, 1, 0.90])
    plt.show()

    return significance_overview


# ---------------------------------------------------------------------------
# Bivariate charts (churn vs feature)
# ---------------------------------------------------------------------------

def plot_churn_vs_tenure(df, churn_col='Churn', tenure_col='tenure', title=_UNSET, subtitle=_UNSET):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import phik
    import pandas as pd

    if title is _UNSET:
        title = _t("Churn vs Tenure")
    if subtitle is _UNSET:
        subtitle = _t("Customer lifecycle impact on churn behavior")

    data = df.copy()
    data[churn_col] = data[churn_col].map({'No': 0, 'Yes': 1})

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        x=df[churn_col],
        y=df[tenure_col],
        palette=['cornflowerblue', 'coral'],
        ax=ax
    )

    ax.set_xlabel(_t('Churn'))
    ax.set_ylabel(_t('Tenure (months)'))

    add_finance_header(fig, title=title, subtitle=subtitle)

    plt.grid(False)
    plt.tight_layout()
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.show()

    x, y = data[[churn_col, tenure_col]].T.values

    phik_value = phik.phik_from_array(x, y, num_vars=['x'])
    significance = phik.significance_from_array(x, y, num_vars=['x'])[1]

    print(f'phik         = {phik_value:.4f}')
    print(f'significance = {significance:.4f}')


def plot_churn_vs_contract(df, churn_col='Churn', contract_col='Contract', title=_UNSET, subtitle=_UNSET):
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as mtick

    if title is _UNSET:
        title = _t("Churn by Contract Type")
    if subtitle is _UNSET:
        subtitle = _t("Impact of contract duration on customer retention")

    contract_churn = df.groupby([contract_col, churn_col]).size().unstack()
    pct = (contract_churn.T * 100.0 / contract_churn.T.sum()).T
    index = pct.index.tolist()

    data_plot = {
        _t('No Churn'): pct['No'].values,
        _t('Churn'): pct['Yes'].values
    }

    colors = ['cornflowerblue', 'coral']
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(index))

    for i, (label, values) in enumerate(data_plot.items()):
        bars = ax.bar(index, values, width, bottom=bottom, label=label, color=colors[i])
        ax.bar_label(bars, label_type='center', fmt='%.1f%%', color='white')
        bottom += values

    add_finance_header(fig, title=title, subtitle=subtitle)

    ax.get_yaxis().set_visible(False)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.show()

    data = df.copy()
    data[churn_col] = data[churn_col].map({'No': 0, 'Yes': 1})

    x, y = data[[churn_col, contract_col]].T.values

    import phik
    phik_value = phik.phik_from_array(x, y)
    significance = phik.significance_from_array(x, y)[1]

    print(f'phik         = {phik_value:.4f}')
    print(f'significance = {significance:.4f}')


def plot_churn_vs_monthly_charges(df: pd.DataFrame, title=_UNSET, subtitle=_UNSET) -> None:
    """
    Plot KDE distribution of Monthly Charges segmented by churn
    and display skewness, kurtosis, and PhiK correlation.
    """
    if title is _UNSET:
        title = _t("Distribution of Monthly Charges by Churn")
    if subtitle is _UNSET:
        subtitle = _t("Charge intensity and churn behavior")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(df[df["Churn"] == "No"]["MonthlyCharges"], ax=ax,
                color="coral", fill=True, linewidth=2, label=_t("No Churn"))
    sns.kdeplot(df[df["Churn"] == "Yes"]["MonthlyCharges"], ax=ax,
                color="cornflowerblue", fill=True, linewidth=2, label=_t("Churn"))

    ax.set_xlabel(_t("Monthly Charges"))
    ax.set_ylabel(_t("Density"))
    ax.legend(loc="upper right")

    add_finance_header(fig, title=title, subtitle=subtitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.show()

    def describe_distribution(series: pd.Series, label: str) -> None:
        skew = series.skew()
        kurt = series.kurtosis()

        if -0.5 <= skew <= 0.5:
            skew_desc = "approximately symmetric"
        elif -1 <= skew <= 1:
            skew_desc = "moderately skewed"
        else:
            skew_desc = "highly skewed"

        kurt_desc = (
            "leptokurtic" if kurt > 0
            else "platykurtic" if kurt < 0
            else "mesokurtic"
        )

        print(f"\n{label}")
        print(f"  Skewness : {skew:.4f} ({skew_desc})")
        print(f"  Kurtosis : {kurt:.4f} ({kurt_desc})")

    describe_distribution(df[df["Churn"] == "No"]["MonthlyCharges"], "No Churn Distribution")
    describe_distribution(df[df["Churn"] == "Yes"]["MonthlyCharges"], "Churn Distribution")

    temp = df[["Churn", "MonthlyCharges"]].copy()
    temp["Churn"] = temp["Churn"].map({"No": 0, "Yes": 1})

    x1, y1 = temp[["Churn", "MonthlyCharges"]].T.values

    print(f"\nphik         = {phik.phik_from_array(x1, y1):.4f}")
    print(f"significance = {phik.significance_from_array(x1, y1)[1]:.4f}")


# ---------------------------------------------------------------------------
# Target / split distribution charts
# ---------------------------------------------------------------------------

def plot_target_distribution(
    df: pd.DataFrame,
    feature: str,
    title: str,
    subtitle: str,
    figsize: tuple[float, float] = (8, 6)
) -> None:
    """
    Plot target distribution as a pie chart.
    """
    distribution = df[feature].value_counts(normalize=True).sort_index() * 100
    labels = distribution.index.astype(str).tolist()
    colors = ["cornflowerblue", "coral"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.pie(
        distribution.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        shadow=True,
        explode=(0.08, 0.08),
        colors=colors
    )

    add_finance_header(fig, title=title, subtitle=subtitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()


def plot_target_distribution_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature: str,
    title: str,
    subtitle: str,
    figsize: tuple[float, float] = (12, 5)
) -> None:
    """
    Plot side-by-side pie charts for train and test target distributions.
    """
    colors = ["cornflowerblue", "coral"]

    train_dist = train_df[feature].value_counts(normalize=True).sort_index() * 100
    test_dist = test_df[feature].value_counts(normalize=True).sort_index() * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].pie(train_dist.values, labels=train_dist.index.astype(str).tolist(),
                autopct="%1.1f%%", startangle=90, shadow=True,
                explode=(0.08, 0.08), colors=colors)
    axes[0].set_title("Train Dataset")

    axes[1].pie(test_dist.values, labels=test_dist.index.astype(str).tolist(),
                autopct="%1.1f%%", startangle=90, shadow=True,
                explode=(0.08, 0.08), colors=colors)
    axes[1].set_title("Test Dataset")

    add_finance_header(fig, title=title, subtitle=subtitle)

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation charts
# ---------------------------------------------------------------------------

def plot_importance(
    model,
    title='',
    train_X=None,
    train_y=None,
    test_X=None,
    test_y=None,
    scoring=recall_macro_scorer,
):
    train_X = train_features if train_X is None else train_X
    train_y = train_labels if train_y is None else train_y
    test_X = test_features if test_X is None else test_X
    test_y = test_labels if test_y is None else test_y

    if not train_X.columns.equals(test_X.columns):
        raise ValueError("Train and test feature columns must match for permutation importance.")

    feature_names = train_X.columns

    train_result = permutation_importance(
        model, train_X, train_y,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=1, scoring=scoring,
    )
    test_result = permutation_importance(
        model, test_X, test_y,
        n_repeats=10, random_state=RANDOM_SEED, n_jobs=1, scoring=scoring,
    )
    sorted_importances_idx = test_result.importances_mean.argsort()

    train_importances = pd.DataFrame(
        train_result.importances[sorted_importances_idx].T,
        columns=feature_names[sorted_importances_idx],
    )
    test_importances = pd.DataFrame(
        test_result.importances[sorted_importances_idx].T,
        columns=feature_names[sorted_importances_idx],
    )

    for name, importances in zip(["train", "test"], [train_importances, test_importances]):
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_title(f" {title} Permutation Importances ({name} set)")
        ax.set_xlabel("Decrease in recall macro score")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.figure.tight_layout()
        plt.grid(False)
        plt.show()


def formatter(x, pos):
    return str(round(x / 1e6, 1)) + " M"
