from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import LabelEncoder


def show_skewness(data: pd.DataFrame, x='', detail=''):
    l_skew = data[x].skew()
    l_kurtosis = data[x].kurt()
    l_mean = data[x].mean()

    # Right or left
    if l_skew < 0:
        l_direction = ' Left '
    elif l_skew > 0:
        l_direction = ' Right '

    # If the skewness is between -0.5 and 0.5, the data are fairly symmetrical
    if l_skew > 0.5 and l_skew < 0.5:
        g_mess = x + ' variable are fairly symmetrical'

    # If the skewness is between -1 and — 0.5 or between 0.5 and 1, the data are moderately skewed
    elif l_skew > -1 and l_skew < 1:
        g_mess = x + ' variable are moderately' + l_direction + 'Skewed'

    # If the skewness is less than -1 or greater than 1, the data are highly skewed
    else:
        g_mess = x + ' variable are highly' + l_direction + 'Skewed'

    # Kurtosis
    if l_kurtosis > l_mean:
        l_mess_kurtosis = ' Leptokurtic '
    elif l_kurtosis < l_mean:
        l_mess_kurtosis = ' Platykurtic '
    else:
        l_mess_kurtosis = ' Mesokurtic '

    g_mess += l_mess_kurtosis
    print(f'''
    Skewness for {x} when customer {detail}:
        Skew     : {l_skew:.4f}
        Kurtosis : {l_kurtosis:.4f}
            - {g_mess}
    ''')


def build_phik_significance_df(df, drop_cols=None):
    """
    Build a dataframe with variable pairs, Phik correlation and significance.
    """
    import phik

    data = df.copy()

    if drop_cols:
        data = data.drop(columns=drop_cols)

    phik_matrix = data.phik_matrix()
    significance_matrix = data.significance_matrix()

    records = []
    cols = phik_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            var1 = cols[i]
            var2 = cols[j]
            records.append({
                'var1': var1,
                'var2': var2,
                'phik': phik_matrix.iloc[i, j],
                'significance': significance_matrix.iloc[i, j]
            })

    result_df = pd.DataFrame(records)
    return result_df.sort_values(by='phik', ascending=False)


def filter_relevant_relationships(df, phik_threshold=0.3, significance_threshold=5):
    """
    Filter relevant relationships based on Phik and significance.
    """
    return df[
        (df['phik'] >= phik_threshold) &
        (df['significance'] >= significance_threshold)
    ].sort_values(by='phik', ascending=False)


def prepare_feature_sets(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_cols: List[str] | None = None
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare a clean dataframe and return categorical and numerical feature lists.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="Churn"
        Target column name.
    id_cols : list[str] | None, default=None
        Identifier columns to exclude from feature selection.

    Returns
    -------
    tuple
        Clean dataframe, categorical feature list, numerical feature list.
    """
    if id_cols is None:
        id_cols = ["customerID"]

    data = df.copy(deep=True).dropna(axis=0)

    categorical_features = data.select_dtypes(
        include=["category", "object"]
    ).columns.tolist()

    numerical_features = data.select_dtypes(
        exclude=["category", "object"]
    ).columns.tolist()

    for col in id_cols:
        if col in categorical_features:
            categorical_features.remove(col)
        if col in numerical_features:
            numerical_features.remove(col)

    if target_col in categorical_features:
        categorical_features.remove(target_col)

    if target_col in numerical_features:
        numerical_features.remove(target_col)

    return data, categorical_features, numerical_features


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_features: List[str]
) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_features : list[str]
        Categorical columns to encode.

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded categorical features.
    """
    data = df.copy(deep=True)
    encoder = LabelEncoder()

    for column in categorical_features:
        data[column] = encoder.fit_transform(data[column])

    return data


def plot_feature_selection_heatmap(
    scores_df: pd.DataFrame,
    score_col: str,
    title: str,
    subtitle: str,
    fmt: str = ".3f",
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot a ranked feature selection heatmap using the project visual style.
    """
    from churn_project.viz import add_finance_header

    plot_df = scores_df.sort_values(by=score_col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        plot_df,
        annot=True,
        cmap=sns.light_palette("cornflowerblue", as_cmap=True),
        linewidths=0.4,
        fmt=fmt,
        cbar=True,
        ax=ax
    )

    add_finance_header(
        fig,
        title=title,
        subtitle=subtitle,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    plt.show()


def plot_chi_squared_feature_selection(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Compute and plot Chi-Squared scores for categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="Churn"
        Target column name.
    id_cols : list[str] | None, default=None
        Identifier columns to exclude.

    Returns
    -------
    pd.DataFrame
        Dataframe with Chi-Squared scores.
    """
    data, categorical_features, _ = prepare_feature_sets(
        df=df,
        target_col=target_col,
        id_cols=id_cols
    )

    encoded_data = encode_categorical_features(data, categorical_features)

    features = encoded_data.loc[:, categorical_features]
    target = encoded_data.loc[:, target_col]

    selector = SelectKBest(score_func=chi2, k="all")
    selector.fit(features, target)

    feature_scores = pd.DataFrame(
        data=selector.scores_,
        index=features.columns,
        columns=["Chi-Squared Score"]
    )

    plot_feature_selection_heatmap(
        scores_df=feature_scores,
        score_col="Chi-Squared Score",
        title="Chi-Squared Test (χ²)",
        subtitle="Statistical relevance of categorical features",
        fmt=".3f",
        figsize=(10, 6)
    )

    return feature_scores.sort_values(by="Chi-Squared Score", ascending=False)


def plot_anova_feature_selection(
    df: pd.DataFrame,
    target_col: str = "Churn",
    id_cols: List[str] | None = None
) -> pd.DataFrame:
    """
    Compute and plot ANOVA F-scores for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="Churn"
        Target column name.
    id_cols : list[str] | None, default=None
        Identifier columns to exclude.

    Returns
    -------
    pd.DataFrame
        Dataframe with ANOVA scores.
    """
    data, _, numerical_features = prepare_feature_sets(
        df=df,
        target_col=target_col,
        id_cols=id_cols
    )

    features = data.loc[:, numerical_features]
    target = data.loc[:, target_col]

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(features, target)

    feature_scores = pd.DataFrame(
        data=selector.scores_,
        index=features.columns,
        columns=["ANOVA Score"]
    )

    plot_feature_selection_heatmap(
        scores_df=feature_scores,
        score_col="ANOVA Score",
        title="ANOVA Test",
        subtitle="Statistical relevance of numerical features",
        fmt=".2f",
        figsize=(8, 4.8)
    )

    return feature_scores.sort_values(by="ANOVA Score", ascending=False)
