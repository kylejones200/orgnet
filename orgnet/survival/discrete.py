"""Discrete time survival analysis using logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class DiscreteTimeSurvivalModel:
    """Discrete time survival model with logistic hazard.

    This class uses scikit learn logistic regression under the hood.
    It treats each person period row as a binary event outcome.
    """

    feature_cols: List[str]
    id_col: str
    period_col: str
    event_col: str
    model: LogisticRegression
    scaler: StandardScaler

    def predict_hazard(self, df: pd.DataFrame) -> pd.Series:
        """Predict hazard per row.

        Args:
            df: Person period table with feature columns.

        Returns:
            Series with hazard probability per row in [0, 1].
        """
        x = df[self.feature_cols].to_numpy(dtype=float)
        x_scaled = self.scaler.transform(x)
        proba = self.model.predict_proba(x_scaled)[:, 1]
        return pd.Series(proba, index=df.index, name="hazard")

    def predict_hazard_and_survival(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict hazard and survival per person period.

        Args:
            df: Person period table with feature columns.

        Returns:
            DataFrame with hazard and survival columns.
            Index matches the input index.
        """
        df = df.copy()
        df = df.sort_values([self.id_col, self.period_col])

        hazard = self.predict_hazard(df)
        df["hazard"] = hazard

        # survival per person as product over periods of (1 - hazard)
        surv_list: List[pd.DataFrame] = []
        for pid, group in df.groupby(self.id_col, sort=False):
            h = group["hazard"].to_numpy()
            s = np.cumprod(1.0 - h)
            g = pd.DataFrame(
                {
                    self.id_col: pid,
                    self.period_col: group[self.period_col].to_numpy(),
                    "hazard": h,
                    "survival": s,
                },
                index=group.index,
            )
            surv_list.append(g)

        out = pd.concat(surv_list, axis=0).sort_index()
        return out


def build_person_period_table(
    features: pd.DataFrame,
    events: pd.DataFrame,
    id_col: str,
    period_col: str,
    event_period_col: str,
    censored_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build discrete person period table.

    This function assumes one row per person period in features.
    It also assumes one row per person in events.

    Example schema

    features
        id_col          person identifier
        period_col      integer period index starting at 1
        feature_cols    any numeric covariates

    events
        id_col              person identifier
        event_period_col    integer period of event or last observed period
        censored_col        optional boolean
                            False if event occurs at event_period_col
                            True if censored at event_period_col

    Args:
        features: Person period feature table.
        events: Person event table.
        id_col: Name of person identifier column.
        period_col: Name of period index column.
        event_period_col: Name of event period column in events.
        censored_col: Optional name of censor flag column.

    Returns:
        Person period table with an event column.
        Rows after the event period are dropped.
    """
    if id_col not in features.columns:
        raise ValueError(f"Missing id column {id_col} in features")
    if period_col not in features.columns:
        raise ValueError(f"Missing period column {period_col} in features")
    if id_col not in events.columns:
        raise ValueError(f"Missing id column {id_col} in events")
    if event_period_col not in events.columns:
        raise ValueError(f"Missing event period column {event_period_col} in events")

    df = features.copy()
    ev = events[[id_col, event_period_col] + ([censored_col] if censored_col else [])]

    df = df.merge(ev, on=id_col, how="left", validate="many_to_one")

    if df[event_period_col].isna().any():
        raise ValueError("Some persons in features have no event record")

    # drop periods after event or censor
    df = df[df[period_col] <= df[event_period_col]]

    if censored_col:
        cens = df[censored_col].astype(bool)
    else:
        # default to no censor
        cens = pd.Series(False, index=df.index)

    event = (df[period_col] == df[event_period_col]) & (~cens)
    df["event"] = event.astype(int)

    return df


def fit_discrete_time_model(
    person_period_df: pd.DataFrame,
    feature_cols: Sequence[str],
    id_col: str = "person_id",
    period_col: str = "period",
    event_col: str = "event",
    penalty: str = "l2",
    C: float = 1.0,
    max_iter: int = 100,
) -> DiscreteTimeSurvivalModel:
    """Fit discrete time survival model with logistic regression.

    Args:
        person_period_df: Table with one row per person period.
        feature_cols: Names of feature columns.
        id_col: Name of person identifier column.
        period_col: Name of period index column.
        event_col: Name of event indicator column.
        penalty: Penalty for logistic regression.
        C: Inverse of regularization strength for logistic regression.
        max_iter: Maximum number of iterations.

    Returns:
        DiscreteTimeSurvivalModel instance.
    """
    for col in [id_col, period_col, event_col]:
        if col not in person_period_df.columns:
            raise ValueError(f"Missing required column {col}")
    for col in feature_cols:
        if col not in person_period_df.columns:
            raise ValueError(f"Missing feature column {col}")

    df = person_period_df.copy()
    df = df.sort_values([id_col, period_col])

    x = df[feature_cols].to_numpy(dtype=float)
    y = df[event_col].to_numpy(dtype=int)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    clf.fit(x_scaled, y)

    return DiscreteTimeSurvivalModel(
        feature_cols=list(feature_cols),
        id_col=id_col,
        period_col=period_col,
        event_col=event_col,
        model=clf,
        scaler=scaler,
    )


def attach_survival_outputs(
    person_period_df: pd.DataFrame,
    dts_model: DiscreteTimeSurvivalModel,
) -> pd.DataFrame:
    """Attach hazard and survival to a person period table.

    Args:
        person_period_df: Table with one row per person period.
        dts_model: Fitted discrete time survival model.

    Returns:
        Copy of input with hazard and survival columns.
    """
    preds = dts_model.predict_hazard_and_survival(person_period_df)
    out = person_period_df.copy()
    out = out.merge(
        preds[[dts_model.id_col, dts_model.period_col, "hazard", "survival"]],
        on=[dts_model.id_col, dts_model.period_col],
        how="left",
        validate="one_to_one",
    )
    return out
