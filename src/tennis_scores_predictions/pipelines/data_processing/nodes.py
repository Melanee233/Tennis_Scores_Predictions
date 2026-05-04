"""Transformation nodes for the data processing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_COLUMNS_BEFORE_MATCH = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_seed",
    "winner_entry",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_ioc",
    "winner_age",
    "loser_id",
    "loser_seed",
    "loser_entry",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_ioc",
    "loser_age",
    "round",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]


def select_columns_before_match(df: pd.DataFrame) -> pd.DataFrame:
    return df[_COLUMNS_BEFORE_MATCH].copy()


def filter_active_players(df: pd.DataFrame) -> pd.DataFrame:
    """Keep matches involving players active since 2020."""
    last_years_df = df[df["tourney_date"] >= 20200000]
    active_players = set(
        np.concatenate(
            [last_years_df["winner_name"].values, last_years_df["loser_name"].values]
        )
    )
    return df[
        df["winner_name"].isin(active_players) | df["loser_name"].isin(active_players)
    ].copy()


def filter_active_players_node(df: pd.DataFrame) -> pd.DataFrame:
    return filter_active_players(df)


def replace_string_nan_with_na(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace("nan", np.nan)


def impute_winner_loser_entry(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["winner_entry"] = out["winner_entry"].fillna("R")
    out["loser_entry"] = out["loser_entry"].fillna("R")
    return out


def drop_seed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["winner_seed", "loser_seed"])


def impute_player_heights(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["winner_ht"] = out["winner_ht"].fillna(round(out["winner_ht"].mean())).astype(int)
    out["loser_ht"] = out["loser_ht"].fillna(round(out["loser_ht"].mean())).astype(int)
    return out


def drop_missing_rank_points(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["winner_rank_points", "loser_rank_points"]).copy()


def drop_missing_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["winner_rank", "loser_rank"]).copy()


def drop_missing_demographics(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(
        subset=["surface", "loser_age", "winner_age", "loser_hand"]
    ).copy()


def drop_match_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["match_num", "winner_id", "loser_id"])


def transform_names(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Map winner/loser columns to symmetric player_A/player_B features."""
    np.random.seed(random_state)

    result_df = df.copy()
    result_df["player_A"] = result_df["winner_name"]
    result_df["player_B"] = result_df["loser_name"]
    result_df["player_A_won"] = 1

    winner_cols = [col for col in df.columns if col.startswith("winner_")]
    loser_cols = [col for col in df.columns if col.startswith("loser_")]

    for winner_col in winner_cols:
        base_name = winner_col.replace("winner_", "")
        loser_col = f"loser_{base_name}"
        if loser_col in df.columns:
            result_df[f"player_A_{base_name}"] = result_df[winner_col]
            result_df[f"player_B_{base_name}"] = result_df[loser_col]

    mask = np.random.rand(len(result_df)) < 0.5

    for col in result_df.columns:
        if not col.startswith("player_A_"):
            continue
        base_name = col.replace("player_A_", "")
        player_b_col = f"player_B_{base_name}"
        if player_b_col not in result_df.columns:
            continue
        temp = result_df.loc[mask, col].copy()
        result_df.loc[mask, col] = result_df.loc[mask, player_b_col]
        result_df.loc[mask, player_b_col] = temp

    temp_players = result_df.loc[mask, "player_A"].copy()
    result_df.loc[mask, "player_A"] = result_df.loc[mask, "player_B"]
    result_df.loc[mask, "player_B"] = temp_players
    result_df.loc[mask, "player_A_won"] = 0
    result_df.loc[~mask, "player_A_won"] = 1

    return result_df.drop(columns=winner_cols + loser_cols, errors="ignore")


def transform_names_node(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return transform_names(df, random_state=params["random_state"])


def add_round_index(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"tourney_date", "round", "tourney_name"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Brakuje wymaganych kolumn: {sorted(missing_columns)}")

    result_df = df.copy()
    result_df["year"] = result_df["tourney_date"].astype(str).str[:4]
    round_order = {
        "R128": 1,
        "R64": 2,
        "R32": 3,
        "R16": 4,
        "QF": 5,
        "SF": 6,
        "F": 7,
        "RR": 8,
        "BR": 8,
    }
    result_df["round_order"] = result_df["round"].map(round_order)
    result_df = result_df.sort_values(["tourney_name", "year", "round_order"])
    result_df["index_round"] = (
        result_df.groupby(["tourney_name", "year"])["round_order"]
        .rank(method="dense")
        .astype(int)
    )
    return result_df.drop(columns=["year", "round_order"])


def compute_elo(
    df: pd.DataFrame, init_rating: float = 1500.0, k_factor: float = 32
) -> pd.DataFrame:
    sorted_df = df.sort_values(
        by=["tourney_date", "tourney_name", "index_round"]
    ).reset_index(drop=True)

    elo: dict[str, float] = {}
    elo_a_before: list[float] = []
    elo_b_before: list[float] = []
    elo_a_after: list[float] = []
    elo_b_after: list[float] = []

    for _, row in sorted_df.iterrows():
        player_a = row["player_A"]
        player_b = row["player_B"]

        rating_a = elo.get(player_a, init_rating)
        rating_b = elo.get(player_b, init_rating)
        elo_a_before.append(rating_a)
        elo_b_before.append(rating_b)

        expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a
        score_a, score_b = (1.0, 0.0) if row["player_A_won"] == 1 else (0.0, 1.0)

        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * (score_b - expected_b)

        elo[player_a] = new_rating_a
        elo[player_b] = new_rating_b
        elo_a_after.append(new_rating_a)
        elo_b_after.append(new_rating_b)

    out = sorted_df.copy()
    out["elo_A_before"] = elo_a_before
    out["elo_B_before"] = elo_b_before
    out["elo_A_after"] = elo_a_after
    out["elo_B_after"] = elo_b_after
    return out


def compute_elo_node(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    return compute_elo(
        df,
        init_rating=params["elo_init_rating"],
        k_factor=params["elo_k_factor"],
    )


def calculate_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    for feature in ["ht", "age", "rank", "rank_points"]:
        col_a = f"player_A_{feature}"
        col_b = f"player_B_{feature}"
        if col_a in result_df.columns and col_b in result_df.columns:
            result_df[f"diff_{feature}"] = result_df[col_a] - result_df[col_b]

    if "elo_A_before" in result_df.columns and "elo_B_before" in result_df.columns:
        result_df["diff_elo_before"] = (
            result_df["elo_A_before"] - result_df["elo_B_before"]
        )

    cols_to_drop = [
        col
        for col in result_df.columns
        if col != "player_A_won"
        and (col.startswith("player_A_") or col.startswith("player_B_"))
    ]
    cols_to_drop += ["elo_A_before", "elo_B_before"]
    return result_df.drop(columns=cols_to_drop, errors="ignore")


def drop_feature_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "tourney_id",
            "tourney_name",
            "tourney_date",
            "player_A",
            "player_B",
            "index_round",
            "elo_A_after",
            "elo_B_after",
        ],
        errors="ignore",
    )


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    result_df = df.copy()
    cat_cols = [col for col in result_df.columns if result_df[col].dtype == "object"]
    return pd.get_dummies(result_df, columns=cat_cols)
