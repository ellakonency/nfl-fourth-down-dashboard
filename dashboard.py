import pandas as pd
import plotly.express as px
import streamlit as st
import nflreadpy as nfl

st.set_page_config(
    page_title="NFL Fourth Down Dashboard",
    layout="wide"
)

st.title("NFL Fourth Down Decision Dashboard")
st.write(
    "Explore fourth-down decisions using conversion rate, EPA, expected conversion probability, "
    "and value over expectation."
)

def bucketize_situation(fourth_down):
    fourth_down["distance_bucket"] = pd.cut(
        fourth_down["ydstogo"],
        bins=[0, 2, 5, 100],
        labels=["Short (1-2)", "Medium (3-5)", "Long (6+)"]
    )

    fourth_down["field_zone"] = pd.cut(
        fourth_down["yardline_100"],
        bins=[0, 40, 60, 100],
        labels=["Opponent Territory", "Midfield", "Own Territory"]
    )

    fourth_down["game_situation"] = pd.cut(
        fourth_down["score_differential"],
        bins=[-100, -8, 0, 8, 100],
        labels=["Trailing 9+", "Trailing 1-8", "Leading 1-8", "Leading 9+"]
    )

    fourth_down["yardline_bucket"] = pd.cut(
        fourth_down["yardline_100"],
        bins=[0, 10, 20, 40, 60, 80, 100],
        labels=["Goal Line", "Red Zone", "Opp 20-40", "Midfield", "Own 20-40", "Backed Up"]
    )

    fourth_down["distance_exact_bucket"] = pd.cut(
        fourth_down["ydstogo"],
        bins=[0, 1, 2, 3, 5, 10, 100],
        labels=["1", "2", "3", "4-5", "6-10", "11+"]
    )

    fourth_down["estimated_fg_distance"] = fourth_down["yardline_100"] + 17

    fourth_down["fg_distance_bucket"] = pd.cut(
        fourth_down["estimated_fg_distance"],
        bins=[0, 35, 45, 55, 60, 100],
        labels=["Short FG", "Medium FG", "Long FG", "Very Long FG", "Not Realistic"]
    )

    return fourth_down


def smooth_rate(raw_rate, sample_size, league_rate, smoothing_weight=75):
    return (
        (raw_rate * sample_size + league_rate * smoothing_weight)
        / (sample_size + smoothing_weight)
    )


def smooth_value(raw_value, sample_size, league_value, smoothing_weight=75):
    return (
        (raw_value * sample_size + league_value * smoothing_weight)
        / (sample_size + smoothing_weight)
    )


def decision_label(best_decision, margin):
    if margin >= 0.50:
        strength = "Strong"
    elif margin >= 0.10:
        strength = "Lean"
    else:
        strength = "Toss-Up"

    if strength == "Toss-Up":
        return f"Toss-Up: {best_decision}"
    return f"{strength} {best_decision}"


# Kept for older chart/table code paths that still call make_recommendation.
def make_recommendation(margin):
    return decision_label("Go", margin)


@st.cache_data
def load_data():
    # nflreadpy returns a Polars DataFrame, so convert to pandas
    df = nfl.load_pbp([2023, 2024, 2025]).to_pandas()

    fourth_down = df[df["down"] == 4].copy()

    cols = [
        "season",
        "week",
        "posteam",
        "defteam",
        "yardline_100",
        "ydstogo",
        "yards_gained",
        "fourth_down_converted",
        "play_type",
        "epa",
        "wp",
        "score_differential",
        "quarter_seconds_remaining"
    ]

    fourth_down = fourth_down[cols].dropna(subset=[
        "season", "week", "posteam", "defteam", "yardline_100", "ydstogo",
        "play_type", "epa", "wp", "score_differential"
    ])

    fourth_down["converted"] = (
        fourth_down["fourth_down_converted"]
        .fillna(0)
        .astype(int)
        .eq(1)
    )

    fourth_down["win_probability"] = fourth_down["wp"] * 100
    fourth_down["is_go_attempt"] = fourth_down["play_type"].isin(["pass", "run"])
    fourth_down = bucketize_situation(fourth_down)

    model_features = [
        "distance_exact_bucket",
        "yardline_bucket",
        "play_type",
        "game_situation"
    ]

    league_avg_conversion = fourth_down["converted"].mean()

    league_model = (
        fourth_down
        .groupby(model_features, observed=True)
        .agg(
            raw_expected_conversion_prob=("converted", "mean"),
            sample_size=("converted", "count")
        )
        .reset_index()
    )

    league_model["expected_conversion_prob"] = smooth_rate(
        league_model["raw_expected_conversion_prob"],
        league_model["sample_size"],
        league_avg_conversion
    )

    fourth_down = fourth_down.merge(
        league_model[model_features + ["expected_conversion_prob"]],
        on=model_features,
        how="left"
    )

    fourth_down["expected_conversion_prob"] = fourth_down["expected_conversion_prob"].fillna(
        league_avg_conversion
    )

    # Expected conversion value = expected conversion probability translated into EPA-like value
    league_avg_epa_success = fourth_down.loc[fourth_down["converted"], "epa"].mean()
    league_avg_epa_failure = fourth_down.loc[~fourth_down["converted"], "epa"].mean()

    fourth_down["expected_conversion_value"] = (
        fourth_down["expected_conversion_prob"] * league_avg_epa_success
        + (1 - fourth_down["expected_conversion_prob"]) * league_avg_epa_failure
    )

    fourth_down["epa_over_expected"] = (
        fourth_down["epa"] - fourth_down["expected_conversion_value"]
    )

    decision_features = [
        "distance_exact_bucket",
        "yardline_bucket",
        "game_situation"
    ]

    # -----------------------------
    # True Go model
    # -----------------------------
    # Estimate going for it only from offensive plays, not punts or field goals.
    go_plays = fourth_down[fourth_down["play_type"].isin(["pass", "run"])].copy()

    league_go_conversion = go_plays["converted"].mean()
    league_go_epa = go_plays["epa"].mean()
    league_go_success_epa = go_plays.loc[go_plays["converted"], "epa"].mean()
    league_go_failure_epa = go_plays.loc[~go_plays["converted"], "epa"].mean()

    go_conversion_model = (
        go_plays
        .groupby(decision_features, observed=True)
        .agg(
            raw_go_conversion_prob=("converted", "mean"),
            go_sample_size=("converted", "count"),
            raw_go_epa=("epa", "mean")
        )
        .reset_index()
    )

    go_success_model = (
        go_plays[go_plays["converted"]]
        .groupby(decision_features, observed=True)
        .agg(
            raw_go_success_epa=("epa", "mean"),
            go_success_sample_size=("epa", "count")
        )
        .reset_index()
    )

    go_failure_model = (
        go_plays[~go_plays["converted"]]
        .groupby(decision_features, observed=True)
        .agg(
            raw_go_failure_epa=("epa", "mean"),
            go_failure_sample_size=("epa", "count")
        )
        .reset_index()
    )

    go_model = go_conversion_model.merge(go_success_model, on=decision_features, how="left")
    go_model = go_model.merge(go_failure_model, on=decision_features, how="left")

    go_model["expected_go_conversion_prob"] = smooth_rate(
        go_model["raw_go_conversion_prob"],
        go_model["go_sample_size"],
        league_go_conversion
    )

    go_model["expected_go_success_epa"] = smooth_value(
        go_model["raw_go_success_epa"].fillna(league_go_success_epa),
        go_model["go_success_sample_size"].fillna(0),
        league_go_success_epa
    )

    go_model["expected_go_failure_epa"] = smooth_value(
        go_model["raw_go_failure_epa"].fillna(league_go_failure_epa),
        go_model["go_failure_sample_size"].fillna(0),
        league_go_failure_epa
    )

    go_model["go_for_it_value"] = (
        go_model["expected_go_conversion_prob"] * go_model["expected_go_success_epa"]
        + (1 - go_model["expected_go_conversion_prob"]) * go_model["expected_go_failure_epa"]
    )

    fourth_down = fourth_down.merge(
        go_model[
            decision_features + [
                "expected_go_conversion_prob",
                "expected_go_success_epa",
                "expected_go_failure_epa",
                "go_for_it_value",
                "go_sample_size"
            ]
        ],
        on=decision_features,
        how="left"
    )

    fourth_down["expected_go_conversion_prob"] = fourth_down["expected_go_conversion_prob"].fillna(league_go_conversion)
    fourth_down["expected_go_success_epa"] = fourth_down["expected_go_success_epa"].fillna(league_go_success_epa)
    fourth_down["expected_go_failure_epa"] = fourth_down["expected_go_failure_epa"].fillna(league_go_failure_epa)
    fourth_down["go_for_it_value"] = fourth_down["go_for_it_value"].fillna(league_go_epa)
    fourth_down["go_sample_size"] = fourth_down["go_sample_size"].fillna(0)

    # -----------------------------
    # True Punt model
    # -----------------------------
    punt_model = (
        fourth_down[fourth_down["play_type"] == "punt"]
        .groupby(decision_features, observed=True)
        .agg(
            raw_punt_epa=("epa", "mean"),
            punt_sample_size=("epa", "count")
        )
        .reset_index()
    )

    league_punt_epa = fourth_down.loc[fourth_down["play_type"] == "punt", "epa"].mean()
    punt_model["expected_punt_epa"] = smooth_value(
        punt_model["raw_punt_epa"],
        punt_model["punt_sample_size"],
        league_punt_epa
    )

    fourth_down = fourth_down.merge(
        punt_model[decision_features + ["expected_punt_epa", "punt_sample_size"]],
        on=decision_features,
        how="left"
    )

    fourth_down["expected_punt_epa"] = fourth_down["expected_punt_epa"].fillna(league_punt_epa)
    fourth_down["punt_sample_size"] = fourth_down["punt_sample_size"].fillna(0)

    # -----------------------------
    # True Field Goal model
    # -----------------------------
    fg_features = ["fg_distance_bucket", "game_situation"]

    fg_model = (
        fourth_down[fourth_down["play_type"] == "field_goal"]
        .groupby(fg_features, observed=True)
        .agg(
            raw_fg_epa=("epa", "mean"),
            fg_sample_size=("epa", "count")
        )
        .reset_index()
    )

    league_fg_epa = fourth_down.loc[fourth_down["play_type"] == "field_goal", "epa"].mean()
    fg_model["expected_fg_epa"] = smooth_value(
        fg_model["raw_fg_epa"],
        fg_model["fg_sample_size"],
        league_fg_epa
    )

    fourth_down = fourth_down.merge(
        fg_model[fg_features + ["expected_fg_epa", "fg_sample_size"]],
        on=fg_features,
        how="left"
    )

    fourth_down["fg_is_realistic"] = fourth_down["estimated_fg_distance"] <= 60
    fourth_down["expected_fg_epa"] = fourth_down["expected_fg_epa"].fillna(league_fg_epa)
    fourth_down.loc[~fourth_down["fg_is_realistic"], "expected_fg_epa"] = -999
    fourth_down["fg_sample_size"] = fourth_down["fg_sample_size"].fillna(0)

    # -----------------------------
    # Final go vs punt vs FG decision
    # -----------------------------
    decision_cols = ["go_for_it_value", "expected_punt_epa", "expected_fg_epa"]
    decision_name_map = {
        "go_for_it_value": "Go",
        "expected_punt_epa": "Punt",
        "expected_fg_epa": "Field Goal"
    }

    fourth_down["best_decision_col"] = fourth_down[decision_cols].idxmax(axis=1)
    fourth_down["best_decision"] = fourth_down["best_decision_col"].map(decision_name_map)
    fourth_down["best_decision_value"] = fourth_down[decision_cols].max(axis=1)
    fourth_down["second_best_decision_value"] = fourth_down[decision_cols].apply(
        lambda row: row.sort_values(ascending=False).iloc[1],
        axis=1
    )
    fourth_down["decision_margin"] = (
        fourth_down["best_decision_value"] - fourth_down["second_best_decision_value"]
    )

    # Backward-compatible names for the old dashboard fields.
    fourth_down["alternative_value"] = fourth_down[["expected_punt_epa", "expected_fg_epa"]].max(axis=1)
    fourth_down["best_alternative"] = fourth_down[["expected_punt_epa", "expected_fg_epa"]].idxmax(axis=1).map({
        "expected_punt_epa": "Punt",
        "expected_fg_epa": "Field Goal"
    })
    fourth_down["recommendation_margin"] = fourth_down["go_for_it_value"] - fourth_down["alternative_value"]
    fourth_down["recommendation"] = fourth_down.apply(
        lambda row: decision_label(row["best_decision"], row["decision_margin"]),
        axis=1
    )

    # For conversion-rate metrics, use only real go-for-it attempts.
    # Punts and field goals remain in the dataset for Punt/FG decision modeling,
    # but they should not count as failed conversion attempts.
    fourth_down["expected_conversion_prob"] = fourth_down["expected_go_conversion_prob"]
    fourth_down["expected_conversion_value"] = (
        fourth_down["expected_go_conversion_prob"] * fourth_down["expected_go_success_epa"]
        + (1 - fourth_down["expected_go_conversion_prob"]) * fourth_down["expected_go_failure_epa"]
    )

    fourth_down["conversion_over_expected"] = (
        fourth_down["converted"].astype(int) - fourth_down["expected_conversion_prob"]
    ).where(fourth_down["is_go_attempt"])

    fourth_down["expected_conversion_pct"] = fourth_down["expected_conversion_prob"] * 100

    fourth_down["epa_over_expected"] = (
        fourth_down["epa"] - fourth_down["expected_conversion_value"]
    ).where(fourth_down["is_go_attempt"])

    fourth_down["epa_result"] = fourth_down["epa"].apply(
        lambda x: "Positive EPA" if x >= 0 else "Negative EPA"
    )

    return fourth_down

df = load_data()

st.sidebar.header("Filters")

teams = sorted(df["posteam"].dropna().unique())
selected_teams = st.sidebar.multiselect(
    "Select Team(s)",
    teams,
    default=teams
)

selected_seasons = st.sidebar.multiselect(
    "Select Season(s)",
    sorted(df["season"].unique()),
    default=sorted(df["season"].unique())
)

wp_range = st.sidebar.slider(
    "Win Probability Range (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5
)

comparison_mode = st.sidebar.checkbox("Team Comparison Mode")

if comparison_mode:
    comparison_teams = st.sidebar.multiselect(
        "Teams to Compare",
        selected_teams,
        default=selected_teams[:2]
    )
else:
    comparison_teams = selected_teams

filtered = df[
    (df["posteam"].isin(comparison_teams))
    & (df["season"].isin(selected_seasons))
    & (df["win_probability"].between(wp_range[0], wp_range[1]))
]

# Conversion-rate metrics should only use actual go-for-it attempts.
# Punts and field goals stay in `filtered` for the Go/Punt/FG decision model.
filtered_go = filtered[filtered["is_go_attempt"]].copy()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("Fourth-Down Plays", len(filtered))
kpi2.metric(
    "Conversion Rate",
    f"{filtered_go['converted'].mean():.1%}" if len(filtered_go) else "N/A"
)
kpi3.metric(
    "Expected Conv. Rate",
    f"{filtered_go['expected_conversion_prob'].mean():.1%}" if len(filtered_go) else "N/A"
)
kpi4.metric(
    "Average EPA",
    f"{filtered['epa'].mean():.2f}" if len(filtered) else "N/A"
)
kpi5.metric(
    "EPA Over Expected",
    f"{filtered['epa_over_expected'].mean():+.2f}" if len(filtered) else "N/A"
)

st.caption(
    "Expected conversion probability is estimated from league-wide fourth-down attempts. "
    "Expected conversion value estimates the EPA value of that situation before "
    "comparing it with actual EPA."
)

st.divider()

st.subheader("True Fourth-Down Decision Model")
st.caption(
    "This compares three separate expected EPA choices: go, punt, and field goal. "
    "The recommendation is the option with the highest estimated EPA."
)

rec_col1, rec_col2 = st.columns(2)

selected_distance = rec_col1.slider(
    "Yards to Go",
    min_value=1,
    max_value=20,
    value=4
)

selected_yardline = rec_col2.slider(
    "Yards from Opponent End Zone",
    min_value=1,
    max_value=100,
    value=50
)

similar_plays = filtered[
    (filtered["ydstogo"].between(selected_distance - 1, selected_distance + 1))
    & (filtered["yardline_100"].between(selected_yardline - 10, selected_yardline + 10))
]

if len(similar_plays) > 0:
    decision_values = pd.Series({
        "Go": similar_plays["go_for_it_value"].mean(),
        "Punt": similar_plays["expected_punt_epa"].mean(),
        "Field Goal": similar_plays.loc[similar_plays["fg_is_realistic"], "expected_fg_epa"].mean()
    }).dropna()

    if "Field Goal" not in decision_values.index:
        decision_values.loc["Field Goal"] = -999

    best_decision = decision_values.idxmax()
    best_value = decision_values.max()
    second_best_value = decision_values.sort_values(ascending=False).iloc[1]
    decision_margin = best_value - second_best_value
    recommendation = decision_label(best_decision, decision_margin)

    st.metric("Recommendation", recommendation)

    value_col1, value_col2, value_col3, value_col4 = st.columns(4)
    value_col1.metric("Go Value", f"{decision_values.get('Go', float('nan')):+.2f} EPA")
    value_col2.metric("Punt Value", f"{decision_values.get('Punt', float('nan')):+.2f} EPA")

    fg_value = decision_values.get("Field Goal", float("nan"))
    value_col3.metric(
        "FG Value",
        "Not realistic" if fg_value <= -900 else f"{fg_value:+.2f} EPA"
    )
    value_col4.metric("Decision Edge", f"{decision_margin:+.2f} EPA")

    st.metric(
        "Expected Conversion Rate",
        f"{similar_plays['expected_go_conversion_prob'].mean():.1%}"
    )

    decision_value_df = pd.DataFrame({
        "Decision": ["Go", "Punt", "Field Goal"],
        "Expected EPA": [
            decision_values.get("Go", float("nan")),
            decision_values.get("Punt", float("nan")),
            None if decision_values.get("Field Goal", -999) <= -900 else decision_values.get("Field Goal")
        ]
    }).dropna()

    fig_decision_values = px.bar(
        decision_value_df,
        x="Decision",
        y="Expected EPA",
        text_auto="+.2f",
        title="Estimated EPA by Decision"
    )

    fig_decision_values.update_traces(
        texttemplate="%{y:.2f}",
        hoverinfo="skip",
        hovertemplate=None
    )
    st.plotly_chart(fig_decision_values, width="stretch")

    st.write(
        f"Based on {len(similar_plays)} similar fourth-down plays, the model recommends **{recommendation}**."
    )
else:
    st.warning("Not enough similar plays found for this situation.")

rec_summary = (
    filtered
    .groupby(["distance_bucket", "field_zone", "best_decision"], observed=True)
    .agg(
        attempts=("converted", "count"),
        avg_go_value=("go_for_it_value", "mean"),
        avg_punt_value=("expected_punt_epa", "mean"),
        avg_fg_value=("expected_fg_epa", "mean"),
        avg_decision_margin=("decision_margin", "mean"),
    )
    .reset_index()
)

fig_decision_summary = px.bar(
    rec_summary,
    x="distance_bucket",
    y="attempts",
    color="best_decision",
    facet_col="field_zone",
    title="Recommended Decision Mix by Distance and Field Position",
    labels={
        "distance_bucket": "Distance Bucket",
        "attempts": "Similar Plays",
        "best_decision": "Recommended Decision"
    }
)

fig_decision_summary.for_each_annotation(
    lambda a: a.update(text=a.text.split("=")[-1])
)

fig_decision_summary.update_traces(
        hoverinfo="skip",
        hovertemplate=None
    )
for a in fig_decision_summary.layout.annotations:
    st.write(a.text)
st.plotly_chart(fig_decision_summary, width="stretch")

conversion_color_map = {
    "conversion_rate": "Actual Conversion Rate",
    "expected_conversion_rate": "Expected Conversion Rate"
}

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    if comparison_mode:
        conversion_by_distance = (
            filtered_go
            .groupby(["posteam", "distance_bucket"], observed=True)
            .agg(
                conversion_rate=("converted", "mean"),
                expected_conversion_rate=("expected_conversion_prob", "mean")
            )
            .reset_index()
        )

        distance_long = conversion_by_distance.melt(
            id_vars=["posteam", "distance_bucket"],
            value_vars=["conversion_rate", "expected_conversion_rate"],
            var_name="metric",
            value_name="rate"
        )
        distance_long["metric"] = distance_long["metric"].map(conversion_color_map)
        distance_long["series"] = distance_long["posteam"] + " — " + distance_long["metric"]

        fig_distance_conversion = px.bar(
            distance_long,
            x="distance_bucket",
            y="rate",
            color="series",
            barmode="group",
            title="Actual vs Expected Conversion Rate by Distance — Team Comparison",
            labels={
                "distance_bucket": "Distance Bucket",
                "rate": "Conversion Rate",
                "series": "Team / Metric"
            },
            text_auto=".1%"
        )
        
    else:
        conversion_by_distance = (
            filtered_go
            .groupby("distance_bucket", observed=True)
            .agg(
                conversion_rate=("converted", "mean"),
                expected_conversion_rate=("expected_conversion_prob", "mean")
            )
            .reset_index()
        )

        distance_long = conversion_by_distance.melt(
            id_vars="distance_bucket",
            value_vars=["conversion_rate", "expected_conversion_rate"],
            var_name="metric",
            value_name="rate"
        )
        distance_long["metric"] = distance_long["metric"].map(conversion_color_map)

        fig_distance_conversion = px.bar(
            distance_long,
            x="distance_bucket",
            y="rate",
            color="metric",
            barmode="group",
            title="Actual vs Expected Conversion Rate by Distance",
            labels={
                "distance_bucket": "Distance Bucket",
                "rate": "Conversion Rate",
                "metric": "Metric"
            },
            text_auto=".1%"
        )

    fig_distance_conversion.update_traces(
        hoverinfo="skip",
        hovertemplate=None
    )
    fig_distance_conversion.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_distance_conversion, width="stretch")

with row1_col2:
    if comparison_mode:
        conversion_by_field = (
            filtered_go
            .groupby(["posteam", "field_zone"], observed=True)
            .agg(
                conversion_rate=("converted", "mean"),
                expected_conversion_rate=("expected_conversion_prob", "mean")
            )
            .reset_index()
        )

        field_conversion_long = conversion_by_field.melt(
            id_vars=["posteam", "field_zone"],
            value_vars=["conversion_rate", "expected_conversion_rate"],
            var_name="metric",
            value_name="rate"
        )
        field_conversion_long["metric"] = field_conversion_long["metric"].map(conversion_color_map)
        field_conversion_long["series"] = field_conversion_long["posteam"] + " — " + field_conversion_long["metric"]

        fig_field_conversion = px.bar(
            field_conversion_long,
            x="field_zone",
            y="rate",
            color="series",
            barmode="group",
            title="Actual vs Expected Conversion Rate by Field Position — Team Comparison",
            labels={
                "field_zone": "Field Zone",
                "rate": "Conversion Rate",
                "series": "Team / Metric"
            },
            text_auto=".1%"
        )
    else:
        conversion_by_field = (
            filtered_go
            .groupby("field_zone", observed=True)
            .agg(
                conversion_rate=("converted", "mean"),
                expected_conversion_rate=("expected_conversion_prob", "mean")
            )
            .reset_index()
        )

        field_conversion_long = conversion_by_field.melt(
            id_vars="field_zone",
            value_vars=["conversion_rate", "expected_conversion_rate"],
            var_name="metric",
            value_name="rate"
        )
        field_conversion_long["metric"] = field_conversion_long["metric"].map(conversion_color_map)

        fig_field_conversion = px.bar(
            field_conversion_long,
            x="field_zone",
            y="rate",
            color="metric",
            barmode="group",
            title="Actual vs Expected Conversion Rate by Field Position",
            labels={
                "field_zone": "Field Zone",
                "rate": "Conversion Rate",
                "metric": "Metric"
            },
            text_auto=".1%"
        )

    fig_field_conversion.update_traces(
        hoverinfo="skip",
        hovertemplate=None
    )
    fig_field_conversion.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_field_conversion, width="stretch")

st.divider()

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    epa_by_distance = (
        filtered_go
        .groupby(["posteam", "distance_bucket"], observed=True) if comparison_mode
        else filtered_go.groupby("distance_bucket", observed=True)
    ).agg(
        epa_over_expected=("epa_over_expected", "mean")
    ).reset_index()

    fig_distance_epa = px.bar(
        epa_by_distance,
        x="distance_bucket",
        y="epa_over_expected",
        color="posteam" if comparison_mode else None,
        barmode="group" if comparison_mode else "relative",
        title="EPA Over Expected by Distance",
        labels={
            "distance_bucket": "Distance Bucket",
            "go_for_it_value": "Go EPA",
        "expected_punt_epa": "Punt EPA",
        "expected_fg_epa": "FG EPA",
        "decision_margin": "Decision Edge",
        "epa_over_expected": "EPA Over Expected",
            "posteam": "Team"
        },
        text_auto=".2f"
    )

    fig_distance_epa.update_traces(
        hoverinfo="skip",
        hovertemplate=None
    )
    st.plotly_chart(fig_distance_epa, width="stretch")

with row2_col2:
    epa_by_field = (
        filtered_go
        .groupby(["posteam", "field_zone"], observed=True) if comparison_mode
        else filtered_go.groupby("field_zone", observed=True)
    ).agg(
        epa_over_expected=("epa_over_expected", "mean")
    ).reset_index()

    fig_field_epa = px.bar(
        epa_by_field,
        x="field_zone",
        y="epa_over_expected",
        color="posteam" if comparison_mode else None,
        barmode="group" if comparison_mode else "relative",
        title="EPA Over Expected by Field Position",
        labels={
            "field_zone": "Field Zone",
            "go_for_it_value": "Go EPA",
        "expected_punt_epa": "Punt EPA",
        "expected_fg_epa": "FG EPA",
        "decision_margin": "Decision Edge",
        "epa_over_expected": "EPA Over Expected",
            "posteam": "Team"
        },
        text_auto=".2f"
    )

    fig_field_epa.update_traces(
        hoverinfo="skip",
        hovertemplate=None
    )
    st.plotly_chart(fig_field_epa, width="stretch")

st.divider()

team_summary = (
    filtered_go
    .groupby("posteam")
    .agg(
        attempts=("converted", "count"),
        conversion_rate=("converted", "mean"),
        expected_conversion_rate=("expected_conversion_prob", "mean"),
        conversion_over_expected=("conversion_over_expected", "mean"),
        avg_epa=("epa", "mean"),
        expected_conversion_value=("expected_conversion_value", "mean"),
        go_for_it_value=("go_for_it_value", "mean"),
        expected_punt_epa=("expected_punt_epa", "mean"),
        expected_fg_epa=("expected_fg_epa", "mean"),
        decision_margin=("decision_margin", "mean"),
        epa_over_expected=("epa_over_expected", "mean"),
        avg_win_probability=("win_probability", "mean")
    )
    .reset_index()
    #.sort_values("conversion_rate", ascending=False)
)

team_summary["conversion_over_expected_pct"] = team_summary["conversion_over_expected"] * 100
team_summary["epa_size"] = team_summary["avg_epa"].abs() + 0.01
team_summary = team_summary.sort_values("epa_over_expected", ascending=False)

fig_team = px.scatter(
    team_summary,
    x="expected_conversion_rate",
    y="conversion_rate",
    size="epa_size",
    color="posteam" if comparison_mode else "epa_over_expected",
    hover_name="posteam",
    hover_data={
        "attempts": True,
        "expected_conversion_rate": ":.1%",
        "conversion_rate": ":.1%",
        "avg_epa": ":.3f",
        "epa_over_expected": ":.3f",
        "epa_size": False
    },
    title="Actual Conversion Rate vs Expected Conversion Rate",
    labels={
        "expected_conversion_rate": "Expected Conversion Rate",
        "conversion_rate": "Actual Conversion Rate",
        "epa_over_expected": "EPA Over Expected",
        "attempts": "Attempts",
        "avg_epa": "Average EPA"
    }
)

fig_team.update_xaxes(tickformat=".0%")
fig_team.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_team, width="stretch")

st.divider()

fig_scatter = px.scatter(
    filtered,
    x="ydstogo",
    y="yardline_100",
    color="posteam" if comparison_mode else "epa_result",
    symbol="epa_result" if comparison_mode else "converted",
    hover_data={
        "season": True,
        "week": True,
        "posteam": True,
        "defteam": True,
        "play_type": True,
        "yards_gained": True,
        "epa": ":.2f"
    },
    title="Fourth Down Attempts by Distance and Field Position",
    labels={
        "ydstogo": "Yards to Go",
        "yardline_100": "Yards from End Zone",
        "epa_result": "EPA Result",
        "converted": "Converted"
    }
)

st.plotly_chart(fig_scatter, width="stretch")

st.subheader("Team Summary Table")

display_table = team_summary.drop(columns=["epa_size"])

display_table["conversion_rate"] *= 100
display_table["expected_conversion_rate"] *= 100

display_table = display_table[[
    "posteam",
    "attempts",
    "conversion_rate",
    "expected_conversion_rate",
    "conversion_over_expected_pct",
    "avg_epa",
    "expected_conversion_value",
    "epa_over_expected",
    "avg_win_probability"
]]

st.dataframe(
    display_table,
    width="stretch",
    column_config={
        "posteam": "Team",
        "attempts": "Go Attempts",
        "conversion_rate": st.column_config.NumberColumn("Conversion Rate", format="%.1f%%"),
        "expected_conversion_rate": st.column_config.NumberColumn("Expected Conversion Rate", format="%.1f%%"),
        "conversion_over_expected_pct": st.column_config.NumberColumn("Conv. Over Expected", format="%+.1f pts"),
        "avg_epa": st.column_config.NumberColumn("Avg. EPA", format="%.3f"),
        "expected_conversion_value": st.column_config.NumberColumn("Expected Conversion Value", format="%.3f"),
        "go_for_it_value": st.column_config.NumberColumn("Go EPA", format="%+.3f"),
        "expected_punt_epa": st.column_config.NumberColumn("Punt EPA", format="%+.3f"),
        "expected_fg_epa": st.column_config.NumberColumn("FG EPA", format="%+.3f"),
        "decision_margin": st.column_config.NumberColumn("Decision Edge", format="%+.3f"),
        "epa_over_expected": st.column_config.NumberColumn("EPA Over Expected", format="%+.3f"),
        "avg_win_probability": st.column_config.NumberColumn("Avg. Win Probability", format="%.1f%%")
    }
)
