# /// script
# dependencies = [
#     "joblib==1.5.3",
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==3.0.0",
#     "scikit-learn==1.8.0",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ### We want to build a Admit to Discharge LOS Predictor, based on ED stay and num times prev admitted last 90 days
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pickle
    import json
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GroupShuffleSplit
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import QuantileRegressor

    from sklearn.pipeline import Pipeline
    return (
        ColumnTransformer,
        GroupShuffleSplit,
        OneHotEncoder,
        Pipeline,
        QuantileRegressor,
        json,
        mo,
        np,
        pd,
        pickle,
        plt,
    )


@app.cell
def _(json, pd):
    adt_df = pd.read_csv('../data/ADT_cleaned_with_tiers.csv')
    fitted_distributions_df = pd.read_csv('../data/states_fitted_distribution_df.csv')
    arrival_rates_df = pd.read_csv('../data/weekday_weekend_arrival_rates_df.csv')
    loc_trans_prob_df = pd.read_csv('../data/probability_matrix.csv')

    with open('../data/trajectory_library_v2.json', 'r') as json_file:
        trajectory_library = json.load(json_file)
    return (trajectory_library,)


@app.cell
def _():
    # for e_id, traj in trajectory_library.items():
    #     for step in traj:
    #         step["time_in"] = pd.to_datetime(step["time_in"])
    #         step["time_out"] = pd.to_datetime(step["time_out"])
    return


@app.cell
def _(trajectory_library):
    trajectory_library['66216876027']
    return


@app.cell
def _(pd, trajectory_library):
    rows = []
    for eid, path in trajectory_library.items():
        enc_id = eid
        for state in path:
            rows.append({
            "eid": eid, 
            "pat_id": state['pat_id'],
            "state": state['state'], 
            "time_in": state['time_in'], 
            "time_out": state['time_out'], 
            "duration": state['duration_hours']
            })

    traj_lib_df = pd.DataFrame(rows).sort_values(by = ['eid', 'time_in'])
    traj_lib_df['hospital'] = traj_lib_df['state'].str.split('_').str[1:].str.join('_')
    traj_lib_df['time_in'] = pd.to_datetime(traj_lib_df["time_in"])
    traj_lib_df['time_out'] = pd.to_datetime(traj_lib_df["time_out"])
    traj_lib_df
    return (traj_lib_df,)


@app.cell
def _(mo):
    mo.md(r"""
    #Preparing Training Df
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Who counts as admitted


    An encounter is considered admitted if the patient is not in ED .

    Patients who:

    - Transfer to ICU or Floor
    - Are boarded in the ED
    are defined as admitted patients
    """)
    return


@app.cell
def _(traj_lib_df):
    # We only want to look at admitted encounters
    # Group by encounter
    admitted_eids = []

    for eid3, group3 in traj_lib_df.groupby('eid'):
        # Check if any state beyond the first is floor or ICU
        all_states = group3['state'] 
        if all_states.str.contains('ICU|FLOOR|BOARDED').any():
            admitted_eids.append(eid3)

    # Filter traj_lib_df to only admitted encounters
    traj_lib_admitted = traj_lib_df[traj_lib_df['eid'].isin(admitted_eids)]
    traj_lib_admitted


    # get hospital the patient was admitted at (hospital at first non-ed row)
    non_ed = traj_lib_admitted[~traj_lib_admitted['state'].str.startswith('ED')].copy()
    first_inpatient = (
        non_ed
        .sort_values(['eid', 'time_in'])
        .groupby('eid')
        .first()
        .reset_index()
    )

    first_inpatient['admit_hospital'] = (
        first_inpatient['state']
        .str.split('_')
        .str[1:]
        .str.join('_')
    )

    traj_lib_admitted = traj_lib_admitted.merge(
        first_inpatient[['eid', 'admit_hospital']],
        on='eid',
        how='left'
    )
    return (traj_lib_admitted,)


@app.cell
def _(traj_lib_admitted):
    traj_lib_admitted
    return


@app.cell
def _(traj_lib_admitted):
    traj_lib_admitted[traj_lib_admitted['eid']=='66210969537']
    return


@app.cell
def _(pd, traj_lib_df):
    # function to find number of times previously come to hosp

    def find_num_prev_admitted(eid):
        pat_id = traj_lib_df[traj_lib_df['eid'] == eid]['pat_id'].iloc[0]
        first_enc_time = traj_lib_df[traj_lib_df['eid']== eid]['time_in'].min()

        three_months_ago = first_enc_time - pd.Timedelta(days=90)

       # Filter for same patient, admissions before this encounter, within last 3 months
        recent_prev_admissions = traj_lib_df[
                                            (traj_lib_df['pat_id'] == pat_id) &
                                            (traj_lib_df['time_in'] >= three_months_ago) &
                                            (traj_lib_df['time_in'] < first_enc_time)
                                        ]['eid'].nunique()

        return recent_prev_admissions
    return


@app.cell
def _(pd, traj_lib_admitted):

    # Define 90-day window
    window_days = 90
    window_timedelta = pd.Timedelta(days=window_days)
    traj_lib_admitted['time_in'] = pd.to_datetime(traj_lib_admitted['time_in'])


    # Initialize a column for previous admissions in the last 90 days
    first_encounter_per_eid = (
        traj_lib_admitted
        .groupby('eid')
        .agg({
            'pat_id': 'first',
            'time_in': 'min'  # encounter start time
        })
        .reset_index()
        .sort_values(['pat_id', 'eid', 'time_in'])
    )

    first_encounter_per_eid['num_prev_90_days'] = 0


    # # Group by patient

    for pat_id, group in first_encounter_per_eid.groupby('pat_id'):
        times = group['time_in'].values

        prev_counts = []
        start_idx = 0

        for idx, t in enumerate(times):
            while times[start_idx] < t - window_timedelta:
                start_idx += 1
            prev_counts.append(idx - start_idx)

        first_encounter_per_eid.loc[group.index, 'num_prev_90_days'] = prev_counts

    def compute_los_features(g):
        g = g.sort_values('time_in')

        total_los = g['duration'].sum()

        # Find first inpatient index
        inpatient_mask = g['state'].str.contains("ICU|FLOOR|BOARDED")

        if inpatient_mask.any():
            first_inpatient_idx = inpatient_mask.idxmax()

            # All rows before that are ED time
            ed_time = g.loc[:first_inpatient_idx - 1, 'duration'].sum()

            admit_los = g.loc[first_inpatient_idx:, 'duration'].sum()
        else:
            ed_time = g[(g['state'].str.contains("ED")) & (~g['state'].str.contains("BOARDED"))]['duration'].sum()
            admit_los = 0

        return pd.Series({
            'pat_id': g['pat_id'].iloc[0],
            'total_los_hours': total_los,
            'time_in_ed': ed_time,
            'admit_to_discharge_los_hours': admit_los,
            'admit_hospital': g['admit_hospital'].iloc[0]
        })

    los_features = (
        traj_lib_admitted
        .groupby('eid')
        .apply(compute_los_features)
        .reset_index()
    )


    # Now you can build training_df_rows in one go without per-eid filtering
    # training_df_rows = []

    # for i, group in traj_lib_admitted.groupby('eid'):
    #     time_in_ed = 0
    #     total_los_hours = group['duration'].sum()
    #     admit_to_discharge_los_hours = total_los_hours
    #     hopsital_admitted_at = group['admit_hospital'].iloc[0]


    #     if "ED" in group['state'].iloc[0]:
    #         time_in_ed = group['duration'].iloc[0]
    #         admit_to_discharge_los_hours = admit_to_discharge_los_hours - time_in_ed
    #         # hopsital_admitted_at = group['admit_hospital'].iloc[1]



    #     training_df_rows.append({
    #         'eid': i,
    #         'pat_id': group['pat_id'].iloc[0],
    #         'time_in_ed': time_in_ed,
    #         'admit_to_discharge_los_hours': admit_to_discharge_los_hours,
    #         'total_los_hours': total_los_hours,
    #         'admit_hospital': hopsital_admitted_at,
    #     })
    return first_encounter_per_eid, los_features


@app.cell
def _(los_features):
    los_features
    return


@app.cell
def _():
    # traj_lib_df_with_hosp = traj_lib_df.merge(
    #     first_inpatient[['eid', 'admit_hospital']],
    #     on='eid',
    #     how='left'
    # )

    # collapsed_traj_lib_df = (
    # traj_lib_df_with_hosp
    #     .groupby('eid')
    #     .apply(compute_los_features)
    #     .reset_index()
    # )
    # collapsed_traj_lib_df
    return


@app.cell
def _(first_encounter_per_eid, los_features):
    training_df_temp = los_features.merge(first_encounter_per_eid[['eid', 'num_prev_90_days']], on = 'eid', how = 'left')
    training_df_temp
    return (training_df_temp,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Median Admit to Discharge LOS per Hospital
    """)
    return


@app.cell
def _(training_df_temp):
    training_df_temp['total_los_hours'].describe()
    return


@app.cell
def _(training_df_temp):
    training_df_temp.groupby('admit_hospital')['admit_to_discharge_los_hours'].median().reset_index()#.to_csv('median_admit_to_discharge_los.csv')
    return


@app.cell
def _(training_df_temp):
    training_df_temp['eid'].nunique()==training_df_temp.shape[0]
    return


@app.cell
def _():
    # training_df_temp.to_csv('los_predictor_training_df.csv')
    return


@app.cell
def _(mo):
    mo.md(r"""
    #Building QuantReg Predictor
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Filter for Admitted Encounters

    Our original dataset contains **all encounters**, including patients who only visited the ED and were discharged without admission.

    For predicting **admit-to-discharge length of stay**, these ED-only encounters are not relevant because they have zero post-ED duration. Including them skews the data and causes models (especially the median/50th percentile) to predict 0 for everyone.

    To address this, we filter the dataset to only include encounters with a **positive admit-to-discharge LOS**:

    - `admit_to_discharge_los_hours > 0`

    This ensures our models are trained on **true admissions**, giving meaningful predictions for LOS beyond the ED.
    """)
    return


@app.cell
def _():
    # test_df.to_csv('los_predictor_training_df.csv')
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(np, pd):
    training_df = pd.read_csv('../data/los_predictor_training_df.csv', index_col=0)
    training_df['log_admit_discharge_los'] = np.log1p(training_df["admit_to_discharge_los_hours"])
    training_df
    return (training_df,)


@app.cell
def _(GroupShuffleSplit, training_df):
    # Features and target
    X = training_df[["time_in_ed", "num_prev_90_days", 'admit_hospital']]
    y = training_df["log_admit_discharge_los"]
    groups = training_df["pat_id"]  # group by patient

    # Use GroupShuffleSplit to split by patient
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_test, y_test


@app.cell
def _(ColumnTransformer, OneHotEncoder, Pipeline, QuantileRegressor):
    CAT_FEATURES = ["admit_hospital"]
    NUM_FEATURES = [
        "time_in_ed",
        "num_prev_90_days"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CAT_FEATURES
            ),
            (
                "num",
                "passthrough",
                NUM_FEATURES
            )
        ]
    )

    def make_quantile_pipeline(q):
        return Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    QuantileRegressor(
                        quantile=q,
                        alpha=0.0,   # no regularization unless needed
                        solver="highs"
                    )
                )
            ]
        )


    #quantiles = [0.25, 0.50, 0.75]
    #models = {}

    #for q in quantiles:
    #    pipe = make_quantile_pipeline(q)
    #    pipe.fit(X_train, y_train)   # y_train is log LOS
    #    models[q] = pipe
    return


@app.cell
def _(pickle):
    with open('../model/los_quantile_models.pkl', 'rb') as f:
        models = pickle.load(f)

    print("Models loaded:", list(models.keys()))
    return (models,)


@app.cell
def _(training_df):
    training_df['time_in_ed'].plot(kind = 'hist')
    return


@app.cell
def _(training_df):
    training_df['num_prev_90_days'].plot(kind = 'hist')
    return


@app.cell
def _(training_df):
    training_df['log_admit_discharge_los'].plot(kind = 'hist')
    # print(training_df['log_admit_discharge_los'].describe())
    return


@app.cell
def _(mo):
    mo.md(r"""
    # EVALUATING
    """)
    return


@app.cell
def _(X_test, models, np, y_test):
    quantiles = [0.25, 0.50, 0.75]

    def pinball_loss(y, yhat, q):
        return np.mean(np.maximum(q*(y-yhat), (q-1)*(y-yhat)))

    pinball_losses = {}

    for quantile in quantiles:
        y_pred = models[quantile].predict(X_test)
        loss = pinball_loss(y_test, y_pred, quantile)
        pinball_losses[quantile] = loss

    # Convert to hours (approximate)
    for q1, loss1 in pinball_losses.items():
        approx_error_hours = np.expm1(loss1)
        print(f"Quantile {q1}: pinball loss = {loss1:.3f}")
    return (pinball_losses,)


@app.cell
def _(mo, np, pinball_losses):
    mo.md(rf"""
    The model predicts log-transformed hospital LOS using just ED time and recent admission history (last 90 days). Pinball loss for the 25th, 50th, and 75th percentiles corresponds to errors of approximately {np.expm1(pinball_losses[0.25]):.3f}, {np.expm1(pinball_losses[0.5]):.3f}, and {np.expm1(pinball_losses[0.75]):.3f} hours, respectively.
    """)
    return


@app.cell
def _(X_test, models):
    y_pred_25th = models[0.25].predict(X_test)
    y_pred_50th = models[0.5].predict(X_test)
    y_pred_75th = models[0.75].predict(X_test)
    return y_pred_25th, y_pred_50th, y_pred_75th


@app.cell
def _(np, y_pred_25th, y_pred_50th, y_pred_75th, y_test):
    # Convert predictions back to hours
    y_test_hours = np.expm1(y_test)
    # y_pred_50_hours = np.expm1(models[0.5].predict(X_test))

    mae_hours_25th = np.mean(np.abs(y_test_hours - y_pred_25th))
    mae_hours_50th = np.mean(np.abs(y_test_hours - y_pred_50th))
    mae_hours_75th = np.mean(np.abs(y_test_hours - y_pred_75th))


    print(f"25th Q model MAE in hours: {mae_hours_25th:.2f}")
    print(f"Median model MAE in hours: {mae_hours_50th:.2f}")
    print(f"75th Q model MAE in hours: {mae_hours_75th:.2f}")
    return


@app.cell
def _(training_df):
    training_df["admit_to_discharge_los_hours"].describe(percentiles=[0.5, 0.75, 0.9, 0.95])
    return


@app.cell
def _(np, y_pred_25th, y_pred_50th, y_pred_75th, y_test):
    frac_below = {
        0.25: np.mean(y_test <= y_pred_25th),
        0.50: np.mean(y_test <= y_pred_50th),
        0.75: np.mean(y_test <= y_pred_75th)
    }

    for qs, frac in frac_below.items():
        print(f"Fraction of true LOS ≤ predicted {int(qs*100)}th percentile: {frac:.3f}")
    return (frac_below,)


@app.cell
def _(plt, y_pred_50th, y_test):
    #checking dist
    plt.figure(figsize=(8,5))

    # Plot predicted values
    plt.hist(y_pred_50th, bins=30, alpha=0.6, label='Predicted (50th pct)', edgecolor='k')

    # Plot actual values
    plt.hist(y_test, bins=30, alpha=0.4, label='Actual', edgecolor='k')

    plt.xlabel('Log-admit-to-discharge LOS')
    plt.ylabel('Count')
    plt.title('Predicted vs Actual LOS (Histogram)')
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, y_pred_75th, y_test):
    plt.scatter(y_pred_75th, y_test, alpha=0.2)
    plt.xlabel("Predicted log LOS (q=0.75)")
    plt.ylabel("Actual log LOS")
    return


@app.cell
def _(plt, y_pred_50th, y_test):
    plt.scatter(y_pred_50th, y_test, alpha=0.2)
    plt.xlabel("Predicted log LOS (q=0.5)")
    plt.ylabel("Actual log LOS")
    return


@app.cell
def _(frac_below, mo):
    mo.md(rf"""
    For the predicted quantiles, we can check how well they align with the observed data. Specifically, for each predicted quantile 
    𝑞
    q, the fraction of actual outcomes below the predicted value should roughly equal 
    𝑞
    q if the model is well-calibrated.

    25th percentile: {(frac_below[0.25])*100:.2f}% of actual LOS values are below the predicted 25th percentile – very close to the expected 25%.

    50th percentile: {(frac_below[0.50])*100:.2f}% of actual LOS values are below the predicted median – almost exactly as expected.

    75th percentile: {(frac_below[0.75])*100:.2f}% of actual LOS values are below the predicted 75th percentile – again, close to the expected 75%.

    This shows that the model’s predicted quantiles are well-calibrated and capture the distribution of hospital LOS accurately.
    """)
    return


@app.cell
def _(traj_lib_df):
    traj_lib_df.groupby('state')['duration'].mean()#.to_csv('average_duration_per_state.csv')
    return


@app.cell
def _(X_test, np, plt, y_pred_25th, y_pred_50th, y_pred_75th, y_test):
    # Build eval_df
    eval_df = X_test.copy()
    eval_df['y_true_log'] = y_test.values
    eval_df['y_true_hrs'] = np.expm1(y_test.values)
    eval_df['y_pred_25_log'] = y_pred_25th
    eval_df['y_pred_50_log'] = y_pred_50th
    eval_df['y_pred_75_log'] = y_pred_75th
    eval_df['y_pred_50_hrs'] = np.expm1(y_pred_50th)

    resid_campus_colors = {'LA_JOLLA': '#1f77b4', 'HILLCREST': '#ff7f0e', 'EAST_CAMPUS': '#2ca02c'}
    resid_fig, resid_axes = plt.subplots(1, 3, figsize=(15, 5))

    for resid_ax, (resid_q, resid_pred_col) in zip(resid_axes, [
        (0.25, 'y_pred_25_log'), (0.50, 'y_pred_50_log'), (0.75, 'y_pred_75_log')
    ]):
        for resid_campus, resid_color in resid_campus_colors.items():
            resid_sub = eval_df[eval_df['admit_hospital'] == resid_campus]
            resid_vals = resid_sub['y_true_log'].values - resid_sub[resid_pred_col].values
            resid_ax.scatter(resid_sub[resid_pred_col].values, resid_vals, alpha=0.15, s=8,
                             color=resid_color, label=resid_campus)
        resid_ax.axhline(0, color='red', linewidth=1.2, linestyle='--')
        resid_ax.set_xlabel(f'Predicted log LOS (q={resid_q})')
        resid_ax.set_ylabel('Residual (Actual - Predicted)')
        resid_ax.set_title(f'Residuals — q={resid_q}')
        if resid_q == 0.25:
            resid_ax.legend(markerscale=3)

    plt.suptitle('Residual Plots by Quantile and Campus', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    return (eval_df,)


@app.cell
def _(eval_df, np):
    # Per-campus calibration check
    print(f"{'Campus':<15} {'q=0.25':>8} {'q=0.50':>8} {'q=0.75':>8} {'MAE(hrs)':>10}")
    print("-" * 50)
    for campus in ['LA_JOLLA', 'HILLCREST', 'EAST_CAMPUS']:
        mask = eval_df['admit_hospital'] == campus
        sub = eval_df[mask]
        f25 = np.mean(sub['y_true_log'].values <= sub['y_pred_25_log'].values)
        f50 = np.mean(sub['y_true_log'].values <= sub['y_pred_50_log'].values)
        f75 = np.mean(sub['y_true_log'].values <= sub['y_pred_75_log'].values)
        mae = np.mean(np.abs(sub['y_true_hrs'].values - np.expm1(sub['y_pred_50_log'].values)))
        print(f"{campus:<15} {f25:>8.3f} {f50:>8.3f} {f75:>8.3f} {mae:>9.1f}hr")
    return


@app.cell
def _(eval_df, plt):
    campus_list2 = ['LA_JOLLA', 'HILLCREST', 'EAST_CAMPUS']
    quantile_list2 = [(0.25, 'y_pred_25_log'), (0.50, 'y_pred_50_log'), (0.75, 'y_pred_75_log')]
    campus_colors2 = {'LA_JOLLA': '#1f77b4', 'HILLCREST': '#ff7f0e', 'EAST_CAMPUS': '#2ca02c'}

    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))

    for row2, campus2 in enumerate(campus_list2):
        sub2 = eval_df[eval_df['admit_hospital'] == campus2]
        for col2, (q2, pred_col2) in enumerate(quantile_list2):
            ax2 = axes2[row2, col2]
            residuals2 = sub2['y_true_log'].values - sub2[pred_col2].values
            ax2.scatter(sub2[pred_col2].values, residuals2,
                        alpha=0.15, s=6, color=campus_colors2[campus2])
            ax2.axhline(0, color='red', linewidth=1.2, linestyle='--')
            ax2.set_title(f'{campus2} — q={q2}')
            ax2.set_xlabel('Predicted log LOS')
            ax2.set_ylabel('Residual (Actual − Predicted)')

    plt.suptitle('Residuals by Campus and Quantile', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        'residual_plots_by_quantile_campus.pdf',
        bbox_inches='tight',
        facecolor='white'
    )
    return


@app.cell
def _(plt):
    plt.savefig('residual_plots_by_quantile_campus.png', dpi=300, bbox_inches='tight')
    plt.show()
    return


@app.cell
def _(eval_df, np, plt):
    ba_campus_colors = {'LA_JOLLA': '#1f77b4', 'HILLCREST': '#ff7f0e', 'EAST_CAMPUS': '#2ca02c'}
    ba_fig, ba_axes = plt.subplots(1, 3, figsize=(15, 5))

    for ba_ax, (ba_q, ba_pred_col) in zip(ba_axes, [
        (0.25, 'y_pred_25_log'), (0.50, 'y_pred_50_log'), (0.75, 'y_pred_75_log')
    ]):
        ba_actual = eval_df['y_true_log'].values
        ba_predicted = eval_df[ba_pred_col].values

        ba_mean = (ba_actual + ba_predicted) / 2
        ba_diff = ba_actual - ba_predicted

        ba_md = np.mean(ba_diff)
        ba_sd = np.std(ba_diff)
        ba_loa_upper = ba_md + 1.96 * ba_sd
        ba_loa_lower = ba_md - 1.96 * ba_sd

        for ba_campus, ba_color in ba_campus_colors.items():
            ba_mask = eval_df['admit_hospital'] == ba_campus
            ba_ax.scatter(ba_mean[ba_mask], ba_diff[ba_mask],
                          alpha=0.15, s=8, color=ba_color, label=ba_campus)

        ba_ax.axhline(ba_md,        color='red',  linewidth=1.5, linestyle='-',
                      label=f'Bias: {ba_md:.1f}hr')
        ba_ax.axhline(ba_loa_upper, color='gray', linewidth=1.2, linestyle='--',
                      label=f'+1.96SD: {ba_loa_upper:.1f}hr')
        ba_ax.axhline(ba_loa_lower, color='gray', linewidth=1.2, linestyle='--',
                      label=f'-1.96SD: {ba_loa_lower:.1f}hr')

        ba_ax.set_xlabel('Mean of Actual & Predicted log LOS')
        ba_ax.set_ylabel('Actual − Predicted log LOS')
        ba_ax.set_title(f'Bland-Altman — q={ba_q}')
        ba_ax.legend(fontsize=7, markerscale=3)


    plt.suptitle('Bland-Altman Plots by Quantile', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(
    EC_combined_usage_df,
    EC_queue_events,
    HC_combined_usage_df,
    HC_queue_events,
    LJ_combined_usage_df,
    LJ_queue_events,
):
    def build_section_34_figures():
        import os
        import numpy as numpy34
        import matplotlib.pyplot as plt34
        import seaborn as sns34
        from scipy.stats import gaussian_kde

        os.makedirs("figures", exist_ok=True)

        KEEP = ["Baseline", "3hr", "8hr", "12hr"]
        PALETTE = {
            "Baseline": "#d62728",
            "3hr":      "#9ecae1",
            "8hr":      "#4292c6",
            "12hr":     "#08519c",
        }
        HUE_ORDER = ["Baseline", "3hr", "8hr", "12hr"]
        CAMPUS_COLORS = {
            "La Jolla":    "#2171b5",
            "Hillcrest":   "#238b45",
            "East Campus": "#d62728",
        }
        THRESH_X = {"Baseline": 0, "3hr": 3, "8hr": 8, "12hr": 12}

        # ── Figure 1: Time-series overlays with capacity annotation ─────────────
        def overlay(combined_usage_df, source_label, campus_key, save_path):
            resources   = [f"BOARDED_{campus_key}", f"ICU_{campus_key}"]
            unit_labels = ["Boarding Unit", "ICU"]
            # Capacity values from g.capacities
            cap_map = {
                "BOARDED_LA_JOLLA": 49, "ICU_LA_JOLLA": 57,
                "BOARDED_HILLCREST": 10, "ICU_HILLCREST": 38,
                "BOARDED_EAST_CAMPUS": 4, "ICU_EAST_CAMPUS": 8,
                "FLOOR_LA_JOLLA": 348, "FLOOR_HILLCREST": 218, "FLOOR_EAST_CAMPUS": 82,
            }
            df = combined_usage_df[combined_usage_df["threshold"].isin(KEEP)].copy()

            fig, axes = plt34.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(
                f"Rerouted from {source_label}: Mean Beds In Use Over Time\n"
                f"by LOS Transfer Threshold", fontsize=12
            )
            for ax, resource, unit_label in zip(axes, resources, unit_labels):
                sub = df[df["resource"] == resource]
                if sub.empty:
                    ax.set_title(f"{unit_label} (no data)"); continue
                sns34.lineplot(
                    data=sub, x="day", y="util_smooth",
                    hue="threshold", hue_order=HUE_ORDER,
                    palette=PALETTE, ax=ax, linewidth=1.8
                )
                # 80% capacity reference line + annotation
                cap = cap_map.get(resource)
                if cap:
                    threshold_80 = cap * 0.8
                    ax.axhline(threshold_80, color="black", linestyle="--",
                               linewidth=1.0, zorder=5)
                    ax.text(2, threshold_80 + 0.3, f"80% capacity ({threshold_80:.0f} beds)",
                            fontsize=8, color="black", va="bottom")
                    # Annotate baseline peak
                    baseline_sub = sub[sub["threshold"] == "Baseline"]
                    if not baseline_sub.empty:
                        peak_val = baseline_sub["util_smooth"].max()
                        peak_day = baseline_sub.loc[
                            baseline_sub["util_smooth"].idxmax(), "day"]
                        ax.annotate(f"Baseline peak\n{peak_val:.0f} beds",
                                    xy=(peak_day, peak_val),
                                    xytext=(peak_day + 10, peak_val + 1),
                                    fontsize=8, color=PALETTE["Baseline"],
                                    arrowprops=dict(arrowstyle="->",
                                                    color=PALETTE["Baseline"],
                                                    lw=1.2))
                ax.set_xlabel("Day", fontsize=10)
                ax.set_ylabel("Mean beds in use (7-day smooth)", fontsize=10)
                ax.set_title(f"{source_label} — {unit_label}", fontsize=10)
                ax.yaxis.grid(True, linestyle="--", alpha=0.4)
                ax.set_axisbelow(True)
                ax.legend(title="Threshold", fontsize=8, loc="upper right")
            plt34.tight_layout()
            plt34.savefig(save_path, dpi=150, bbox_inches="tight")
            plt34.close()
            print(f"Saved: {save_path}")

        overlay(LJ_combined_usage_df, "La Jolla",    "LA_JOLLA",    "figures/overlay_la_jolla.png")
        overlay(HC_combined_usage_df, "Hillcrest",   "HILLCREST",   "figures/overlay_hillcrest.png")
        overlay(EC_combined_usage_df, "East Campus", "EAST_CAMPUS", "figures/overlay_east_campus.png")

        # ── Figure 2: Sensitivity curve ──────────────────────────────────────────
        def sensitivity(queue_events_dict, focal_resources, save_path):
            n = len(focal_resources)
            fig, axes = plt34.subplots(1, n, figsize=(6 * n, 5))
            if n == 1: axes = [axes]
            for ax, resource in zip(axes, focal_resources):
                for campus_label, qdf in queue_events_dict.items():
                    sub = qdf[
                        (qdf["resource"] == resource) &
                        (qdf["threshold"].isin(KEEP))
                    ]
                    if sub.empty: continue
                    pts = (sub.groupby("threshold")["wait_time"]
                             .mean().reset_index())
                    pts["x"] = pts["threshold"].map(THRESH_X)
                    pts = pts.sort_values("x")
                    ax.plot(pts["x"], pts["wait_time"],
                            marker="o",
                            color=CAMPUS_COLORS[campus_label],
                            label=campus_label, linewidth=2.2)
                ax.set_xlabel("LOS Threshold (hours)\n[0 = Baseline]", fontsize=9)
                ax.set_ylabel("Mean wait time (hours)", fontsize=9)
                ax.set_title(resource.replace("_", " ").title(), fontsize=10)
                ax.set_xticks([0, 3, 8, 12])
                ax.yaxis.grid(True, linestyle="--", alpha=0.5)
                ax.set_axisbelow(True)
                ax.legend(fontsize=8)
            fig.suptitle(
                "Sensitivity of Mean Wait Time to LOS Transfer Threshold\nby Source Campus",
                fontsize=12, y=1.02
            )
            plt34.tight_layout()
            plt34.savefig(save_path, dpi=150, bbox_inches="tight")
            plt34.close()
            print(f"Saved: {save_path}")

        sensitivity(
            {"La Jolla": LJ_queue_events,
             "Hillcrest": HC_queue_events,
             "East Campus": EC_queue_events},
            ["BOARDED_LA_JOLLA", "ICU_EAST_CAMPUS"],
            "figures/sensitivity_curve.png"
        )

        # ── Figure 3: Stacked ridge plots — 3 panels, one per campus experiment ──
        def ridge_three(lj_qdf, hc_qdf, ec_qdf, save_path, max_wait=200):
            specs = [
                (lj_qdf, "BOARDED_LA_JOLLA", "Rerouted from La Jolla\nBoarding Unit"),
                (hc_qdf, "ICU_HILLCREST",     "Rerouted from Hillcrest\nICU"),
                (ec_qdf, "ICU_EAST_CAMPUS",   "Rerouted from East Campus\nICU"),
            ]
            x_grid = numpy34.linspace(0, max_wait, 600)
            RIDGE_COLORS = {
                "Baseline": "#d62728",
                "3hr":      "#08519c",
                "8hr":      "#4292c6",
                "12hr":     "#9ecae1",
            }
            RIDGE_ORDER = ["12hr", "8hr", "3hr", "Baseline"]  # bottom to top

            fig, axes = plt34.subplots(1, 3, figsize=(18, 6), facecolor="#f5f5f5")
            fig.suptitle(
                "Wait Time Distributions at Most Constrained Resource\n"
                "by Source Campus and LOS Transfer Threshold",
                fontsize=13, y=1.01
            )

            for ax, (qdf, resource, title) in zip(axes, specs):
                ax.set_facecolor("#f5f5f5")
                sub = qdf[
                    (qdf["resource"] == resource) &
                    (qdf["threshold"].isin(KEEP)) &
                    (qdf["wait_time"] > 0) &
                    (qdf["wait_time"] <= max_wait)
                ]

                densities = {}
                for thresh in RIDGE_ORDER:
                    vals = sub[sub["threshold"] == thresh]["wait_time"].values
                    if len(vals) < 10:
                        continue
                    kde = gaussian_kde(vals, bw_method=0.3)
                    densities[thresh] = kde(x_grid)

                if not densities:
                    ax.set_title(f"{title}\n(no data)")
                    continue

                max_d   = max(d.max() for d in densities.values())
                spacing = max_d * 1.8

                for i, thresh in enumerate(RIDGE_ORDER):
                    if thresh not in densities:
                        continue
                    density = densities[thresh]
                    offset  = i * spacing
                    color   = RIDGE_COLORS[thresh]

                    ax.fill_between(x_grid, offset, density + offset,
                                    color=color, alpha=0.8, zorder=i)
                    ax.plot(x_grid, density + offset,
                            color="white", linewidth=1.2, zorder=i + 0.5)
                    ax.text(-6, offset + density.max() * 0.35,
                            thresh, ha="right", va="center",
                            fontsize=10, fontweight="bold", color=color)

                ax.set_xlabel("Wait time (hours)", fontsize=10)
                ax.set_title(title, fontsize=11, pad=10)
                ax.set_yticks([])
                ax.set_xlim(-20, max_wait)
                for spine in ["left", "top", "right"]:
                    ax.spines[spine].set_visible(False)

            plt34.tight_layout()
            plt34.savefig(save_path, dpi=150, bbox_inches="tight")
            plt34.close()
            print(f"Saved: {save_path}")

        ridge_three(LJ_queue_events, HC_queue_events, EC_queue_events,
                    "figures/ridge_plots.png")
        print("\nAll figures saved to ./figures/")


    build_section_34_figures()
    return


if __name__ == "__main__":
    app.run()
