# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==3.0.1",
#     "seaborn==0.13.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os

    # Print current working directory
    print(os.getcwd())
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    return mo, np, pd, plt, sns


@app.class_definition
class g:
    """
    Global model configuration
    """


    # -------------------------
    # Capacities (by state)
    # -------------------------
    # calculated by taking the max utilized when cap was set to 9999

    UNITS = ["ED", "FLOOR", "ICU", "BOARDED"]
    HOSPITALS = ["EAST_CAMPUS", "HILLCREST", "LA_JOLLA"]

    STATES = [
                "ED_EAST_CAMPUS"
                "ED_HILLCREST"
                "ED_LA_JOLLA"
                "FLOOR_EAST_CAMPUS"
                "FLOOR_HILLCREST"
                "FLOOR_LA_JOLLA"
                "ICU_EAST_CAMPUS"
                "ICU_HILLCREST"
                "ICU_LA_JOLLA"
                "BOARDED_EAST_CAMPUS"
                "BOARDED_HILLCREST"
                "BOARDED_LA_JOLLA"
    ]

    capacities ={
        "ED_EAST_CAMPUS":12,
        "ED_HILLCREST":31,
        "ED_LA_JOLLA":67,

        "FLOOR_EAST_CAMPUS":82, 
        "FLOOR_HILLCREST":210, 
        "FLOOR_LA_JOLLA":340, 

        "ICU_EAST_CAMPUS": 8, 
        "ICU_HILLCREST":30, 
        "ICU_LA_JOLLA":57, 

        "BOARDED_EAST_CAMPUS":4, 
        "BOARDED_HILLCREST":10, 
        "BOARDED_LA_JOLLA":49, 
        }

    arrival_rate = 0.15#0.33  # mean inter-arrival time in hours


    arrival_rates = {
        "EAST_CAMPUS": {
            "weekday": {"mean": 92.4, "std": 10.5},
            "weekend": {"mean": 83.3, "std": 10.5},
        },
        "HILLCREST": {
            "weekday": {"mean": 144.7, "std": 19.9},
            "weekend": {"mean": 125.1, "std": 10.3},
        },
        "LA_JOLLA": {
            "weekday": {"mean": 198.2, "std": 26.2},
            "weekend": {"mean": 148.9, "std": 14.5},
        },
    }

    # state_transition_probability = loc_trans_prob_df

    fitted_dist_per_state = {
  "EAST_CAMPUS": {
    "BOARDED_EAST_CAMPUS": {
      "dist": "weibull_min",
      "params": [
        0.8747476524911071,
        0.016666666666666663,
        4.492451830696613
      ],
      "sse": 0.0005844883522429119
    },
    "ED_EAST_CAMPUS": {
      "dist": "expon",
      "params": [
        0.016666666666666666,
        2.6315651977723156
      ],
      "sse": 0.00025691356483194754
    },
    "FLOOR_EAST_CAMPUS": {
      "dist": "lognorm",
      "params": [
        1.0007485942371375,
        -1.7840455052346713,
        59.88185430817963
      ],
      "sse": 2.3532698010305996e-07
    },
    "ICU_EAST_CAMPUS": {
      "dist": "lognorm",
      "params": [
        0.9463194275708006,
        -2.649104066705131,
        48.31343840460024
      ],
      "sse": 2.2796787319679487e-05
    }
  },
  "HILLCREST": {
    "BOARDED_HILLCREST": {
      "dist": "lognorm",
      "params": [
        0.9706204200542583,
        -0.2848100588281825,
        5.662325166284325
      ],
      "sse": 0.0007761886274700568
    },
    "ED_HILLCREST": {
      "dist": "lognorm",
      "params": [
        0.6088348293399375,
        -1.1303920373019136,
        4.936473598018322
      ],
      "sse": 0.0005381139256369019
    },
    "FLOOR_HILLCREST": {
      "dist": "lognorm",
      "params": [
        1.1855928624837022,
        -3.2598445837837953,
        59.24241254863264
      ],
      "sse": 1.6273464527147265e-07
    },
    "ICU_HILLCREST": {
      "dist": "lognorm",
      "params": [
        1.0023067732204871,
        -2.175497258041816,
        59.64845898841404
      ],
      "sse": 9.646418510824467e-07
    }
  },
  "LA_JOLLA": {
    "BOARDED_LA_JOLLA": {
      "dist": "expon",
      "params": [
        0.016666666666666666,
        31.774376046263885
      ],
      "sse": 0.00028116168095593534
    },
    "ED_LA_JOLLA": {
      "dist": "lognorm",
      "params": [
        0.5361751979309719,
        -1.2519059748277743,
        4.660214954932898
      ],
      "sse": 0.0009388285553675288
    },
    "FLOOR_LA_JOLLA": {
      "dist": "lognorm",
      "params": [
        1.176461352034086,
        -3.3450226925826176,
        52.9991865125243
      ],
      "sse": 1.2740239783820567e-06
    },
    "ICU_LA_JOLLA": {
      "dist": "lognorm",
      "params": [
        1.1196423422590809,
        -0.9556218608005012,
        46.63698482208767
      ],
      "sse": 1.0572961444719779e-07
    }
  }
}

    # -------------------------
    # Deterioration / clinical rules
    # -------------------------
    deterioration_rate = 0.05
    deterioration_transfer_prob = 0.95
    transfer_delay_hours = (2, 4)

    # -------------------------
    # Simulation controls
    # -------------------------
    sim_duration = 4320 #6 months #8760 #one year in hours
    number_of_runs = 3
    random_number_set = 42
    audit_interval = 1

    # for predictor:
    q = 0.75
    threshold = [3, 8, 12]


@app.cell
def _(pd):
    LJ_df = pd.read_csv('../results/simulation_logs/LJ_combined_df_all_thresholds+baseline.csv')
    HC_df = pd.read_csv('../results/simulation_logs/HC_combined_df_all_thresholds+baseline.csv')
    EC_df = pd.read_csv('../results/simulation_logs/EC_combined_df_all_thresholds+baseline.csv')
    return EC_df, HC_df, LJ_df


@app.cell
def _(np, pd):
    def compute_usage(df, resource):
        events = df[(df["event"]== f"{resource}_begins") | 
                    (df["event"] == f"{resource}_ends")].copy()

        events = events.sort_values( "time").reset_index(drop = True)

        in_use = 0
        records = []
        for i, row in events.iloc[:-1].iterrows():
            next_row = events.iloc[i + 1]

            if row["event"].endswith("begins"):
                in_use += 1
            else:
                in_use -= 1

            start_time = row['time']
            end_time = next_row['time']
            duration = end_time - start_time

            records.append({
                'start_time': row["time"],
                'duration': duration, 
                'in_use': in_use}
             )

        util_df = pd.DataFrame(records)
        if util_df.empty:
            return pd.DataFrame(columns = ['start_time', 'duration', 'in_use', 'hour', 'day', 'util_smooth'])

        util_df['hour'] = np.floor(util_df['start_time']).astype(int)
        util_df['day'] =  util_df["hour"] // 24


        daily_util = util_df.groupby('day')['in_use'].mean().reset_index()

        daily_util['util_smooth'] = daily_util['in_use'].rolling(window=7, min_periods=1).mean()
        return daily_util
    return (compute_usage,)


@app.cell
def _(compute_usage, pd):
    def create_combined_usage_df(df):
        resources = [
            "ED_LA_JOLLA", "ICU_LA_JOLLA", "FLOOR_LA_JOLLA", "BOARDED_LA_JOLLA",
            "ED_HILLCREST", "ICU_HILLCREST", "FLOOR_HILLCREST", "BOARDED_HILLCREST",
            "ED_EAST_CAMPUS", "ICU_EAST_CAMPUS", "FLOOR_EAST_CAMPUS", "BOARDED_EAST_CAMPUS"
        ]
        all_usage_list = []

        for run_id, run_df in df.groupby("run_number"):
            for resource in resources:
                usage_df = compute_usage(run_df, resource)

                if usage_df.empty:
                    continue

                usage_df["percent_utilized"] = (
                    usage_df["in_use"] / g.capacities[resource]
                )

                usage_df["run_number"] = run_id
                usage_df["resource"] = resource

                all_usage_list.append(usage_df)

        combined_usage = pd.concat(all_usage_list, ignore_index=True)
        return combined_usage
    return (create_combined_usage_df,)


@app.cell
def _(LJ_df, create_combined_usage_df):
    LJ_combined_usage_df = create_combined_usage_df(LJ_df)
    return (LJ_combined_usage_df,)


@app.cell
def _(EC_df, HC_df, create_combined_usage_df):
    HC_combined_usage_df = create_combined_usage_df(HC_df)
    EC_combined_usage_df = create_combined_usage_df(EC_df)
    return EC_combined_usage_df, HC_combined_usage_df


@app.cell
def _(EC_combined_usage_df, HC_combined_usage_df, LJ_combined_usage_df):
    run_to_threshold = {
        0: "Baseline",
        1: "3hr",
        2: "8hr",
        3: "12hr",
        4: "24hr",
        5: "48hr",
        6: "72hr"
    }

    LJ_combined_usage_df['threshold'] = LJ_combined_usage_df['run_number'].map(run_to_threshold)
    EC_combined_usage_df['threshold'] = EC_combined_usage_df['run_number'].map(run_to_threshold)
    HC_combined_usage_df['threshold'] = HC_combined_usage_df['run_number'].map(run_to_threshold)
    return


@app.cell
def _(HC_combined_usage_df):
    HC_combined_usage_df
    return


@app.cell
def _(plt, sns):
    def plot_utilization(curr_resource, df, hospital_name, filename = None):
        run_labels = {
                0: "Baseline",
                1: "3hr",
                2: "8hr",
                3: "12hr",
                4: "24hr",
                5: "48hr",
                6: "72hr"
            }

        hue_order=run_labels.values()
        # curr_resource = 'FLOOR_LA_JOLLA'
        plt.figure(figsize=(25, 8))
        plt.axhspan(0.8, 1.0, color="gray", alpha=0.2)


        unique_runs = sorted(df["threshold"].unique())

        # 🎨 Monochrome palette (e.g., Blues)
        mono_palette = sns.color_palette("Blues", n_colors=len(unique_runs))

        # Map run → color
        palette = dict(zip(hue_order, mono_palette))
        palette['Baseline'] = 'red'  # override baseline

        ax = sns.lineplot(
            data=df[df["resource"] == curr_resource],
            x="day",
            y="percent_utilized",
            hue="threshold",
            palette=palette,
            hue_order=hue_order
        )

        # Labels
        plt.title(f'Rerouted from {hospital_name}: Percent Utilization of {curr_resource.replace('_', ' ')} per Threshold')
        plt.xlabel("Time (days)")
        plt.ylabel("Percent Utilized")
        plt.ylim(0, 1.1)
        ax.legend(title='Threshold', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)

        if filename:  # if a filename is provided, save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")


        plt.show()
    return (plot_utilization,)


@app.cell
def _(plt, sns):
    def plot_percent_above_threshold(combined_usage_df, hospital_name, filename = None):
        resource_order = [
        "ED_LA_JOLLA",
        "BOARDED_LA_JOLLA",
        "ICU_LA_JOLLA",
        "FLOOR_LA_JOLLA",
        "ED_HILLCREST",
        "BOARDED_HILLCREST",
        "ICU_HILLCREST",
        "FLOOR_HILLCREST",
        "ED_EAST_CAMPUS",
        "BOARDED_EAST_CAMPUS",
        "ICU_EAST_CAMPUS",
        "FLOOR_EAST_CAMPUS",
    ]
        unit_labels = [r.split("_")[0] for r in resource_order]
        run_labels = {
                0: "Baseline",
                1: "3hr",
                2: "8hr",
                3: "12hr",
                4: "24hr",
                5: "48hr",
                6: "72hr"
            }

        hue_order=run_labels.values()

        df = combined_usage_df.copy()
        df['in_range'] = (df['percent_utilized'] >= 0.8) & (df['percent_utilized'] <= 1.0)


        avg_time_in_range = (
                        df.groupby(['threshold', 'run_number', 'resource'])['in_range']
                          .mean()
                          .reset_index()
                    )

        # unique_runs = sorted(avg_time_in_range["run_number"].unique())

        # 🎨 Monochrome palette (e.g., Blues)
        mono_palette = sns.color_palette("Blues", n_colors=len(hue_order))

        # Map run → color
        palette = dict(zip(hue_order, mono_palette))
        palette['Baseline'] = 'red'  # override baseline

        plt.figure(figsize=(14,6))

        ax = sns.barplot(
            data=avg_time_in_range,
            x='resource',
            y='in_range',
            hue='threshold',
            palette=palette, 
            order = resource_order,
            hue_order = hue_order
        )

        ax.set_xticklabels(unit_labels, rotation=0)


        hospitals = ["LA JOLLA", "HILLCREST", 'EAST CAMPUS']
        group_size = 4  # number of units per hospital

        for i, hospital in enumerate(hospitals):
            center = i * group_size + (group_size - 1) / 2
            ax.text(
                center,
                -0.07,                 # move downward
                hospital,
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight='bold'
            )
        for i in range(1, len(hospitals)):
            ax.axvline(i * group_size - 0.5, color='black', linewidth=1)

        # plt.xticks(rotation=45)
        plt.ylabel("Proportion of days")
        ax.set_xlabel("Resource", labelpad=25)    
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
        ax.legend(title='Threshold', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)
        plt.title(f"{hospital_name}: Proportion of time above 80% utilization ")
        plt.tight_layout()


        if filename:  # if a filename is provided, save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")


        plt.show()


        return avg_time_in_range
    return (plot_percent_above_threshold,)


@app.cell
def _(
    EC_combined_usage_df,
    HC_combined_usage_df,
    LJ_combined_usage_df,
    plot_percent_above_threshold,
    plot_utilization,
):
    plot_utilization('FLOOR_LA_JOLLA', LJ_combined_usage_df, 'La Jolla',)# "LJ_FLOOR_LA_JOLLA_utilization.png")
    plot_utilization('BOARDED_LA_JOLLA', LJ_combined_usage_df, 'La Jolla',)# "LJ_BOARDED_LA_JOLLA_utilization.png")
    plot_utilization('ED_LA_JOLLA', LJ_combined_usage_df, 'La Jolla',)# 'LJ_ED_LA_JOLLA_utilization.png')
    plot_utilization('ICU_LA_JOLLA', LJ_combined_usage_df, 'La Jolla',)# 'LJ_ICU_LA_JOLLA_utilization.png')

    LJ_percent_above_80 = plot_percent_above_threshold(LJ_combined_usage_df, 'La Jolla',)# "LJ_percent_above_80_utilization.png")
    EC_percent_above_80 =plot_percent_above_threshold(EC_combined_usage_df, 'East Camous',)# "EC_percent_above_80_utilization.png")
    HC_percent_above_80 =plot_percent_above_threshold(HC_combined_usage_df, 'Hillcrest',)# "HC_percent_above_80_utilization.png")
    return EC_percent_above_80, HC_percent_above_80, LJ_percent_above_80


@app.cell
def _(mo):
    mo.md(r"""
    Data tables for proportion of days each resource was above 80% utilization (for every hospital rerouting stemmed from)
    """)
    return


@app.cell
def _(LJ_percent_above_80):
    LJ_percent_above_80.groupby(['run_number','threshold', 'resource'])['in_range'].mean()
    return


@app.cell
def _(HC_percent_above_80):
    HC_percent_above_80.groupby(['run_number','threshold', 'resource'])['in_range'].mean()
    return


@app.cell
def _(EC_percent_above_80):
    EC_percent_above_80.groupby(['run_number','threshold', 'resource'])['in_range'].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # WAIT TIME
    """)
    return


@app.function
def calculate_wait_times(df):

    run_to_threshold = {
    0: "Baseline",
    1: "3hr",
    2: "8hr",
    3: "12hr",
    4: "24hr",
    5: "48hr",
    6: "72hr"
}


# calculate wait times
    queue_events = df.sort_values(
        ["run_number", "entity_id", "time"]
    ).copy()

    queue_events["next_event"] = (
        queue_events
            .groupby(["run_number", "entity_id"])["event"]
            .shift(-1)
    )

    queue_events['resource'] = queue_events['event'].str.split('_').str[:-2].str.join('_')

    queue_events["next_time"] = (
        queue_events
            .groupby(["run_number", "entity_id"])["time"]
            .shift(-1)
    )

    queue_events["next_resource"] = (
        queue_events
            .groupby(["run_number", "entity_id"])["resource"]
            .shift(-1)
    )
    queue_events = queue_events[
        queue_events["event_type"] == "queue"
    ]
    queue_events["wait_time"] = (
    queue_events["next_time"] - queue_events["time"]
)

    queue_events['threshold'] = queue_events['run_number'].map(run_to_threshold)

    return queue_events


@app.cell
def _(plt, sns):
    def plot_avg_wait_per_resource(queue_event_df, hospital_name, filename = None):
        resource_order = [
        "ED_LA_JOLLA",
        "BOARDED_LA_JOLLA",
        "ICU_LA_JOLLA",
        "FLOOR_LA_JOLLA",
        "ED_HILLCREST",
        "BOARDED_HILLCREST",
        "ICU_HILLCREST",
        "FLOOR_HILLCREST",
        "ED_EAST_CAMPUS",
        "BOARDED_EAST_CAMPUS",
        "ICU_EAST_CAMPUS",
        "FLOOR_EAST_CAMPUS",
    ]
        unit_labels = [r.split("_")[0] for r in resource_order]

        run_labels = {
                0: "Baseline",
                1: "3hr",
                2: "8hr",
                3: "12hr",
                4: "24hr",
                5: "48hr",
                6: "72hr"
            }

        hue_order=run_labels.values()

        avg_wait = (
            queue_event_df
                .groupby(["run_number", "threshold", "resource"])["wait_time"]
                .mean()
                .reset_index()
        )
        avg_wait['wait_time_minutes'] = avg_wait['wait_time'] * 60


        # unique_runs = sorted(avg_wait["threshold"].unique())

            # 🎨 Monochrome palette (e.g., Blues)
        mono_palette = sns.color_palette("Blues", n_colors=len(hue_order))

        # Map run → color
        palette = dict(zip(hue_order, mono_palette))
        palette['Baseline'] = 'red'  # override baseline

        plt.figure(figsize=(12,6))

        ax = sns.barplot(
            data=avg_wait,
            x='resource',
            y='wait_time',
            hue='threshold',
            palette=palette, 
            order = resource_order,
            hue_order = hue_order,
        )

        ax.set_xticklabels(unit_labels, rotation=0)


        hospitals = ["LA JOLLA", "HILLCREST", 'EAST CAMPUS']
        group_size = 4  # number of units per hospital

        for i, hospital in enumerate(hospitals):
            center = i * group_size + (group_size - 1) / 2
            ax.text(
                center,
                -0.07,                 # move downward
                hospital,
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight='bold'
            )
        for i in range(1, len(hospitals)):
            ax.axvline(i * group_size - 0.5, color='black', linewidth=1)

        # plt.xticks(rotation=45)
        plt.ylabel("Time (minutes)")
        ax.set_xlabel("Resource", labelpad=20)    
        ax.legend(title='Threshold', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)
        plt.title(f"Rerouted from {hospital_name}: Average Wait Time per Resource by Threshold")
        plt.tight_layout()


        if filename:  # if a filename is provided, save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")


        plt.show()

        return avg_wait
    return (plot_avg_wait_per_resource,)


@app.cell
def _(plt, sns):
    def plot_probability_of_waiting(queue_events, hospital_name, filename = None):
        resource_order = [
        "ED_LA_JOLLA",
        "BOARDED_LA_JOLLA",
        "ICU_LA_JOLLA",
        "FLOOR_LA_JOLLA",
        "ED_HILLCREST",
        "BOARDED_HILLCREST",
        "ICU_HILLCREST",
        "FLOOR_HILLCREST",
        "ED_EAST_CAMPUS",
        "BOARDED_EAST_CAMPUS",
        "ICU_EAST_CAMPUS",
        "FLOOR_EAST_CAMPUS",
    ]
        unit_labels = [r.split("_")[0] for r in resource_order]

        run_labels = {
                0: "Baseline",
                1: "3hr",
                2: "8hr",
                3: "12hr",
                4: "24hr",
                5: "48hr",
                6: "72hr"
            }

        hue_order=run_labels.values()

        wait_by_resource = (
            queue_events
            .assign(waited=lambda x: x["wait_time"] > 0)
            .groupby(["run_number", 'threshold', "resource"])["waited"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )


        mono_palette = sns.color_palette("Blues", n_colors=len(hue_order))

        # Map run → color
        palette = dict(zip(hue_order, mono_palette))
        palette['Baseline'] = 'red'  # override baseline


        plt.figure(figsize=(12,6))

        ax = sns.barplot(
            data=wait_by_resource,
            x='resource',
            y='waited',
            hue='threshold',
            palette=palette, 
            order=resource_order,
            hue_order = hue_order

        )

        ax.set_xticklabels(unit_labels, rotation=0)


        hospitals = ["LA JOLLA", "HILLCREST", 'EAST CAMPUS']
        group_size = 4  # number of units per hospital

        for i, hospital in enumerate(hospitals):
            center = i * group_size + (group_size - 1) / 2
            ax.text(
                center,
                -0.07,                 # move downward
                hospital,
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=12,
                fontweight='bold'
            )
        for i in range(1, len(hospitals)):
            ax.axvline(i * group_size - 0.5, color='black', linewidth=1)


        # plt.xticks(rotation=45)
        ax.set_xlabel("Resource", labelpad=25)    
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.5)
        plt.ylabel("Probability of Waiting")
        plt.title(f"Rerouted from {hospital_name}: Probability of Experiencing a Wait by Resource by Threshold")    
        ax.legend(title='Threshold', loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12)
        plt.tight_layout()


        if filename:  # if a filename is provided, save the figure
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")


        plt.show()

        return wait_by_resource
    return (plot_probability_of_waiting,)


@app.cell
def _(EC_df, HC_df, LJ_df):
    LJ_queue_events = calculate_wait_times(LJ_df)
    HC_queue_events = calculate_wait_times(HC_df)
    EC_queue_events = calculate_wait_times(EC_df)
    return EC_queue_events, HC_queue_events, LJ_queue_events


@app.cell
def _(mo):
    mo.md(r"""
    ## Average wait time per resource and threshold, by hospital rerouting stemmed from
    """)
    return


@app.cell
def _(
    EC_queue_events,
    HC_queue_events,
    LJ_queue_events,
    plot_avg_wait_per_resource,
):
    LJ_avg_wait = plot_avg_wait_per_resource(LJ_queue_events, "La Jolla",)# 'LJ_average_wait_per_resource.png')
    HC_avg_wait = plot_avg_wait_per_resource(HC_queue_events, 'Hillcrest',)#'HC_average_wait_per_resource.png')
    EC_avg_wait = plot_avg_wait_per_resource(EC_queue_events, 'East Campus',)# 'EC_average_wait_per_resource.png')
    return (LJ_avg_wait,)


@app.cell
def _(mo):
    mo.md(r"""
    La jolla Data
    """)
    return


@app.cell
def _(LJ_avg_wait):
    LJ_avg_wait.groupby(['run_number', 'threshold', 'resource'])['wait_time'].mean()
    return


@app.cell
def _():
    # EC_avg_wait.groupby(['run_number', 'threshold', 'resource'])['wait_time'].mean()
    return


@app.cell
def _():
    # HC_avg_wait.groupby(['run_number', 'threshold', 'resource'])['wait_time'].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Probability a patient will experience a wait, by resource, by threshold
    """)
    return


@app.cell
def _(
    EC_queue_events,
    HC_queue_events,
    LJ_queue_events,
    plot_probability_of_waiting,
):
    LJ_probability_wait = plot_probability_of_waiting(LJ_queue_events, "La Jolla", )#'LJ_probability_of_wait.png')
    HC_probability_wait = plot_probability_of_waiting(HC_queue_events, 'Hillcrest',)# 'HC_probability_of_wait.png')
    EC_probability_wait = plot_probability_of_waiting(EC_queue_events, 'East Campus',)#'EC_probability_of_wait.png')
    return (LJ_probability_wait,)


@app.cell
def _(mo):
    mo.md(r"""
    This data is from the run that applied rerouting logic to La Jolla Admissions Only (replace hosp initials for other hospital data)
    """)
    return


@app.cell
def _(LJ_probability_wait):
    LJ_probability_wait.groupby(['run_number','threshold', 'resource',])['waited'].mean()
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
                "BOARDED_HILLCREST": 10, "ICU_HILLCREST": 30,
                "BOARDED_EAST_CAMPUS": 4,  "ICU_EAST_CAMPUS": 8,
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

                # Special case: if only baseline rendered (e.g. LJ boarding),
                # add explanatory annotation
                threshold_curves_missing = [t for t in ["3hr","8hr","12hr"]
                                            if t not in densities]

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

                # If threshold runs produced near-zero waits, annotate instead
                if threshold_curves_missing:
                    ax.text(
                        0.55, 0.5,
                        "Under transfer policy:\nnear-zero wait events recorded\n"
                        "(patients rerouted before boarding)",
                        transform=ax.transAxes,
                        fontsize=9, color="#2171b5",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="white",
                                  edgecolor="#2171b5",
                                  alpha=0.85)
                    )

            plt34.tight_layout()
            plt34.savefig(save_path, dpi=150, bbox_inches="tight")
            plt34.close()
            print(f"Saved: {save_path}")

        ridge_three(LJ_queue_events, HC_queue_events, EC_queue_events,
                    "figures/ridge_plots.png")
        print("\nAll figures saved to ./figures/")


    build_section_34_figures()
    return


@app.cell
def _(EC_df, HC_df, LJ_df):
    def compute_transfer_volume():
        """
        Counts how many patients were transferred per day under each threshold.
        Paste this as a new cell AFTER Kayanne's data loading cells.
        Requires: LJ_df, HC_df, EC_df already loaded.
        """
        import pandas as pd34

        SIM_DURATION_DAYS = 4328 / 24   # ~180 days
        WARMUP_DAYS = 720 / 24          # 30 days warmup excluded
        ACTIVE_DAYS = SIM_DURATION_DAYS - WARMUP_DAYS  # ~150 days

        run_to_threshold = {
            0: "Baseline",
            1: "3hr",
            2: "8hr",
            3: "12hr",
        }

        results = []

        for campus_label, df in [("La Jolla", LJ_df),
                                  ("Hillcrest", HC_df),
                                  ("East Campus", EC_df)]:

            # Transfers are logged as arrival_departure events where the
            # event string contains "transfer" — adjust if your event
            # string uses a different keyword
            transfer_events = df[
                df["event"].str.contains("transfer", case=False, na=False) &
                (df["time"] >= 720)   # exclude warmup
            ]

            # If no "transfer" keyword found, fall back to counting
            # unique patients who appear at a DIFFERENT campus than source
            if transfer_events.empty:
                # Alternative: count patients whose first event campus
                # differs from source campus — edit resource prefix as needed
                source_prefix = {
                    "La Jolla": "LA_JOLLA",
                    "Hillcrest": "HILLCREST",
                    "East Campus": "EAST_CAMPUS"
                }[campus_label]

                # Count entity_ids that have events at non-source campuses
                non_source = df[
                    ~df["event"].str.contains(source_prefix, na=False) &
                    (df["event_type"] == "resource_use") &
                    (df["time"] >= 720)
                ]
                transfer_events = non_source.drop_duplicates(
                    subset=["run_number", "entity_id"]
                )

            grouped = (transfer_events
                       .groupby("run_number")["entity_id"]
                       .nunique()
                       .reset_index()
                       .rename(columns={"entity_id": "total_transfers"}))

            grouped["threshold"] = grouped["run_number"].map(run_to_threshold)
            grouped["transfers_per_day"] = (grouped["total_transfers"]
                                            / ACTIVE_DAYS).round(1)
            grouped["source_campus"] = campus_label
            results.append(grouped[grouped["run_number"] > 0])  # exclude baseline

        summary = pd34.concat(results, ignore_index=True)
        summary = summary[["source_campus", "threshold",
                            "total_transfers", "transfers_per_day"]]
        summary = summary.sort_values(["source_campus", "threshold"])

        print("\nTransfer Volume by Source Campus and Threshold")
        print("=" * 55)
        print(summary.to_string(index=False))

        # Pivot for a cleaner table
        pivot = summary.pivot_table(
            index="threshold",
            columns="source_campus",
            values="transfers_per_day"
        )
        pivot.index.name = "Threshold"
        pivot.columns.name = "Transfers/day from →"
        print("\nTransfers per day (pivoted):")
        print(pivot.to_string())

        return summary


    transfer_summary = compute_transfer_volume()
    return


if __name__ == "__main__":
    app.run()
