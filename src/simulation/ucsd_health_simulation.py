# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==2.3.3",
#     "scipy==1.17.0",
#     "sim-tools==1.0.3",
#     "simpy==4.1.1",
#     "vidigi==1.2.2",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # import ibis
    # import ibis.selectors as s
    # from ibis import _
    import simpy
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sim_tools.distributions import Exponential, Lognormal
    from vidigi.resources import VidigiStore
    from vidigi.logging import EventLogger
    from vidigi.utils import EventPosition, create_event_position_df
    from vidigi.animation import animate_activity_log
    import re
    from scipy import stats
    import json

    # Set options
    # ibis.options.interactive = True
    pd.set_option("display.max_columns", None)
    return (
        EventLogger,
        Exponential,
        VidigiStore,
        json,
        mo,
        np,
        pd,
        plt,
        random,
        simpy,
    )


@app.cell
def _(json, pd):
    # import all the data needed for sim
    adt_df = pd.read_csv('ADT_cleaned_with_tiers.csv')
    fitted_distributions_df = pd.read_csv('states_fitted_distribution_df.csv')
    arrival_rates_df = pd.read_csv('weekday_weekend_arrival_rates_df.csv')
    loc_trans_prob_df = pd.read_csv('probability_matrix.csv')

    with open('trajectory_library.json', 'r') as json_file:
        trajectory_library = json.load(json_file)
    return arrival_rates_df, trajectory_library


@app.cell
def _(arrival_rates_df):
    arrival_rates_df
    return


@app.cell
def _():
    # # Preview a p
    # for eid, path in list(trajectory_library.items())[:5]:
    #     traj_str = ' → '.join([f"{state} ({dur:.1f}h)" for state, dur in path])
    #     print(f"{eid}: {traj_str}")
    return


@app.cell
def _(trajectory_library):
    trajectory_library_by_campus = {
        "LA_JOLLA": {},
        "HILLCREST": {},
        "EAST_CAMPUS": {}
    }

    for enc_id, traj in trajectory_library.items():
        first_state = traj[0][0]  # first state's name
        campus = "_".join(first_state.split("_")[1:])

        if campus is not None:
            trajectory_library_by_campus[campus][enc_id] = traj
    return (trajectory_library_by_campus,)


@app.cell
def _(trajectory_library_by_campus):
    trajectory_library_by_campus.keys()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Simulation build
    """)
    return


@app.cell
def _():
    HOSPITALS = ["EAST_CAMPUS", "HILLCREST", "LA_JOLLA"]
    UNITS = ["ED", "FLOOR", "ICU"]

    STATES = [f"{u}_{h}" for u in UNITS for h in HOSPITALS]
    STATES
    return


@app.cell
def _():
    9999
    return


@app.class_definition
class g:
    """
    Global model configuration
    """

    UNITS = ["ED", "FLOOR", "ICU"]
    HOSPITALS = ["EAST_CAMPUS", "HILLCREST", "LA_JOLLA"]

    STATES = [f"{u}_{h}" for u in UNITS for h in HOSPITALS]

    # -------------------------
    # Capacities (by state)
    # -------------------------
    # calculated by taking the max utilized when cap was set to 999
    capacities = {
        "ED_LA_JOLLA": 9999,
        "ICU_LA_JOLLA": 9999,
        "FLOOR_LA_JOLLA": 9999,

        "ED_HILLCREST": 9999,
        "ICU_HILLCREST":9999,
        "FLOOR_HILLCREST": 9999,

        "ED_EAST_CAMPUS": 9999,
        "ICU_EAST_CAMPUS": 9999,
        "FLOOR_EAST_CAMPUS": 9999,
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

    # -------------------------
    # Length-of-stay distributions (by state)
    # You can plug in scipy, empirical samplers, etc.
    # -------------------------
    los_distributions = {
        # example placeholders
        # "EC_ED": Exponential(mean=4),
        # "HC_ICU": LogNormal(mu=..., sigma=...),
    }

    # -------------------------
    # Transition matrices
    # Can be swapped by scenario
    # -------------------------
    transition_matrices = {
        # "baseline": {...},
        # "policy_A": {...},
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
    sim_duration = 730 #one month in hours
    number_of_runs = 1
    random_number_set = 42
    audit_interval = 1


@app.cell
def _():
    from collections import defaultdict

    class Patient:
        def __init__(self, p_id, encounter_id, path, campus = None):
            # Identifiers
            self.identifier = p_id
            self.encounter_id = encounter_id

            # Current state in the simulation (e.g. "EC_ED")
            self.current_state = None

            # History
            self.path = path
            self.state_history = []

            # Timing
            self.arrival = None
            self.total_time = None

            # Wait and service times keyed by state
            self.wait_times = defaultdict(list) # ex. {ED_LA_JOLLA: [0,49,3], FLOOR_LA_JOLLA:[]}
            self.service_times = defaultdict(list)

            # Clinical flags
            self.ever_icu = None
            self.deteriorated = False

            # Transfers (ordered list of states or hospitals)
            self.transfer_list = []


            self.campus = campus
    return (Patient,)


@app.cell
def _(EventLogger, Exponential, Patient, VidigiStore, np, pd, random, simpy):
    class Model:
        def __init__(self, run_number, trajectory_library):
            # Create a SimPy environment in which everything will live
            self.env = simpy.Environment()

            # Create a patient counter (which we'll use as a patient ID)
            self.patient_counter = 0
            # self.patient_objects = []

            # Create an empty list to store our patient objects - these can be handy
            # to look at later
            self.patients = []

            # Create our resources
            self.init_resources()

            # Resource monitoring
            # self.resource_log = []

            # Store the passed in run number
            self.run_number = run_number

            self.patient_inter_arrival_dist = Exponential(
                mean=g.arrival_rate,
                random_seed=(self.run_number + 1) * g.random_number_set
    )

            # compute prob of entering thru any campus
            # total_rate = sum(g.arrival_rates_weekday.values())
            # self.weekday_campus_probs = {
            #     campus: rate / total_rate for campus, rate in g.arrival_rates_weekday.items()}
            # self.campuses = list(self.campus_probs.keys())
            # self.probs = list(self.campus_probs.values())

            # Build patient library from collapsed activity library
            # {LA_JOLLA: {trajectories}, HILLCREST: {trajectories}, EAST_CAMPUS: {trajectories} }
            self.patient_library = trajectory_library

            # Precompute all possible real patients by campus
            self.unique_encounters = {'LA_JOLLA': list(self.patient_library['LA_JOLLA'].keys()),

                                    'HILLCREST': list(self.patient_library['HILLCREST'].keys()),

                                    'EAST_CAMPUS': list(self.patient_library['EAST_CAMPUS'].keys())
                                    }



            # Logger
            self.logger = EventLogger(
                env=self.env,
                run_number=self.run_number
            )

            # self.acuity_flags = acuity_flags


            self.results_df = pd.DataFrame()

            self.results_df["patient_id"] = [1]

            for state in g.STATES:
                self.results_df[f"q_time_{state.lower()}"] = [0.0]
                self.results_df[f"time_in_{state.lower()}"] = [0.0]

            self.results_df.set_index("patient_id", inplace=True)

            self.lj_mean_q_time_ed = 0
            self.lj_mean_q_time_icu = 0
            self.lj_mean_q_time_floor = 0

            self.hc_mean_q_time_ed = 0
            self.hc_mean_q_time_icu = 0
            self.hc_mean_q_time_floor = 0

            self.ec_mean_q_time_ed = 0
            self.ec_mean_q_time_icu = 0
            self.ec_mean_q_time_floor = 0


            self.utilization_audit = []

            # need to add these for utilization plot 
            self.lj_ed_utilization = []
            self.lj_icu_utilization = []
            self.lj_floor_utilization = []

            self.hc_ed_utilization = []
            self.hc_icu_utilization = []
            self.hc_floor_utilization = []

            self.ec_ed_utilization = []
            self.ec_icu_utilization = []
            self.ec_floor_utilization = []


        def init_resources(self):
            self.resources = {}  # dict to hold state_name -> VidigiStore object

            for state in g.STATES:
                cap = g.capacities.get(state, 0)
                if cap <= 0:
                    continue  # skip states with no capacity defined

                store = VidigiStore(
                    self.env,
                    num_resources=cap,
                    capacity=cap
                )

                # Save dynamically as attribute AND in dict
                # ed_la_jolla_beds, floor_la_jolla_beds, etc
                setattr(self, f"{state.lower()}_beds", store)

                # list of resources
                self.resources[state] = store

        def get_mean_interarrival(self, campus, current_time):
            day = int(current_time // 24) % 7

            if day < 5:
                mean_daily = g.arrival_rates[campus]["weekday"]["mean"]
            else:
                mean_daily = g.arrival_rates[campus]["weekend"]["mean"]

            return 24 / mean_daily


        def generator_patient_arrivals(self, campus):
            # dist = self.arrival_rates[campus]

            while True:
                self.patient_counter += 1

                encounter_id = random.choice(self.unique_encounters[campus])
                path = self.patient_library[campus][encounter_id]


                p = Patient(
                    p_id=self.patient_counter,
                    encounter_id=encounter_id,
                    path=path,
                    campus = campus,
                )

                self.patients.append(p)

                self.env.process(self.patient_journey(p))

                mean_inter = self.get_mean_interarrival(campus, self.env.now)
                yield self.env.timeout(
                    np.random.exponential(mean_inter)
                )

        def patient_journey(self, patient):
            patient.arrival = self.env.now

            self.logger.log_arrival(entity_id=patient.identifier)


            for state, duration in patient.path:
                # Map state to resource if you have one
                resource_pool = self.resources.get(state, None)  # returns None if no resource
                # Start wait time measurement
                start_wait = self.env.now

                self.logger.log_queue(entity_id=patient.identifier, 
                              event_type='queue',
                              event=f"{state}_wait_begins"
                             )

                if resource_pool is not None:
                    # Request the resource (will automatically wait if busy)
                    with resource_pool.request() as req:
                        bed = yield req

                        # Wait time = time spent waiting for resource
                        wait_time = self.env.now - start_wait
                        patient.wait_times[state].append(wait_time)

                        self.logger.log_resource_use_start(
                            entity_id=patient.identifier,
                            event_type='resource_use',
                            event=f"{state}_begins",
                            resource_id=bed.id_attribute
                        )

                        # Log service
                        yield self.env.timeout(duration)
                        patient.service_times[state].append(duration)

                        self.logger.log_resource_use_end(
                            entity_id=patient.identifier,
                            event_type='resource_use_end',
                            event=f"{state}_ends",
                            resource_id=bed.id_attribute
                        )

                else:
                    # No resource: just wait for the duration

                    self.logger.log_event(entity_id=patient.identifier,
                                          event_type='resource_use',
                                          event=f"{state}_begins")

                    yield self.env.timeout(duration)

                    self.logger.log_event(entity_id=patient.identifier, 
                                          event_type='resource_use_end',
                                          event=f"{state}_ends")

                    patient.wait_times[state].append(0.0)
                    patient.service_times[state].append(duration)

                patient.total_time = self.env.now - patient.arrival

                # Done with the patient
                patient.total_time = self.env.now - patient.arrival
                self.logger.log_event(
                    entity_id=patient.identifier,
                    event_type='arrival_departure',
                    event='depart'
                )

                # After journey, store totals in results_df
                # Create a dict to fill results_df row
                row = {"patient_id": patient.identifier}
                for state in g.STATES:
                    row[f"q_time_{state.lower()}"] = sum(patient.wait_times[state])
                    row[f"time_in_{state.lower()}"] = sum(patient.service_times[state])

                self.results_df = pd.concat([self.results_df, pd.DataFrame([row]).set_index("patient_id")])


        def interval_audit_utilization(self, resources, interval=1):
            """
            Track resource utilization over time.
            Since VidigiStore doesn't expose .users attribute,
            we calculate utilization based on capacity and items.
            """
            while True:
                for r in resources:
                    resource_obj = r["resource_object"]

                    # VidigiStore is based on SimPy Store
                    # items = list of items currently in the store
                    # For beds, when a patient "gets" a bed, it's removed from store
                    # So utilization = capacity - current items available

                    current_items = len(resource_obj.items) if hasattr(resource_obj, 'items') else 0

                    # DEBUG PRINTS
                    # if r["resource_name"] == "ED":
                    #     print(f"[t={self.env.now}] ED items:", len(self. ed_beds.items))
                    # if r["resource_name"] == "ICU":
                    #     print(f"[t={self.env.now}] ICU items:", len(self. icu_beds.items))
                    # if r["resource_name"] == "INPATIENT":
                    #     print(f"[t={self.env.now}] INPATIENT items:", len(self. inpatient_beds.items))


                    self.utilization_audit.append({
                        'resource_name': r["resource_name"],
                        'simulation_time': self.env.now,
                        # Number in use = capacity minus what's available
                        'number_utilized': resource_obj.capacity - current_items,
                        'number_available': resource_obj.capacity,
                        # Queue length - check if get_queue exists
                        'queue_length': len(resource_obj.get_queue) if hasattr(resource_obj, 'get_queue') else 0,
                    })

                yield self.env.timeout(interval)

        def run(self):
            # Start up our DES entity generators that create new patients.  We've
            # only got one in this model, but we'd need to do this for each one if
            # we had multiple generators.

            for campus in g.arrival_rates.keys():
                self.env.process(self.generator_patient_arrivals(campus)) 

            self.env.process(
                            self.interval_audit_utilization(
                                resources=[
                                    {"resource_name": "ED_LA_JOLLA", "resource_object": self.ed_la_jolla_beds},
                                    {"resource_name": "ICU_LA_JOLLA", "resource_object": self.icu_la_jolla_beds},
                                    {"resource_name": "FLOOR_LA_JOLLA", "resource_object": self.floor_la_jolla_beds},

                                    {"resource_name": "ED_HILLCREST", "resource_object": self.ed_hillcrest_beds},
                                    {"resource_name": "ICU_HILLCREST", "resource_object": self.icu_hillcrest_beds},
                                    {"resource_name": "FLOOR_HILLCREST", "resource_object": self.floor_hillcrest_beds},

                                    {"resource_name": "ED_EASTCAMPUS", "resource_object": self.ed_east_campus_beds},
                                    {"resource_name": "ICU_EASTCAMPUS", "resource_object": self.icu_east_campus_beds},
                                    {"resource_name": "FLOOR_EASTCAMPUS", "resource_object": self.floor_east_campus_beds},
                                ],
                                interval=g.audit_interval
                            )
                        )

            # Run the model for the duration specified in g class
            self.env.run(until=g.sim_duration)
    return (Model,)


@app.cell
def _(Model, pd):
    class Trial:
        def __init__(self, trajectory_library):
            self.patient_library = trajectory_library
            self.df_trial_results = pd.DataFrame(columns=[
                "Mean Queue Time La Jolla ED",
                "Mean Service Time La Jolla ED",
                "Mean Queue Time La Jolla ICU",
                "Mean Service Time La Jolla ICU",
                "Mean Queue Time La Jolla Floor",
                "Mean Service Time  La Jolla Floor",

                "Mean Queue Time Hillcrest ED",
                "Mean Service Time Hillcrest ED",
                "Mean Queue Time Hillcrest ICU",
                "Mean Service Time Hillcrest ICU",
                "Mean Queue Time Hillcrest Floor",
                "Mean Service Time Hillcrest Floor",

                "Mean Queue Time East Campus ED",
                "Mean Service Time East Campus ED",
                "Mean Queue Time East Campus ICU",
                "Mean Service Time East Campus ICU",
                "Mean Queue Time East Campus Floor",
                "Mean Service Time East Campus Floor",
            ])
            self.all_event_logs = [] 
            self.all_event_logs_df = pd.DataFrame() 

        def run_trial(self):
            for run in range(1, g.number_of_runs + 1):
                my_model = Model(run, self.patient_library)
                my_model.run()

                # Compute mean queue and service times for all states
                mean_row = my_model.results_df.mean(axis=0)

                # Save results in trial DataFrame
                self.df_trial_results.loc[run] = {
                    # La Jolla
                    "Mean Queue Time La Jolla ED": mean_row.get("q_time_ed_la_jolla", 0),
                    "Mean Service Time La Jolla ED": mean_row.get("time_in_ed_la_jolla", 0),
                    "Mean Queue Time La Jolla ICU": mean_row.get("q_time_icu_la_jolla", 0),
                    "Mean Service Time La Jolla ICU": mean_row.get("time_in_icu_la_jolla", 0),
                    "Mean Queue Time La Jolla Floor": mean_row.get("q_time_floor_la_jolla", 0),
                    "Mean Service Time  La Jolla Floor": mean_row.get("time_in_floor_la_jolla", 0),

                    # Hillcrest
                    "Mean Queue Time Hillcrest ED": mean_row.get("q_time_ed_hillcrest", 0),
                    "Mean Service Time Hillcrest ED": mean_row.get("time_in_ed_hillcrest", 0),
                    "Mean Queue Time Hillcrest ICU": mean_row.get("q_time_icu_hillcrest", 0),
                    "Mean Service Time Hillcrest ICU": mean_row.get("time_in_icu_hillcrest", 0),
                    "Mean Queue Time Hillcrest Floor": mean_row.get("q_time_floor_hillcrest", 0),
                    "Mean Service Time Hillcrest Floor": mean_row.get("time_in_floor_hillcrest", 0),

                    # East Campus
                    "Mean Queue Time East Campus ED": mean_row.get("q_time_ed_east_campus", 0),
                    "Mean Service Time East Campus ED": mean_row.get("time_in_ed_east_campus", 0),
                    "Mean Queue Time East Campus ICU": mean_row.get("q_time_icu_east_campus", 0),
                    "Mean Service Time East Campus ICU": mean_row.get("time_in_icu_east_campus", 0),
                    "Mean Queue Time East Campus Floor": mean_row.get("q_time_floor_east_campus", 0),
                    "Mean Service Time East Campus Floor": mean_row.get("time_in_floor_east_campus", 0),
                }

                # Optionally, save the event logs
                self.all_event_logs.append(my_model.logger)

            # Combine all logs into one DataFrame
            self.all_event_logs_df = pd.concat([logger.to_dataframe() for logger in self.all_event_logs])
            self.all_event_logs_df.sort_values(by=["time", "entity_id"], inplace=True)
            self.all_event_logs_df.reset_index(drop=True, inplace=True)
    return (Trial,)


@app.cell
def _(Trial, trajectory_library_by_campus):
    my_trial = Trial(trajectory_library_by_campus)

    my_trial.run_trial()
    return (my_trial,)


@app.cell
def _(my_trial):
    single_run_event_log_df = my_trial.all_event_logs_df[my_trial.all_event_logs_df['run_number']==1]
    single_run_event_log_df
    return (single_run_event_log_df,)


@app.cell
def _(pd, plt, single_run_event_log_df):
    def compute_usage(df, resource):
        events = df[(df["event"].str.contains(f"{resource}_begins")) | 
                    (df["event"].str.contains(f"{resource}_ends"))].copy()

        events = events.sort_values("time")

        in_use = 0
        usage = []
        for _, row in events.iterrows():
            if row["event"].endswith("begins"):
                in_use += 1
            else:
                in_use -= 1
            usage.append((row["time"], in_use))

        return pd.DataFrame(usage, columns=["time", "in_use"])


    ed_lj_usage = compute_usage(single_run_event_log_df, "ED_LA_JOLLA")
    icu_lj_usage = compute_usage(single_run_event_log_df, "ICU_LA_JOLLA")
    floor_lj_usage = compute_usage(single_run_event_log_df, "FLOOR_LA_JOLLA")

    ed_hc_usage = compute_usage(single_run_event_log_df, "ED_HILLCREST")
    icu_hc_usage = compute_usage(single_run_event_log_df, "ICU_HILLCREST")
    floor_hc_usage = compute_usage(single_run_event_log_df, "FLOOR_HILLCREST")

    ed_ec_usage = compute_usage(single_run_event_log_df, "ED_EAST_CAMPUS")
    icu_ec_usage = compute_usage(single_run_event_log_df, "ICU_EAST_CAMPUS")
    floor_ec_usage = compute_usage(single_run_event_log_df, "FLOOR_EAST_CAMPUS")

    plt.figure(figsize=(10, 6))

    plt.plot(ed_lj_usage["time"], ed_lj_usage["in_use"], label="La Jolla ED")
    plt.plot(icu_lj_usage["time"], icu_lj_usage["in_use"], label="La Jolla ICU")
    plt.plot(floor_lj_usage["time"], floor_lj_usage["in_use"], label="La Jolla Floor")

    lj_ed_cap = ed_lj_usage["in_use"].max()
    lj_icu_cap = icu_lj_usage["in_use"].max()
    lj_floor_cap = floor_lj_usage["in_use"].max()

    plt.axhline(lj_ed_cap, linestyle="--", alpha=0.5)
    plt.axhline(lj_icu_cap, linestyle="--", alpha=0.5, color = 'orange')
    plt.axhline(lj_floor_cap, linestyle="--", alpha=0.5, color = 'green')



    plt.xlabel("Time (hours)")
    plt.ylabel("Beds in Use")
    plt.title("La Jolla - Resource Usage Over Time")
    plt.legend()
    plt.show()
    return (
        ed_ec_usage,
        ed_hc_usage,
        ed_lj_usage,
        floor_ec_usage,
        floor_hc_usage,
        floor_lj_usage,
        icu_ec_usage,
        icu_hc_usage,
        icu_lj_usage,
        lj_ed_cap,
        lj_floor_cap,
        lj_icu_cap,
    )


@app.cell
def _(ed_hc_usage, floor_hc_usage, icu_hc_usage, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(ed_hc_usage["time"], ed_hc_usage["in_use"], label="Hillcrest ED")
    plt.plot(icu_hc_usage["time"], icu_hc_usage["in_use"], label="Hillcrest ICU")
    plt.plot(floor_hc_usage["time"], floor_hc_usage["in_use"], label="Hillcrest Floor")

    hc_ed_cap = ed_hc_usage["in_use"].max()
    hc_icu_cap = icu_hc_usage["in_use"].max()
    hc_floor_cap = floor_hc_usage["in_use"].max()

    plt.axhline(hc_ed_cap, linestyle="--", alpha=0.5)
    plt.axhline(hc_icu_cap, linestyle="--", alpha=0.5, color = 'orange')
    plt.axhline(hc_floor_cap, linestyle="--", alpha=0.5, color = 'green')



    plt.xlabel("Time (hours)")
    plt.ylabel("Beds in Use")
    plt.title("Hillcrest - Resource Usage Over Time")
    plt.legend()
    plt.show()
    return hc_ed_cap, hc_floor_cap, hc_icu_cap


@app.cell
def _(ed_ec_usage, floor_ec_usage, icu_ec_usage, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(ed_ec_usage["time"], ed_ec_usage["in_use"], label="East Campus ED")
    plt.plot(icu_ec_usage["time"], icu_ec_usage["in_use"], label="East Campus ICU")
    plt.plot(floor_ec_usage["time"], floor_ec_usage["in_use"], label="East Campus Floor")

    ec_ed_cap = ed_ec_usage["in_use"].max()
    ec_icu_cap = icu_ec_usage["in_use"].max()
    ec_floor_cap = floor_ec_usage["in_use"].max()

    plt.axhline(ec_ed_cap, linestyle="--", alpha=0.5)
    plt.axhline(ec_icu_cap, linestyle="--", alpha=0.5, color = 'orange')
    plt.axhline(ec_floor_cap, linestyle="--", alpha=0.5, color = 'green')

    plt.xlabel("Time (hours)")
    plt.ylabel("Beds in Use")
    plt.title("East Campus - Resource Usage Over Time")
    plt.legend()
    plt.show()
    return ec_ed_cap, ec_floor_cap, ec_icu_cap


@app.cell
def _(
    ec_ed_cap,
    ec_floor_cap,
    ec_icu_cap,
    ed_ec_usage,
    ed_hc_usage,
    ed_lj_usage,
    floor_ec_usage,
    floor_hc_usage,
    floor_lj_usage,
    hc_ed_cap,
    hc_floor_cap,
    hc_icu_cap,
    icu_ec_usage,
    icu_hc_usage,
    icu_lj_usage,
    lj_ed_cap,
    lj_floor_cap,
    lj_icu_cap,
    np,
    pd,
):
    def infer_capacity(usage_df):
        u = usage_df.copy()
        u["dt"] = u["time"].shift(-1) - u["time"]

        u = u[u["in_use"] > 0]

        if u.empty:
            return np.nan  # or None

        return (
            u.groupby("in_use")["dt"]
            .sum()
            .idxmax()
        )

    inferred_capacity_df = pd.DataFrame([{
        'ed_lj_usage': infer_capacity(ed_lj_usage),
        'icu_lj_usage': infer_capacity(icu_lj_usage),
        'floor_lj_usage': infer_capacity(floor_lj_usage),
        'ed_ec_usage': infer_capacity(ed_ec_usage),
        'icu_ec_usage': infer_capacity(icu_ec_usage),
        'floor_ec_usage': infer_capacity(floor_ec_usage),
        'ed_hc_usage': infer_capacity(ed_hc_usage),
        'icu_hc_usage': infer_capacity(icu_hc_usage),
        'floor_hc_usage': infer_capacity(floor_hc_usage),
    }]).T



    max_cap_df = pd.DataFrame([{'ed_lj_cap':lj_ed_cap,
                                'icu_lj_cap': lj_icu_cap,
                                'floor_lj_cap': lj_floor_cap,
                                'ed_hc_cap': hc_ed_cap,
                                'icu_hc_cap': hc_icu_cap,
                                'floor_hc_cap': hc_floor_cap,
                                'ed_ec_cap': ec_ed_cap,
                                'icu_ec_cap': ec_icu_cap,
                                'floor_ec_cap': ec_floor_cap,
    }]).T

    max_cap_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Hospital Bed Capacity Parameters

    The following table lists the maximum bed capacities used in the model for each hospital and care level. These values correspond to the maximum observed occupancy under simulation runs (for ~ 2 months) with arrival rates set to fully saturate each resource.

    | Hospital      | ED Capacity | ICU Capacity | Floor Capacity |
    |---------------|-------------|--------------|----------------|
    | La Jolla      | 24          | 15           | 86             |
    | Hillcrest     | 16          | 10           | 56             |
    | East Campus   | 11          | 4            | 21             |
    """)
    return


@app.cell
def _():
    # max_cap_df.to_csv('max_resource_capacity.csv')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
