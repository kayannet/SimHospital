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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This Simulation contains new resource, boarded ED patients. AKA patients who are admitted, but are still physically in the ER
    """)
    return


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
    import joblib
    import pickle

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
        pickle,
        plt,
        random,
        simpy,
    )


@app.cell
def _(pickle):
    with open("../model/los_quantile_models.pkl", "rb") as f:
        models = pickle.load(f)
    return (models,)


@app.cell
def _(json, pd):
    # import all the data needed for sim
    adt_df = pd.read_csv('../data/ADT_cleaned_with_tiers.csv')
    fitted_distributions_df = pd.read_csv('../data/states_fitted_distribution_df.csv')
    arrival_rates_df = pd.read_csv('../data/weekday_weekend_arrival_rates_df.csv')
    loc_trans_prob_df = pd.read_csv('../data/probability_matrix.csv')

    with open('../data/trajectory_library_v2.json', 'r') as json_file:
        trajectory_library = json.load(json_file)
    return (trajectory_library,)


@app.cell
def _(json):
    with open('../data/steady_state_resource_caps.json', 'r') as json_file_2:
        steady_state_caps = json.load(json_file_2)
    steady_state_caps
    return


@app.cell
def _(trajectory_library):
    trajectory_library_by_campus = {
        "LA_JOLLA": {},
        "HILLCREST": {},
        "EAST_CAMPUS": {}
    }

    for enc_id, traj in trajectory_library.items():
        first_state = traj[0]['state']  # first state's name
        campus = "_".join(first_state.split("_")[1:])

        if campus is not None:
            trajectory_library_by_campus[campus][enc_id] = traj
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
    UNITS = ["ED", "FLOOR", "ICU", "BOARDED"]

    STATES = [f"{u}_{h}" for u in UNITS for h in HOSPITALS]
    STATES
    return


@app.class_definition
class g:
    """
    Global model configuration
    """

    UNITS = ["ED", "FLOOR", "ICU", "BOARDED"]
    HOSPITALS = ["EAST_CAMPUS", "HILLCREST", "LA_JOLLA"]

    STATES = [f"{u}_{h}" for u in UNITS for h in HOSPITALS]

    # -------------------------
    # Capacities (by state)
    # -------------------------
    # calculated by taking the max utilized when cap was set to 9999


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
    # capacities = {
    #     "ED_LA_JOLLA": 9999,
    #     "ICU_LA_JOLLA": 9999,
    #     "FLOOR_LA_JOLLA": 9999,
    #     "BOARDED_LA_JOLLA": 9999,

    #     "ED_HILLCREST": 9999,
    #     "ICU_HILLCREST": 9999,
    #     "FLOOR_HILLCREST": 9999,
    #     "BOARDED_HILLCREST": 9999,

    #     "ED_EAST_CAMPUS": 9999,
    #     "ICU_EAST_CAMPUS": 9999,
    #     "FLOOR_EAST_CAMPUS": 9999,
    #     "BOARDED_EAST_CAMPUS": 9999,

    # }

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
    sim_duration = 4320 #6 months #8760 #one year in hours
    number_of_runs = 1
    random_number_set = 42
    audit_interval = 1


@app.cell
def _(models, np, pd):
    def predict_LOS(admit_hospital,
                    time_in_ed, 
                    num_prev_90_days,
                    q,
                   ):

        if q not in [0.25, 0.5, 0.75]:
            return "Invalid Quantile, must be 0.25, 0.5, or 0.75"

        if admit_hospital not in g.HOSPITALS:
            return "Invalid Hospital"

        if not isinstance(num_prev_90_days, int):
            return "Num Prev 90 Days must be interger"

        if not isinstance(time_in_ed, float):
            return "Time in ED must be valid float"



        # Make a single prediction
        X_patient = pd.DataFrame([{
            "admit_hospital": admit_hospital,
            "time_in_ed": time_in_ed,
            "num_prev_90_days": num_prev_90_days
        }])

        pred_log_los = models[q].predict(X_patient)

        # Convert back from log-hours to hours if needed
        pred_los_hours = np.expm1(pred_log_los[0])
        return pred_los_hours


    predict_LOS(
        "LA_JOLLA", 
        30.6, 
        2, 
        0.50
    )
    return


@app.cell
def _(VidigiStore):
    class Hospital:
        def __init__(self, env, campus_name, capacities, arrival_rate):
            self.env = env
            self.campus_name = campus_name
            self.capacities = {}
            self.resources = {}
            self.arrival_rate =arrival_rate

            self._init_resources(capacities)

            # self.mean_q_times = {unit: 0 for unit in units}
            # self.utilization_history = {unit: [] for unit in units}


        def _init_resources(self, capacities):
            for unit in g.UNITS:
                state_key = f"{unit}_{(self.campus_name).upper()}"
                cap = capacities.get(state_key, 0)


                if cap <= 0:
                    continue
                self.capacities[state_key] = cap

                store = VidigiStore(
                    self.env,
                    num_resources=cap,
                    capacity=cap
                )

                self.resources[state_key] = store

        def get_resource(self, state_name):
            return self.resources.get(state_name, None)

        def snapshot_utilization(self):
            """
            Returns utilization snapshot for ALL resources in this hospital.
            No loops over time — just current state.
            """
            snapshot = []

            for name, resource in self.resources.items():
                current_available = len(resource.items)

                snapshot.append({
                    "hospital": self.campus_name,
                    "resource_name": name,
                    "simulation_time": self.env.now,
                    "number_utilized": resource.capacity - current_available,
                    "number_available": resource.capacity,
                    "queue_length": len(resource.get_queue),
                })

            return snapshot

    return (Hospital,)


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
def _(EventLogger, Exponential, Hospital, Patient, np, pd, random, simpy):
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
            self.init_hospitals()

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
            self.utilization_audit = []

        def init_hospitals(self):
            self.hospitals = {}

            for campus, rates in g.arrival_rates.items():
                self.hospitals[campus] = Hospital(
                    env=self.env,
                    campus_name=campus,
                    capacities=g.capacities, 
                    arrival_rate = rates, 
                )

        def get_mean_interarrival(self, campus, current_time):
            day = int(current_time // 24) % 7
            hospital = self.hospitals[campus]

            if day < 5:
                mean_daily = hospital.arrival_rate["weekday"]["mean"]
            else:
                mean_daily = hospital.arrival_rate["weekend"]["mean"]

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


            for step in patient.path:
                state = step['state']
                duration = step['duration_hours']

                unit, campus = state.split("_", 1)
                hospital = self.hospitals[campus]
                resource_pool = hospital.get_resource(state)
                # Map state to resource if you have one
                # resource_pool = self.resources.get(state, None)  # returns None if no resource
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

            all_resources = []
            for hospital in self.hospitals.values():
                for state_name, resource_obj in hospital.resources.items():
                    all_resources.append({
                        "resource_name": state_name,
                        "resource_object": resource_obj
                    })

            # Start audit process
            self.env.process(
                self.interval_audit_utilization(
                    resources=all_resources,
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

            # Dynamically build columns based on g.CAMPUSES and g.UNITS
            columns = []
            for campus in g.HOSPITALS:
                for unit in g.UNITS:
                    columns.append(f"Mean Queue Time {campus} {unit}")
                    columns.append(f"Mean Service Time {campus} {unit}")

            self.df_trial_results = pd.DataFrame(columns=columns)
            self.all_event_logs = [] 
            self.all_event_logs_df = pd.DataFrame() 

        def run_trial(self):
            for run in range(1, g.number_of_runs + 1):
                my_model = Model(run, self.patient_library)
                my_model.run()

                # Compute mean queue and service times for all states
                mean_row = my_model.results_df.mean(axis=0)

                # Dynamically populate the trial results
                row_dict = {}
                for campus in g.HOSPITALS:
                    for unit in g.UNITS:
                        state_key = f"{unit.lower()}_{campus.lower().replace(' ', '_')}"
                        row_dict[f"Mean Queue Time {campus} {unit}"] = mean_row.get(f"q_time_{state_key}", 0)
                        row_dict[f"Mean Service Time {campus} {unit}"] = mean_row.get(f"time_in_{state_key}", 0)

                self.df_trial_results.loc[run] = row_dict

                # Save event logs
                self.all_event_logs.append(my_model.logger)

            # Combine all logs into one DataFrame
            self.all_event_logs_df = pd.concat([logger.to_dataframe() for logger in self.all_event_logs])
            self.all_event_logs_df.sort_values(by=["time", "entity_id"], inplace=True)
            self.all_event_logs_df.reset_index(drop=True, inplace=True)

    return


@app.cell
def _():
    # my_trial = Trial(trajectory_library_by_campus)

    # my_trial.run_trial()
    return


@app.cell
def _(pd):
    # single_run_event_log_df = my_trial.all_event_logs_df[my_trial.all_event_logs_df['run_number']==1]
    single_run_event_log_df = pd.read_csv('../data/six_months_simulation_log_v2.csv')
    single_run_event_log_df
    return (single_run_event_log_df,)


@app.cell
def _():
    # single_run_event_log_df.to_csv('six_months_simulation_log_v2.csv')
    return


@app.cell
def _(np, pd, single_run_event_log_df):
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
        util_df['hour'] = np.floor(util_df['start_time']).astype(int)
        util_df['day'] =  util_df["hour"] // 24


        daily_util = util_df.groupby('day')['in_use'].mean().reset_index()

        daily_util['util_smooth'] = daily_util['in_use'].rolling(window=7, min_periods=1).mean()
        return daily_util



    ed_lj_usage = compute_usage(single_run_event_log_df, "ED_LA_JOLLA")
    icu_lj_usage = compute_usage(single_run_event_log_df, "ICU_LA_JOLLA")
    floor_lj_usage = compute_usage(single_run_event_log_df, "FLOOR_LA_JOLLA")
    boarded_lj_usage = compute_usage(single_run_event_log_df, "BOARDED_LA_JOLLA")

    ed_hc_usage = compute_usage(single_run_event_log_df, "ED_HILLCREST")
    icu_hc_usage = compute_usage(single_run_event_log_df, "ICU_HILLCREST")
    floor_hc_usage = compute_usage(single_run_event_log_df, "FLOOR_HILLCREST")
    boarded_hc_usage = compute_usage(single_run_event_log_df, "BOARDED_HILLCREST")


    ed_ec_usage = compute_usage(single_run_event_log_df, "ED_EAST_CAMPUS")
    icu_ec_usage = compute_usage(single_run_event_log_df, "ICU_EAST_CAMPUS")
    floor_ec_usage = compute_usage(single_run_event_log_df, "FLOOR_EAST_CAMPUS")
    boarded_ec_usage = compute_usage(single_run_event_log_df, "BOARDED_EAST_CAMPUS")
    return (
        boarded_ec_usage,
        boarded_hc_usage,
        boarded_lj_usage,
        ed_ec_usage,
        ed_hc_usage,
        ed_lj_usage,
        floor_ec_usage,
        floor_hc_usage,
        floor_lj_usage,
        icu_ec_usage,
        icu_hc_usage,
        icu_lj_usage,
    )


@app.cell
def _():
    return


@app.cell
def _(
    boarded_ec_usage,
    boarded_hc_usage,
    boarded_lj_usage,
    ed_ec_usage,
    ed_hc_usage,
    ed_lj_usage,
    floor_ec_usage,
    floor_hc_usage,
    floor_lj_usage,
    icu_ec_usage,
    icu_hc_usage,
    icu_lj_usage,
):
    def compute_percent_utilized(usage_df, resource_name):
        df = usage_df.copy()

        df['percent_utilized'] = df['in_use']/g.capacities[resource_name]

        return df

    ed_lj_usage_percent = compute_percent_utilized(ed_lj_usage, "ED_LA_JOLLA")
    icu_lj_usage_percent = compute_percent_utilized(icu_lj_usage, "ICU_LA_JOLLA")
    floor_lj_usage_percent = compute_percent_utilized(floor_lj_usage, "FLOOR_LA_JOLLA")
    boarded_lj_usage_percent = compute_percent_utilized(boarded_lj_usage, "BOARDED_LA_JOLLA")

    ed_hc_usage_percent = compute_percent_utilized(ed_hc_usage, "ED_HILLCREST")
    icu_hc_usage_percent = compute_percent_utilized(icu_hc_usage, "ICU_HILLCREST")
    floor_hc_usage_percent = compute_percent_utilized(floor_hc_usage, "FLOOR_HILLCREST")
    boarded_hc_usage_percent = compute_percent_utilized(boarded_hc_usage, "BOARDED_HILLCREST")

    ed_ec_usage_percent = compute_percent_utilized(ed_ec_usage, "ED_EAST_CAMPUS")
    icu_ec_usage_percent = compute_percent_utilized(icu_ec_usage, "ICU_EAST_CAMPUS")
    floor_ec_usage_percent = compute_percent_utilized(floor_ec_usage, "FLOOR_EAST_CAMPUS")
    boarded_ec_usage_percent = compute_percent_utilized(boarded_ec_usage, "BOARDED_EAST_CAMPUS")
    return (
        boarded_ec_usage_percent,
        boarded_hc_usage_percent,
        boarded_lj_usage_percent,
        ed_ec_usage_percent,
        ed_hc_usage_percent,
        ed_lj_usage_percent,
        floor_ec_usage_percent,
        floor_hc_usage_percent,
        floor_lj_usage_percent,
        icu_ec_usage_percent,
        icu_hc_usage_percent,
        icu_lj_usage_percent,
    )


@app.cell
def _():
    # from our infinite cap run, lets pull the average occupancy after a warm up period of 50 days
    def infer_capacity(usage_df, warm_up_days = 50):
        steady_state_util = usage_df.iloc[warm_up_days:]
        return steady_state_util['utilization'].mean()

    # g.STATES # all resource names

    # steady_state_caps = {}

    # for resource in g.STATES:
    #     util_list = []
    #     for run in range(1, g.number_of_runs +1):
    #         curr_run = my_trial.all_event_logs_df[my_trial.all_event_logs_df['run_number']==run]
    #         usage_df = compute_usage(curr_run, resource)
    #         util_list.append(infer_capacity(usage_df, warm_up_days=50))
    #     steady_state_caps[resource] = np.mean(util_list)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### La Jolla Utilization Plots
    """)
    return


@app.cell
def _(boarded_lj_usage, ed_lj_usage, floor_lj_usage, icu_lj_usage, plt):
    plt.figure(figsize=(20, 8))

    plt.plot(ed_lj_usage["day"], ed_lj_usage["util_smooth"], label="La Jolla ED", color = 'blue')
    plt.plot(icu_lj_usage["day"], icu_lj_usage["util_smooth"], label="La Jolla ICU", color = 'orange')
    plt.plot(floor_lj_usage["day"], floor_lj_usage["util_smooth"], label="La Jolla Floor", color = 'green')
    plt.plot(boarded_lj_usage["day"], boarded_lj_usage["util_smooth"], label="La Jolla Boarded", color = 'red')




    # plt.axhline(g.capacities['ED_LA_JOLLA'], linestyle="--", alpha=0.5, color =  'blue',)
    # plt.axhline(g.capacities['ICU_LA_JOLLA'], linestyle="--", alpha=0.5, color = 'orange')
    # plt.axhline(g.capacities['FLOOR_LA_JOLLA'], linestyle="--", alpha=0.5, color = 'green')
    # plt.axhline(g.capacities['BOARDED_LA_JOLLA'], linestyle="--", alpha=0.5, color = 'red')



    plt.xlabel("Time (days)")
    plt.ylabel("Bed Utilization")
    plt.title("La Jolla - Resource Usage Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _(ed_lj_usage_percent):
    ed_lj_usage_percent
    return


@app.cell
def _(
    boarded_lj_usage_percent,
    ed_lj_usage_percent,
    floor_lj_usage_percent,
    icu_lj_usage_percent,
    plt,
):

    plt.figure(figsize=(25, 8))
    plt.axhspan(0.8, 1.0, color="gray", alpha=0.2)

    plt.plot(ed_lj_usage_percent["day"], ed_lj_usage_percent["percent_utilized"], label="La Jolla ED", color = 'blue')
    plt.plot(icu_lj_usage_percent["day"], icu_lj_usage_percent["percent_utilized"], label="La Jolla ICU", color = 'orange')
    plt.plot(floor_lj_usage_percent["day"], floor_lj_usage_percent["percent_utilized"], label="La Jolla Floor", color = 'green')
    plt.plot(boarded_lj_usage_percent["day"], boarded_lj_usage_percent["percent_utilized"], label="La Jolla Boarded", color = 'red')


    plt.xlabel("Time (days)")
    plt.ylabel("Percent Utilized")
    plt.ylim(0,1.1)
    plt.title("La Jolla - Percent Utilization Over Time")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Hillcrest Utilization Plots
    """)
    return


@app.cell
def _(boarded_hc_usage, ed_hc_usage, floor_hc_usage, icu_hc_usage, plt):
    plt.figure(figsize=(20, 8))
    plt.plot(ed_hc_usage["day"], ed_hc_usage["util_smooth"], label="Hillcrest ED")
    plt.plot(icu_hc_usage["day"], icu_hc_usage["util_smooth"], label="Hillcrest ICU")
    plt.plot(floor_hc_usage["day"], floor_hc_usage["util_smooth"], label="Hillcrest Floor")
    plt.plot(boarded_hc_usage["day"], boarded_hc_usage["util_smooth"], label="Hillcrest Boarded")


    # plt.axhline(g.capacities['ED_HILLCREST'], linestyle="--", alpha=0.5)
    # plt.axhline(g.capacities['ICU_HILLCREST'], linestyle="--", alpha=0.5, color = 'orange')
    # plt.axhline(g.capacities['FLOOR_HILLCREST'], linestyle="--", alpha=0.5, color = 'green')
    # plt.axhline(g.capacities['BOARDED_HILLCREST'], linestyle="--", alpha=0.5, color = 'red')


    plt.xlabel("Time (days)")
    plt.ylabel("Average Beds in Use")
    plt.title("Hillcrest - Average Daily Resource Use Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    boarded_hc_usage_percent,
    ed_hc_usage_percent,
    floor_hc_usage_percent,
    icu_hc_usage_percent,
    plt,
):
    plt.figure(figsize=(25, 8))
    plt.axhspan(0.8, 1.0, color="gray", alpha=0.2)

    plt.plot(ed_hc_usage_percent["day"], ed_hc_usage_percent["percent_utilized"], label="Hillcrest ED", color = 'blue')
    plt.plot(icu_hc_usage_percent["day"], icu_hc_usage_percent["percent_utilized"], label="Hillcrest ICU", color = 'orange')
    plt.plot(floor_hc_usage_percent["day"], floor_hc_usage_percent["percent_utilized"], label="Hillcrest Floor", color = 'green')
    plt.plot(boarded_hc_usage_percent["day"], boarded_hc_usage_percent["percent_utilized"], label="Hillcrest Boarded", color = 'red')


    plt.xlabel("Time (days)")
    plt.ylabel("Percent Utilized")
    plt.ylim(0,1.1)
    plt.title("Hillcrest - Percent Utilization Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _(boarded_ec_usage, ed_ec_usage, floor_ec_usage, icu_ec_usage, plt):
    plt.figure(figsize=(20, 8))
    plt.plot(ed_ec_usage["day"], ed_ec_usage["util_smooth"], label="East Campus ED")
    plt.plot(icu_ec_usage["day"], icu_ec_usage["util_smooth"], label="East Campus ICU")
    plt.plot(floor_ec_usage["day"], floor_ec_usage["util_smooth"], label="East Campus Floor")
    plt.plot(boarded_ec_usage["day"], boarded_ec_usage["util_smooth"], label="East Campus Boarded")

    # plt.axhline(g.capacities['ED_EAST_CAMPUS'], linestyle="--", alpha=0.5)
    # plt.axhline(g.capacities['ICU_EAST_CAMPUS'], linestyle="--", alpha=0.5, color = 'orange')
    # plt.axhline(g.capacities['FLOOR_EAST_CAMPUS'], linestyle="--", alpha=0.5, color = 'green')
    # plt.axhline(g.capacities['BOARDED_EAST_CAMPUS'], linestyle="--", alpha=0.5, color = 'red')


    plt.xlabel("Time (days)")
    plt.ylabel("Average Beds in Use")
    plt.title("East Campus - Average Daily Resource Use Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _(
    boarded_ec_usage_percent,
    ed_ec_usage_percent,
    floor_ec_usage_percent,
    icu_ec_usage_percent,
    plt,
):
    plt.figure(figsize=(25, 8))
    plt.axhspan(0.8, 1.0, color="gray", alpha=0.2)

    plt.plot(ed_ec_usage_percent["day"], ed_ec_usage_percent["percent_utilized"], label="East Campus ED", color = 'blue')
    plt.plot(icu_ec_usage_percent["day"], icu_ec_usage_percent["percent_utilized"], label="East Campus ICU", color = 'orange')
    plt.plot(floor_ec_usage_percent["day"], floor_ec_usage_percent["percent_utilized"], label="East Campus Floor", color = 'green')
    plt.plot(boarded_ec_usage_percent["day"], boarded_ec_usage_percent["percent_utilized"], label="East Campus Boarded", color = 'red')


    plt.xlabel("Time (days)")
    plt.ylabel("Percent Utilized")
    plt.ylim(0,1.1)
    plt.title("East Campus - Percent Utilization Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    # def infer_capacity(usage_df):
    #     u = usage_df.copy()
    #     u["dt"] = u["time"].shift(-1) - u["time"]

    #     u = u[u["in_use"] > 0]

    #     if u.empty:
    #         return np.nan  # or None

    #     return (
    #         u.groupby("in_use")["dt"]
    #         .sum()
    #         .idxmax()
    #     )

    # inferred_capacity_df = pd.DataFrame([{
    #     'ed_lj_usage': infer_capacity(ed_lj_usage),
    #     'icu_lj_usage': infer_capacity(icu_lj_usage),
    #     'floor_lj_usage': infer_capacity(floor_lj_usage),
    #     'ed_ec_usage': infer_capacity(ed_ec_usage),
    #     'icu_ec_usage': infer_capacity(icu_ec_usage),
    #     'floor_ec_usage': infer_capacity(floor_ec_usage),
    #     'ed_hc_usage': infer_capacity(ed_hc_usage),
    #     'icu_hc_usage': infer_capacity(icu_hc_usage),
    #     'floor_hc_usage': infer_capacity(floor_hc_usage),
    # }]).T



    # max_cap_df = pd.DataFrame([{'ed_lj_cap':lj_ed_cap,
    #                             'icu_lj_cap': lj_icu_cap,
    #                             'floor_lj_cap': lj_floor_cap,
    #                             'boarded_lj_cap': lj_boarded_cap,
    #                             'ed_hc_cap': hc_ed_cap,
    #                             'icu_hc_cap': hc_icu_cap,
    #                             'floor_hc_cap': hc_floor_cap,
    #                             'boarded_hc_cap': hc_boarded_cap,
    #                             'ed_ec_cap': ec_ed_cap,
    #                             'icu_ec_cap': ec_icu_cap,
    #                             'floor_ec_cap': ec_floor_cap,
    #                             'boarded_ec_cap': ec_boarded_cap,
    # }]).T

    # max_cap_df#.to_csv('max_resource_capacity_v2.csv')
    return


@app.cell
def _(
    boarded_ec_usage_percent,
    boarded_hc_usage_percent,
    boarded_lj_usage_percent,
    ed_ec_usage_percent,
    ed_hc_usage_percent,
    ed_lj_usage_percent,
    floor_ec_usage_percent,
    floor_hc_usage_percent,
    floor_lj_usage_percent,
    icu_ec_usage_percent,
    icu_hc_usage_percent,
    icu_lj_usage_percent,
    pd,
):
    def find_percent_above_80_util(df):
        df["in_range"] = (df["percent_utilized"] >= 0.8) & (df["percent_utilized"] <= 1.0)
        return df['in_range'].mean()



    percent_above_80 = {
                        'ED_LA_JOLLA': find_percent_above_80_util(ed_lj_usage_percent),
                        'ICU_LA_JOLLA': find_percent_above_80_util(icu_lj_usage_percent),
                        'FLOOR_LA_JOLLA': find_percent_above_80_util(floor_lj_usage_percent),
                        'BOARDED_LA_JOLLA': find_percent_above_80_util(boarded_lj_usage_percent),

                        'ED_HILLCREST': find_percent_above_80_util(ed_hc_usage_percent),
                        'ICU_HILLCREST': find_percent_above_80_util(icu_hc_usage_percent),
                        'FLOOR_HILLCREST': find_percent_above_80_util(floor_hc_usage_percent),
                        'BOARDED_HILLCREST': find_percent_above_80_util(boarded_hc_usage_percent),

                        'ED_EAST_CAMPUS': find_percent_above_80_util(ed_ec_usage_percent),
                        'ICU_EAST_CAMPUS': find_percent_above_80_util(icu_ec_usage_percent),
                        'FLOOR_EAST_CAMPUS': find_percent_above_80_util(floor_ec_usage_percent),
                        'BOARDED_EAST_CAMPUS': find_percent_above_80_util(boarded_ec_usage_percent),
                    }

    percent_above_80_df = pd.DataFrame(
        list(percent_above_80.items()),  # convert dict to list of (key, value) tuples
        columns=['resource', 'percent_above_80_util']
    )


    percent_above_80_df
    return (percent_above_80_df,)


@app.cell
def _(percent_above_80_df, plt):
    plt.figure(figsize = (10, 6))
    plt.bar(
        x = percent_above_80_df['resource'], 
        height =  percent_above_80_df['percent_above_80_util'], 
    )
    plt.title('Proportion of time over 80% Utilization')
    plt.ylabel('Proportion')
    plt.xlabel('Resource')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluating Baseline_v2 with Boarded Resources
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LOS and Service Times by Resource
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Average LOS by patient
    There'is a median stay of 5.3 hours (across all hospitals)
    """)
    return


@app.cell
def _(single_run_event_log_df):
    enter_time = single_run_event_log_df.groupby('entity_id')['time'].min()

    exit_time = single_run_event_log_df.groupby('entity_id')['time'].max()


    duration_by_entity = exit_time - enter_time
    duration_by_entity.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Average service time per resource

    - We can see that patients spend the most time on average on the floor of all three hospitals
    - La Jolla ED has boarded patients stay longer than the other two hospitals
    """)
    return


@app.cell
def _(pd, plt, single_run_event_log_df):
    service_start = single_run_event_log_df[
                (single_run_event_log_df['event_type']
    =='resource_use'     ) &
               ( single_run_event_log_df["event"].str.endswith("_begins") )
            ].copy()

    service_end = single_run_event_log_df[
        (single_run_event_log_df['event_type']
    =='resource_use_end'     ) &
                single_run_event_log_df["event"].str.endswith("_ends")
            ].copy()

    service_start = service_start.rename(columns={"time": "time_start"})
    service_end = service_end.rename(columns={"time": "time_end"})

    service = pd.merge_asof(
        service_start.sort_values("time_start"),
        service_end.sort_values("time_end"),
        by="entity_id",
        left_on="time_start",
        right_on="time_end",
        direction="forward",
        allow_exact_matches=False,
        suffixes=("_start", "_end")
    )

    service["service_time"] = service["time_end"] - service["time_start"]
    service['resource'] = service['event_start'].str.split("_").str[:-1].str.join('_')
    average_service_time_per_resource = service.groupby('resource')['service_time'].mean().reset_index()

    plt.figure(figsize = (10, 6))
    plt.bar(
        x = average_service_time_per_resource['resource'], 
        height = average_service_time_per_resource['service_time'], 
    )
    plt.xticks(rotation = 45)
    plt.xlabel("Resource")
    plt.ylabel('Time (hours)')
    plt.title("Average Time Spent in Each Resource (hours)")
    plt.tight_layout()
    plt.show()
    plt.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wait times
    """)
    return


@app.cell
def _(single_run_event_log_df):
    single_run_event_log_df[single_run_event_log_df['entity_id']==50]
    return


@app.cell
def _(single_run_event_log_df):
    single_run_event_log_df["resource"] = (
        single_run_event_log_df["event"]
        .str.replace("_wait_begins", "", regex=False)
        .str.replace("_begins", "", regex=False)
        .str.replace("_ends", "", regex=False)
    )
    single_run_event_log_df
    return


@app.cell
def _(single_run_event_log_df):
    single_run_event_log_df[single_run_event_log_df['entity_id']==31570]
    return


@app.cell
def _(single_run_event_log_df):
    queue_events = single_run_event_log_df.sort_values(["entity_id", "time"]).copy()
    queue_events["next_event"] = queue_events.groupby("entity_id")["event"].shift(-1)
    queue_events["next_time"] = queue_events.groupby("entity_id")["time"].shift(-1)
    queue_events["next_resource"] = queue_events.groupby("entity_id")["resource"].shift(-1)

    queue_events = queue_events[queue_events['event_type']=='queue']
    queue_events

    # sanity check, do all queue_events line up to the correct next resource use
    still_in_system = queue_events[queue_events["resource"] != queue_events["next_resource"]]
    print("Number of patients still in system", len(still_in_system))

    queue_events["wait_time"] = queue_events["next_time"] - queue_events["time"]
    # # No negative waits
    # # assert (queue_events["wait_time"] >= 0).all()

    # # How many real waits?
    print("Percent of events that actually had a wait time:", (queue_events["wait_time"] > 0).mean())

    # # Distribution
    queue_events["wait_time"].describe()
    return (queue_events,)


@app.cell
def _(plt, queue_events):
    all_resources = queue_events['resource'].unique()

    # Group by resource, compute mean wait
    avg_wait_per_resource = (
        queue_events.groupby('resource')['wait_time']
        .mean()
        .reindex(all_resources, fill_value=0)  # keep all resources, fill missing with 0
        .reset_index()
        .sort_values(by = ['resource'])
    )

    avg_wait_per_resource['wait_time_minutes'] = avg_wait_per_resource['wait_time']*60

    plt.figure(figsize=(10, 6))
    plt.bar(
        x = avg_wait_per_resource['resource'], 
        height = avg_wait_per_resource['wait_time']
    )
    plt.title('Average Wait Time per Resource (hours)')
    plt.xlabel('Resource')
    plt.ylabel('Time (hours)')
    plt.xticks(rotation = 45 )
    plt.tight_layout()
    plt.show()
    plt.close()
    # avg_wait_per_resource
    return


@app.cell
def _(plt, queue_events):
    wait_by_resource = (
        queue_events
        .assign(waited=lambda x: x["wait_time"] > 0)
        .groupby("resource")["waited"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    wait_by_resource
    plt.figure(figsize = (10, 6))
    plt.bar(
        x = wait_by_resource['resource'],
        height =   wait_by_resource['waited']
           )
    plt.ylabel("Proportion of queue events with wait")
    plt.xticks(rotation = 45)
    plt.title("Wait rate by resource")
    plt.ylim(0, wait_by_resource['waited'].max() * 1.2)
    plt.show()
    plt.tight_layout()
    plt.close()
    return


if __name__ == "__main__":
    app.run()
