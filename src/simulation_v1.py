# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ibis-framework[duckdb]==11.0.0",
#     "marimo>=0.19.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "sim-tools==1.0.1",
#     "simpy==4.1.1",
#     "vidigi==1.1.1",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import ibis
    import ibis.selectors as s
    from ibis import _
    ibis.options.interactive = True

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

    import marimo as mo
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    return (
        EventLogger,
        EventPosition,
        Exponential,
        VidigiStore,
        animate_activity_log,
        create_event_position_df,
        ibis,
        mo,
        np,
        pd,
        plt,
        random,
        simpy,
    )


@app.cell
def _(pd):
    activities_df = pd.read_csv('MIMIC_ED/processed/all_transfers_df.csv')

    activities_df['hadm_id'].fillna(0, inplace=True)

    activities_df
    return (activities_df,)


@app.cell
def _(activities_df):
    activities_df['careunit'].unique()
    return


@app.cell
def _(activities_df, np):
    # simplify locations to: ED, ICU, INPATIENT, DISCHARGED
    # Define simplified location rules
    conditions = [
        activities_df['careunit'].str.lower().str.contains('emergency', case=False, na=False),
        activities_df['careunit'].str.lower().str.contains('icu', case=False, na=False),
        activities_df['eventtype'].str.contains('discharge', case=False, na=False)
    ]

    choices = ['ED', 'ICU', 'DISCHARGE']

    # Apply mapping, default to 'inpatient'
    activities_df['simplified_location'] = np.select(conditions, choices, default='INPATIENT')
    return


@app.cell
def _(activities_df, np):
    activities_df["encounter_id"] = np.where(
        activities_df["hadm_id"] != 0,
        activities_df["subject_id"].astype(str) + "_" + activities_df["hadm_id"].astype(str),
        "EDONLY_" + activities_df["transfer_id"].astype(str)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Only 8 rows have negative durations, let's just drop them
    """)
    return


@app.cell
def _(activities_df):
    neg_rows = activities_df[activities_df['duration_hours'] < 0]
    neg_rows
    return


@app.cell
def _(activities_df):
    activities_df_clean = activities_df[activities_df['duration_hours'] >= 0]
    return (activities_df_clean,)


@app.cell
def _(activities_df_clean, ibis):
    activities_memtable = ibis.memtable(activities_df_clean)
    activities_memtable
    return (activities_memtable,)


@app.cell
def _(activities_memtable, ibis):
    # import ibis

    # t = activities_memtable  # your memtable

    w = ibis.window(
        group_by=activities_memtable.encounter_id,
        order_by=activities_memtable.intime
    )

    collapsed = (
        activities_memtable
        .mutate(
            prev_loc=activities_memtable.simplified_location.lag().over(w),
            is_new=(activities_memtable.simplified_location != activities_memtable.simplified_location.lag().over(w)).ifelse(1, 0),
        )
        .mutate(block=_.is_new.cumsum().over(w))
        .group_by(["encounter_id", "block", "simplified_location"])
        .agg(duration_hours=_.duration_hours.sum())
        .order_by(["encounter_id", "block"])
    )
    return (collapsed,)


@app.cell
def _(collapsed):
    collapsed
    collapsed_pd = collapsed.execute()
    collapsed_library = (
        collapsed_pd
        .sort_values(["encounter_id", "block"])
        .groupby("encounter_id")
        .apply(lambda g: list(zip(g["simplified_location"], g["duration_hours"])))
        .to_dict()
    )
    collapsed_library
    return (collapsed_library,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each row correlates to a unique transfer_id so we will sample that, then pull the subject_id, hadm_id pair correlating to the sampled transfer_id



    if hadm_id == 0 for that transfer_id, then we know that the encounter sequence is only one activity long, only need to use that one row
    """)
    return


@app.cell
def _(activities_df):
    activities_df['transfer_id'].nunique() == activities_df.shape[0]
    return


@app.cell
def _(collapsed_library):
    # acuity flags
    acuity_flags = {
        encounter_id: any(location == "ICU" for location, _ in path)
        for encounter_id, path in collapsed_library.items()
    }
    acuity_flags
    return (acuity_flags,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DISCRETE EVENT SIM
    """)
    return


@app.class_definition
class g:
    # H1
    ed_bed_cap = 60 # The number of ED beds
    inpatient_bed_cap = 250
    icu_bed_cap = 40

    # H2 (% of H1 inpatient)
    h2_inpatient_bed_cap = int(inpatient_bed_cap * 0.3)

    # H3  (% of H1 inpatient)
    h3_inpatient_bed_cap =  int(inpatient_bed_cap * 0.3)

    # Arrival rate (placeholder, used Karandeep's code ~3.33 patients per hour)
    arrival_rate = 0.15


    # deteriation rate
    deteriation_rate = 0.05
    

    # Simulation running parameters
    sim_duration = 168 # The number of time units the simulation will run for
    number_of_runs = 1 # The number of times the simulation will be run with different random number streams
    random_number_set = 42

    audit_interval = 1


@app.class_definition
class Patient:
    def __init__(self, p_id, encounter_id, path, ever_icu):
        self.identifier = p_id
        # Real-world identifier
        self.encounter_id = encounter_id  # formerly transfer_id
        # Patient journey: list of (location, duration)
        self.path = path  
        # Outcomes
        self.arrival = None
        self.total_time = None
        self.ed_wait_times = []           # Changed from 0.0 to [], it is appending and can't append float it needs to be list
        self.icu_wait_times = []          # Changed from 0.0 to []
        self.inpatient_wait_times = []    # Changed from 0.0 to []
        self.ed_service_times = []        # Changed from 0.0 to []
        self.icu_service_times = []       # Changed from 0.0 to []
        self.inpatient_service_times = []

        # NEW!

        # logging lists for h2 and h3
        self.h2_inpatient_wait_times = []    # Changed from 0.0 to []
        self.h2_inpatient_service_times = []

        self.h3_inpatient_wait_times = []    # Changed from 0.0 to []
        self.h3_inpatient_service_times = []

        
        self.ever_icu = ever_icu
        # can be assigned in assign_initial_hosp
        self.assigned_hospital = None
        
        # list of hospitals they've ever been to
        self.transfer_list = []

        # deteriation/transfer logics
        deterioration_transfer_prob = 0.95
        overflow_transfer_prob = 0.5


@app.cell
def _(EventLogger, Exponential, VidigiStore, pd, random, simpy):
    class Model:
        def __init__(self, run_number, df, acuity_flags):
            # Create a SimPy environment in which everything will live
            self.env = simpy.Environment()

            # Create a patient counter (which we'll use as a patient ID)
            self.patient_counter = 0
            self.patient_objects = []

            # Create an empty list to store our patient objects - these can be handy
            # to look at later
            self.patients = []

            # Create our resources
            self.init_resources()

            # Resource monitoring
            # self.resource_log = []

            # Store the passed in run number
            self.run_number = run_number

            # Inter-arrival distribution (still synthetic)
            self.patient_inter_arrival_dist = Exponential(
                mean=g.arrival_rate,
                random_seed=(self.run_number + 1) * g.random_number_set
            )

            # Build patient library from collapsed activity library
            self.patient_library = df
            # Precompute all possible real patients
            self.unique_encounters = list(self.patient_library.keys())

            # Logger
            self.logger = EventLogger(
                env=self.env,
                run_number=self.run_number
            )

            self.acuity_flags = acuity_flags

    

            # new pandas df that will store results against patient ID
            self.results_df = pd.DataFrame()
            self.results_df['Patient ID'] = [1]
            self.results_df['Q Time ED'] = [0.0]
            self.results_df['Time in ED'] = [0.0]
            self.results_df['Q Time ICU'] = [0.0]
            self.results_df['Time in ICU'] = [0.0]
            self.results_df['Q Time INPATIENT'] = [0.0]
            self.results_df['Time in INPATIENT'] = [0.0]

            # adding results col for h2 and h3 resource
            self.results_df['Q Time H2 INPATIENT'] = [0.0]
            self.results_df['Time in H2 INPATIENT'] = [0.0]
            self.results_df['Q Time H3 INPATIENT'] = [0.0]
            self.results_df['Time in H3 INPATIENT'] = [0.0]
        
            self.results_df.set_index('Patient ID', inplace = True)

            self.mean_q_time_ed = 0
            self.mean_q_time_icu = 0
            self.mean_q_time_inpatient = 0

            # mean times for h2 and h3
            self.mean_q_time_h2_inpatient = 0
            self.mean_q_time_h3_inpatient = 0


            self.utilization_audit = []

            # need to add these for utilization plot 
            self.ed_utilization = []
            self.icu_utilization = []
            self.inpatient_utilization = []

            # H2 and H3 inpatient util
            self.h2_inpatient_utilization = []
            self.h3_inpatient_utilization = []



        def init_resources(self):
            '''
            Init the number of resources

            Resource list:
                1. Nurses/treatment bays (same thing in this model)

            '''
            # without vidigi
            # self.treatment_cubicles = simpy.Resource(self.env, capacity=g.n_cubicles)

            # with vidigi
            self.ed_beds = VidigiStore( 
                self.env, 
                num_resources=g.ed_bed_cap, 
                capacity = g.ed_bed_cap
                ) 
            self.icu_beds = VidigiStore( 
                self.env, 
                num_resources=g.icu_bed_cap, 
                capacity = g.icu_bed_cap
                ) 
            self.inpatient_beds = VidigiStore( 
                self.env, 
                num_resources=g.inpatient_bed_cap, 
                capacity = g.inpatient_bed_cap
                ) 

            # adding resource for h2 and h3
            self.h2_inpatient_beds = VidigiStore( 
                self.env, 
                num_resources= g.h2_inpatient_bed_cap, 
                capacity = g.h2_inpatient_bed_cap
                )  
        

            self.h3_inpatient_beds = VidigiStore( 
                self.env, 
                num_resources=g.h3_inpatient_bed_cap, 
                capacity = g.h3_inpatient_bed_cap
                )  



        def generator_patient_arrivals(self):
            # We use an infinite loop here to keep doing this indefinitely whilst
            # the simulation runs
            while True:
                # Increment the patient counter by 1 (this means our first patient
                # will have an ID of 1)
                self.patient_counter += 1

                # Create a new patient - an instance of the Patient Class we
                # defined above. 
                # Sample random transfer_id (unique_encounters), pull correlating subject_id and hadm_id, then pull pathway_df
                encounter_id = random.choice(self.unique_encounters)
                path = self.patient_library[encounter_id]

                p = Patient(
                    p_id=self.patient_counter,
                    encounter_id=encounter_id,
                    path=path,
                    ever_icu = self.acuity_flags[encounter_id], # track if this patient has ever been to icu from MIMIC collapsed lib
                ) 

                self.patient_objects.append(p)

                # Store patient in our patient list for later easy access
                self.patients.append(p)

                # Tell SimPy to start up the patient_journey generator function with
                # this patient (the generator function that will model the
                # patient's journey through the system)
                self.env.process(self.patient_journey(p))

                # Randomly sample the time to the next patient arriving.  Here, we
                # sample from an exponential distribution (common for inter-arrival times)
                sampled_inter = self.patient_inter_arrival_dist.sample()

                # Freeze this instance of this function in place until the
                # inter-arrival time we sampled above has elapsed.
                # Note - time in SimPy progresses in "Time Units", which can represent anything
                # you like - just make sure you're consistent within the model
                yield self.env.timeout(sampled_inter)


        # quickly defined, can be changed however
        def assign_initial_hospital(self, patient):
            if patient.ever_icu:
                patient.assigned_hospital = "H1"
            else:
                patient.assigned_hospital = random.choice(["H2_INPATIENT", "H3_INPATIENT"])


        def patient_journey(self, patient):
            patient.arrival = self.env.now
            self.logger.log_arrival(entity_id=patient.identifier)

           # Map location to resource pool and patient attributes
            resource_map = {
                'ED': (self. ed_beds, 
                        'ed_wait_times', 
                        'ed_service_times', 
                        self.ed_utilization, 
                        self.results_df['Q Time ED'], 
                        self.results_df['Time in ED'],
                      ),
                'ICU': (self.icu_beds, 
                        'icu_wait_times', 
                        'icu_service_times', 
                        self.icu_utilization, 
                        self.results_df['Q Time ICU'], 
                        self.results_df['Time in ICU'],
                       ),

                'INPATIENT': (self.inpatient_beds, 
                              'inpatient_wait_times',
                              'inpatient_service_times', 
                              self.inpatient_utilization,
                              self.results_df['Q Time INPATIENT'],
                              self.results_df['Time in INPATIENT'],
                             ),

                'H2_INPATIENT': (self.h2_inpatient_beds, 
                              'h2_inpatient_wait_times',
                              'h2_inpatient_service_times', 
                              self.h2_inpatient_utilization,
                              self.results_df['Q Time H2 INPATIENT'],
                              self.results_df['Time in H2 INPATIENT'],
                             ), 
                'H3_INPATIENT': (self.h3_inpatient_beds, 
                              'h3_inpatient_wait_times',
                              'h3_inpatient_service_times', 
                              self.h3_inpatient_utilization,
                              self.results_df['Q Time H3 INPATIENT'],
                              self.results_df['Time in H3 INPATIENT'],
                             ), 
            }

            # need to restructure this to not follow the path directly, based on transfer probs
            for location, duration in patient.path:
                # If location is a resource, get pool and patient attribute names
                if location in resource_map:
                    resource_pool, wait_attr, service_attr, _, _, _ = resource_map[location]

                    start_wait = self.env.now
                    self.logger.log_queue(entity_id=patient.identifier, 
                                          event_type = 'queue',
                                          event=f"{location}_wait_begins")

                    with resource_pool.request() as req:
                        bed = yield req
                    
                        # debug print
                        # in_use = resource_pool.capacity - len(resource_pool.items)
                        # available = len(resource_pool.items)
                        # print(f"[t={self.env.now}] {location} in use: {in_use}, available: {available}")

                        wait_time = self.env.now - start_wait
                        getattr(patient, wait_attr).append(wait_time)

                    

                        self.logger.log_resource_use_start(
                            entity_id=patient.identifier,
                            event_type = 'resource_use',
                            event=f"{location}_begins",
                            resource_id=bed.id_attribute
                        )

                        yield self.env.timeout(duration)
                        getattr(patient, service_attr).append(duration)

                        self.logger.log_resource_use_end(
                            entity_id=patient.identifier,
                            event_type = 'resource_use_end',
                            event=f"{location}_ends",
                            resource_id=bed.id_attribute
                        )
                    

                else:
                    # Non-resource activity
                    self.logger.log_event(entity_id=patient.identifier,
                                          event_type = 'resource_use',
                                          event=f"{location}_begins")
                    yield self.env.timeout(duration)
                    self.logger.log_event(entity_id=patient.identifier, 
                                          event_type = 'resource_use_end',
                                          event=f"{location}_ends")

            patient.total_time = self.env.now - patient.arrival
            self.logger.log_event(
                entity_id=patient.identifier,
                event_type='arrival_departure',
                event='depart'
    )
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
            self.env.process(self.generator_patient_arrivals())
            self.env.process(
                            self.interval_audit_utilization(
                                resources=[
                                    {"resource_name": "ED", "resource_object": self. ed_beds},
                                    {"resource_name": "ICU", "resource_object": self. icu_beds},
                                    {"resource_name": "INPATIENT", "resource_object": self.inpatient_beds},
                                    {"resource_name": "H2_INPATIENT", "resource_object": self.h2_inpatient_beds},
                                    {"resource_name": "H3_INPATIENT", "resource_object": self.h3_inpatient_beds},
                                
                                ],
                                interval=g.audit_interval
                            )
            )
            # Run the model for the duration specified in g class
            self.env.run(until=g.sim_duration)
    return (Model,)


@app.cell
def _(Model, np, pd):
    class Trial:
        def __init__(self, df, acuity_flags):
            self.patient_library = df
            self.acuity_flags = acuity_flags
            self.df_trial_results = pd.DataFrame(columns=[
                "Mean Queue Time ED",
                "Mean Service Time ED",
                "Mean Queue Time ICU",
                "Mean Service Time ICU",
                "Mean Queue Time Inpatient",
                "Mean Service Time Inpatient",
                # "Mean Queue Time H2 Inpatient", 
                # "Mean Service Time H2 Inpatient", 
                # "Mean Queue Time H3 Inpatient", 
                # "Mean Service Time H3 Inpatient", 
                "Mean Total Time"
            ])
            self.all_event_logs = [] 
            self.all_event_logs_df = pd.DataFrame() 

        def run_trial(self):
            for run in range(1, g.number_of_runs + 1):
                my_model = Model(run, self.patient_library, self.acuity_flags)
                my_model.run()

                # Compute mean waits
                mean_wait_ED = np.mean([np.mean(p.ed_wait_times) if p.ed_wait_times else 0 for p in my_model.patients])
                mean_wait_ICU = np.mean([np.mean(p.icu_wait_times) if p.icu_wait_times else 0 for p in my_model.patients])
                mean_wait_inpatient = np.mean([np.mean(p.inpatient_wait_times) if p.inpatient_wait_times else 0 for p in my_model.patients])

                ## mean wait times for H2, H3, commented out for now
                # mean_wait_h2_inpatient = np.mean([np.mean(p.h2_inpatient_wait_times) if p.h2_inpatient_wait_times else 0 for p in my_model.patients])
                # mean_wait_h3_inpatient = np.mean([np.mean(p.h3_inpatient_wait_times) if p.h3_inpatient_wait_times else 0 for p in my_model.patients])

                # Compute mean service times
                mean_service_ED = np.mean([np.mean(p.ed_service_times) if p.ed_service_times else 0 for p in my_model.patients])
                mean_service_ICU = np.mean([np.mean(p.icu_service_times) if p.icu_service_times else 0 for p in my_model.patients])
                mean_service_inpatient = np.mean([np.mean(p.inpatient_service_times) if p.inpatient_service_times else 0 for p in my_model.patients])


            ###  mean service times for h2, h3 (commented out for now)
               # mean_service_h2_inpatient = np.mean([np.mean(p.h2_inpatient_service_times) if p.h2_inpatient_service_times else 0 for p in my_model.patients])
               #  mean_service_h3_inpatient = np.mean([np.mean(p.h3_inpatient_service_times) if p.h3_inpatient_service_times else 0 for p in my_model.patients])

                # Mean total time in system per patient
                total_times = [p.total_time for p in my_model.patients if p.total_time is not None]
                mean_total_time = np.mean(total_times) if total_times else 0    

                # Save results in the DataFrame
                self.df_trial_results.loc[run] = {
                    "Mean Queue Time ED": mean_wait_ED,
                    "Mean Service Time ED": mean_service_ED,
                    "Mean Queue Time ICU": mean_wait_ICU,
                    "Mean Service Time ICU": mean_service_ICU,
                    "Mean Queue Time Inpatient": mean_wait_inpatient,
                    "Mean Service Time Inpatient": mean_service_inpatient,
                    # "Mean Queue Time H2 Inpatient": mean_wait_h2_inpatient,
                    # "Mean Service Time H2 Inpatient": mean_service_h2_inpatient,
                    # "Mean Queue Time H3 Inpatient": mean_wait_h3_inpatient,
                    # "Mean Service Time H3 Inpatient": mean_service_h3_inpatient,
                    "Mean Total Time": mean_total_time
                }

                # Store event logger
                self.all_event_logs.append(my_model.logger)

            # Combine all logs into one large DataFrame
            self.all_event_logs_df = pd.concat([logger.to_dataframe() for logger in self.all_event_logs])
            self.all_event_logs_df.sort_values(by=["time", "entity_id"], inplace=True)
            self.all_event_logs_df.reset_index(drop=True, inplace=True)
    return (Trial,)


@app.cell
def _(Trial, acuity_flags, collapsed_library):
    my_trial = Trial(collapsed_library, acuity_flags)

    my_trial.run_trial()
    return (my_trial,)


@app.cell
def _(my_trial):
    my_trial.all_event_logs_df
    return


@app.cell
def _(EventPosition, create_event_position_df):
    event_position_df = create_event_position_df([
        # Arrival
        EventPosition(event='arrival', x=50, y=450, label="Arrival"),

        # ED
        EventPosition(event='ED_wait_begins', x=200, y=375, label="Waiting for ED"),
        EventPosition(event='ED_begins', x=100, y=200, label="Being Treated in ED", resource='ed_bed_cap'),

        # ICU
        EventPosition(event='ICU_wait_begins', x=400, y=200, label="Waiting for ICU"),
        EventPosition(event='ICU_begins', x=400, y=100, label="Being Treated in ICU", resource='icu_bed_cap'),

        # Inpatient
        EventPosition(event='INPATIENT_wait_begins', x=600, y=400, label="Waiting for Inpatient"),
        EventPosition(event='INPATIENT_begins', x=600, y=300, label="Being Treated in Inpatient", resource='inpatient_bed_cap'),

        # Exit / Discharge
        EventPosition(event='depart', x=800, y=100, label="Discharge")
    ])
    return (event_position_df,)


@app.cell
def _(animate_activity_log, event_position_df, my_trial):

    # Filter our dataframe down to a single run
    single_run_event_log_df = my_trial.all_event_logs_df[my_trial.all_event_logs_df['run_number']==1]

    animate_activity_log(
            # Pass in our filtered event log
            event_log=single_run_event_log_df,
            # Pass in our event position dataframe
            event_position_df= event_position_df,
            # Use an instance of the g class as our scenario so that it can access the required
            # information about how many resources are available
            scenario=g(),
            # How long should the animation last? We can pass in any value here - but I've chosen to
            # make it last as long as our originally defined simulation duration
            limit_duration=g.sim_duration,
            # Turn on logging messages
            debug_mode=True,
            # Turn on axis units - this can help with honing your event_position_df iteratively
            setup_mode=True,
            # How big should the time steps be? Here,
            every_x_time_units= 1,
            # Should the animation allow you to just drag a slider to progress through the animation,
            # or should it include a play button?
            include_play_button=True,
            # How big should the icons representing our entities be?
            entity_icon_size=20,
            # How big should the icons representing our resources be?
            resource_icon_size=20,
            # How big should the gap between our entities be when they are queueing?
            gap_between_entities=6,
            gap_between_resources=10,
            # When we wrap the entities to fit more neatly on the screen, how big should the vertical
            # gap be between these rows?
            gap_between_queue_rows=25,
            # How tall, in pixels, should the plotly plot be?
            plotly_height=800,
            # How wide, in pixels, should the plotly plot be?
            plotly_width=1000,
            # How long, in milliseconds, should each frame last?
            frame_duration= 600,
            # How long, in milliseconds, should the transition between each pair of frames be?
            frame_transition_duration=600,
            # How wide, in coordinates, should our plot's internal coordinate system be?
            # override_x_max=1000,
            # How tall, in coordinates, should our plot's internal coordinate system be?
            # override_y_max=500,
            # How long should a queue be before it starts wrapping vertically?
            wrap_queues_at=25,
            # What are the maximum numbers of entities that should be displayed in any queueing steps
            # before displaying additional entities as a text string like '+ 37 more'
            step_snapshot_max=125,
            # What should the time display units be underneath the simulation?
            time_display_units="simulation_day_clock_ampm",
            simulation_time_unit='hours',
            # display our Label column from our event_position_df to identify the position of each icon
            display_stage_labels=True
        )
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


    ed_usage = compute_usage(single_run_event_log_df, "ED")
    icu_usage = compute_usage(single_run_event_log_df, "ICU")
    inp_usage = compute_usage(single_run_event_log_df, "INPATIENT")

    plt.figure(figsize=(10, 6))

    plt.plot(ed_usage["time"], ed_usage["in_use"], label="ED")
    plt.plot(icu_usage["time"], icu_usage["in_use"], label="ICU")
    plt.plot(inp_usage["time"], inp_usage["in_use"], label="Inpatient")

    plt.xlabel("Time")
    plt.ylabel("Beds in use")
    plt.title("Resource Usage Over Time")
    plt.legend()
    plt.show()
    return


@app.cell
def _(single_run_event_log_df):
    def compute_max_beds(df, resource):
        """
        Compute the maximum concurrent patients for a given resource.
    
        df: event log dataframe with columns 'time' and 'event'
        resource: string, e.g., 'ED', 'ICU', 'INPATIENT'
        """
        # Filter for begins/ends events for this resource
        events = df[(df["event"].str.contains(f"{resource}_begins")) |
                    (df["event"].str.contains(f"{resource}_ends"))].copy()
    
        # Sort by time
        events = events.sort_values("time")
    
        in_use = 0
        max_in_use = 0
        for _, row in events.iterrows():
            if row["event"].endswith("begins"):
                in_use += 1
            else:
                in_use -= 1
            if in_use > max_in_use:
                max_in_use = in_use
    
        return max_in_use

    # Example usage
    ed_max = compute_max_beds(single_run_event_log_df, "ED")
    icu_max = compute_max_beds(single_run_event_log_df, "ICU")
    inp_max = compute_max_beds(single_run_event_log_df, "INPATIENT")

    print(f"Realistic ED bed capacity: {ed_max}")
    print(f"Realistic ICU bed capacity: {icu_max}")
    print(f"Realistic Inpatient bed capacity: {inp_max}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
