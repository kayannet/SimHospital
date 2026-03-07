# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.20.4",
#     "pandas==2.3.3",
#     "pillow==12.1.1",
#     "simpy==4.1.1",
#     "vidigi==1.2.2",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from vidigi.resources import VidigiStore
    from vidigi.logging import EventLogger
    from vidigi.utils import EventPosition, create_event_position_df
    from vidigi.animation import animate_activity_log
    import simpy
    mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 10_000_000_000
    return (
        EventPosition,
        VidigiStore,
        animate_activity_log,
        create_event_position_df,
        pd,
        simpy,
    )


@app.cell
def _():
    return


@app.cell
def _(pd):
    simulation_log = pd.read_csv('../../results/simulation_logs/LJ_combined_df_all_thresholds+baseline.csv')
    return (simulation_log,)


@app.cell
def _(simulation_log):
    simulation_log
    return


@app.cell
def _(simpy):
    class g:
        """
        Global model configuration
        """


        # -------------------------
        # Capacities (by state)
        # -------------------------
        # calculated by taking the max utilized when cap was set to 9999
        env = simpy.Environment()

        UNITS = ["ED", "FLOOR", "ICU", "BOARDED"]
        HOSPITALS = ["EAST_CAMPUS", "HILLCREST", "LA_JOLLA"]

        STATES = [
                    "ED_EAST_CAMPUS",
                    "ED_HILLCREST",
                    "ED_LA_JOLLA",
                    "FLOOR_EAST_CAMPUS",
                    "FLOOR_HILLCREST",
                    "FLOOR_LA_JOLLA",
                    "ICU_EAST_CAMPUS",
                    "ICU_HILLCREST",
                    "ICU_LA_JOLLA",
                    "BOARDED_EAST_CAMPUS",
                    "BOARDED_HILLCREST",
                    "BOARDED_LA_JOLLA",
        ]
        # RESOURCE CAPS
        ED_EAST_CAMPUS = 12
        ED_HILLCREST = 31
        ED_LA_JOLLA = 67

        # FLOOR
        FLOOR_EAST_CAMPUS = 82
        FLOOR_HILLCREST = 210
        FLOOR_LA_JOLLA = 340

        # ICU
        ICU_EAST_CAMPUS = 8
        ICU_HILLCREST = 30
        ICU_LA_JOLLA = 57

        # BOARDED
        BOARDED_EAST_CAMPUS = 4
        BOARDED_HILLCREST = 10
        BOARDED_LA_JOLLA = 49

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
        number_of_runs = 6
        random_number_set = 42
        audit_interval = 1

        # for predictor:
        q = 0.75
        threshold = [3, 8, 12, 24, 48, 72] # 45



    return (g,)


@app.cell
def _(VidigiStore, g):
    class Hospital:
        def __init__(self, env, campus_name, capacities, arrival_rate):
            self.env = env
            self.campus_name = campus_name
            self.capacities = {}
            self.resources = {}
            self.arrival_rate =arrival_rate

            self._init_resources(capacities)
            self.median_los_per_resource = {}
            self.best_params_per_resource = g.fitted_dist_per_state[self.campus_name]

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

    return


@app.cell
def _():
    # hospital_classes = {}
    # for hospital in g.HOSPITALS:
    #     hospital_classes[hospital] = Hospital(
    #                         env= g.env,
    #                         campus_name=hospital,
    #                         capacities=g.capacities, 
    #                         arrival_rate = g.arrival_rates[hospital], 
    #                     )


    # scenario = g()  # create your config object

    # # Attach resources from all hospitals
    # for campus_name, hospital in hospital_classes.items():
    #     for state_name, resource in hospital.resources.items():
    #         setattr(scenario, state_name, hospital.capacities[state_name])
    return


@app.cell
def _(EventPosition, create_event_position_df):
    event_position_df = create_event_position_df([

        # ARRIVAL
        EventPosition(event='arrival', x=50, y=670, label="Arrival"),

        # LA JOLLA
        EventPosition(event='ED_LA_JOLLA_wait_begins', x=230, y=600, label="Waiting for ED (La Jolla)"),
        EventPosition(event='ED_LA_JOLLA_begins', x=230, y=490, label="Being Treated in ED (La Jolla)", resource='ED_LA_JOLLA'),

        EventPosition(event='BOARD_LA_JOLLA_wait_begins', x=230, y=410, label="Waiting for Boarding (La Jolla)"),
        EventPosition(event='BOARD_LA_JOLLA_begins', x=230, y=350, label="Boarded in ED (La Jolla)", resource='BOARDED_LA_JOLLA'),

        EventPosition(event='ICU_LA_JOLLA_wait_begins', x=230, y=280, label="Waiting for ICU (La Jolla)"),
        EventPosition(event='ICU_LA_JOLLA_begins', x=230, y=210, label="Being Treated in ICU (La Jolla)", resource='ICU_LA_JOLLA'),

        EventPosition(event='FLOOR_LA_JOLLA_wait_begins', x=230, y=150, label="Waiting for Floor (La Jolla)"),
        EventPosition(event='FLOOR_LA_JOLLA_begins', x=230, y=50, label="Being Treated on Floor (La Jolla)", resource='FLOOR_LA_JOLLA'),



        # HILLCREST
        EventPosition(event='ED_HILLCREST_wait_begins', x=495, y=600, label="Waiting for ED (Hillcrest)"),
        EventPosition(event='ED_HILLCREST_begins', x=495, y=490, label="Being Treated in ED (Hillcrest)", resource='ED_HILLCREST'),

        EventPosition(event='BOARD_HILLCREST_wait_begins', x=495, y=410, label="Waiting for Boarding (Hillcrest)"),
        EventPosition(event='BOARD_HILLCREST_begins', x=495, y=350, label="Boarded in ED (Hillcrest)", resource='BOARDED_HILLCREST'),

        EventPosition(event='ICU_HILLCREST_wait_begins', x=495, y=280, label="Waiting for ICU (Hillcrest)"),
        EventPosition(event='ICU_HILLCREST_begins', x=495, y=210, label="Being Treated in ICU (Hillcrest)", resource='ICU_HILLCREST'),

        EventPosition(event='FLOOR_HILLCREST_wait_begins', x=495, y=150, label="Waiting for Floor (Hillcrest)"),
        EventPosition(event='FLOOR_HILLCREST_begins', x=495, y=50, label="Being Treated on Floor (Hillcrest)", resource='FLOOR_HILLCREST'),


         # EAST CAMPUS
        EventPosition(event='ED_EAST_CAMPUS_wait_begins', x=740, y=600, label="Waiting for ED (East)"),
        EventPosition(event='ED_EAST_CAMPUS_begins', x=740, y=490, label="Being Treated in ED (East)", resource='ED_EAST_CAMPUS'),

        EventPosition(event='BOARD_EAST_CAMPUS_wait_begins', x=740, y=410, label="Waiting for Boarding (East)"),
        EventPosition(event='BOARD_EAST_CAMPUS_begins', x=740, y=350, label="Boarded in ED (East)", resource='BOARDED_EAST_CAMPUS'),

        EventPosition(event='ICU_EAST_CAMPUS_wait_begins', x=740, y=280, label="Waiting for ICU (East)"),
        EventPosition(event='ICU_EAST_CAMPUS_begins', x=740, y=210, label="Being Treated in ICU (East)", resource='ICU_EAST_CAMPUS'),

        EventPosition(event='FLOOR_EAST_CAMPUS_wait_begins', x=740, y=150, label="Waiting for Floor (East)"),
        EventPosition(event='FLOOR_EAST_CAMPUS_begins', x=740, y=50, label="Being Treated on Floor (East)", resource='FLOOR_EAST_CAMPUS'),

        # DEPART
        EventPosition(event='depart', x=350, y=10, label="Discharge")
    ])
    return (event_position_df,)


@app.cell
def _():
    return


@app.cell
def _(simulation_log):
    single_run_event_log_df = simulation_log[simulation_log['run_number'] == 5]
    single_run_event_log_df
    return (single_run_event_log_df,)


@app.cell
def _():
    from pathlib import Path
    from PIL import Image

    BASE_DIR = Path(__file__).resolve().parent
    bg_path = BASE_DIR / "../../assets/sim_v2_bg_img.png"

    bg_img = Image.open(bg_path)
    return (bg_img,)


@app.cell
def _(
    animate_activity_log,
    bg_img,
    event_position_df,
    g,
    single_run_event_log_df,
):
    animate_activity_log(
            # Pass in our filtered event log
            event_log=single_run_event_log_df,
            # Pass in our event position dataframe
            event_position_df= event_position_df,
            # Use an instance of the g class as our scenario so that it can access the required
            # information about how many resources are available
            scenario = g(),
            # How long should the animation last? We can pass in any value here - but I've chosen to
            # make it last as long as our originally defined simulation duration
            limit_duration= 100,#g.sim_duration,
            # Turn on logging messages
            debug_mode=True,
            # Turn on axis units - this can help with honing your event_position_df iteratively
            setup_mode=False,
            # How big should the time steps be? Here,
            every_x_time_units= 1,
            # Should the animation allow you to just drag a slider to progress through the animation,
            # or should it include a play button?
            include_play_button=True,
            # How big should the icons representing our entities be?
            entity_icon_size=10,
            # How big should the icons representing our resources be?
            resource_icon_size=10,
            text_size = 10,
            # How big should the gap between our entities be when they are queueing?
            gap_between_entities=5,
            gap_between_resources=10,
            # When we wrap the entities to fit more neatly on the screen, how big should the vertical
            # gap be between these rows?
            gap_between_queue_rows= 5,
            gap_between_resource_rows=5,



            # How tall, in pixels, should the plotly plot be?
            plotly_height=900,
            plotly_width=1000,
            override_x_max=800,
            override_y_max=675,
            # How long, in milliseconds, should each frame last?
            frame_duration= 1000,
            # How long, in milliseconds, should the transition between each pair of frames be?
            frame_transition_duration=1000,

            # How long should a queue be before it starts wrapping vertically?
            wrap_queues_at=15,
            wrap_resources_at=15,

            # What are the maximum numbers of entities that should be displayed in any queueing steps
            # before displaying additional entities as a text string like '+ 37 more'
            step_snapshot_max=125,
            # What should the time display units be underneath the simulation?
            time_display_units="simulation_day_clock_ampm",
            simulation_time_unit='hours',
            # display our Label column from our event_position_df to identify the position of each icon
            display_stage_labels=False, 

            add_background_image= bg_img, 
            background_image_opacity=1,
            resource_opacity= 0,
        )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
