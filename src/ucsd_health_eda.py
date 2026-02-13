# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "ibis-framework[duckdb]==12.0.0",
#     "marimo>=0.19.7",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "pandas==2.3.3",
#     "scipy==1.17.0",
#     "seaborn==0.13.2",
#     "simpy==4.1.1",
#     "vidigi==1.2.2",
# ]
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import ibis
    import ibis.selectors as s
    from ibis import _
    import simpy
    import random
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from vidigi.resources import VidigiStore
    from vidigi.logging import EventLogger
    from vidigi.utils import EventPosition, create_event_position_df
    from vidigi.animation import animate_activity_log
    import re
    from scipy import stats

    # Set options
    ibis.options.interactive = True
    pd.set_option("display.max_columns", None)
    return mo, np, pd, re, stats


@app.cell
def _():
    PHI_COLS = ['event_id', 'pat_enc_csn_id', 'pat_id']

    return (PHI_COLS,)


@app.cell
def _(pd, re):
    def to_snake(name: str) -> str:
        name = name.strip()
        name = re.sub(r"[ \-]+", "_", name)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"_+", "_", name)
        return name.lower()

    # Load data
    df = pd.read_csv('JCHI_ADT Extract_Event Dates CY2025.csv')

    # convert columns to all snake case
    df.columns = [to_snake(c) for c in df.columns]

    # Convert time and sort
    df["effective_time"] = pd.to_datetime(df["effective_time"])
    df = df.sort_values(
        by=[
            "pat_enc_csn_id",
            "effective_time",
            "seq_num_in_enc",
            "event_id",
        ]
    )

    # Show result
    df.loc[:, ~df.columns.isin(['event_id', 'pat_enc_csn_id', 'pat_id'])]
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    Goal: to understand all of the columns, their data types, and events in each column.
    """)
    return


@app.cell
def _(df):
    # Display all columns and their data types
    df.dtypes
    return


@app.cell
def _(mo):
    mo.md("""
    # Column Definitions - JCHI ADT Data

    ## Patient Identifiers
    - **pat_id** - Patient medical record number (same person across multiple visits)
    - **pat_enc_csn_id** - Patient encounter ID (one unique hospital visit)

    ## Event Information
    - **event_id** - Unique ID for this event record
    - **event_type** - What happened: Admission, Transfer In, Transfer Out, Discharge, Census, Patient Update
    - **event_type_c** - Numeric code for event_type
    - **effective_time** - When this event occurred
    - **seq_num_in_enc** - Order number of events within this encounter (1, 2, 3...)

    ## Transfer Tracking
    - **linked_transfer_in_event_id** - Links to the "transfer in" event_id
    - **linked_transfer_out_event_id** - Links to the "transfer out" event_id
    - **transfer_type** - Why patient transferred (ED to Inpatient, Trauma, STEMI, etc.)

    ## Location Information
    - **department_id** - Department ID number
    - **department_name** - Name of department/unit (e.g., "LJ EMERGENCY DEPT", "HC 2-SICU")
    - **department_group** - Care level grouping (MedSurg, ICU, PCU, OB, etc.)
    - **location** - Specific location description
    - **room_id** / **room_csn_id** / **room** - Room identifiers and name
    - **bed_id** / **bed_csn_id** / **bed** - Bed identifiers and name

    ## Patient Classification
    - **base_patient_class** - Simplified: Emergency, Inpatient, or Outpatient
    - **base_pat_class_c** - Numeric code for base_patient_class
    - **patient_class** - Detailed class (e.g., "Inpatient Admission", "Emergency Department - Encounter")
    - **pat_class_c** - Numeric code for patient_class
    - **patient_levelof_care** - Level of care needed (text)
    - **pat_lvl_of_care_c** - Level of care code (numeric)

    ## Clinical Service
    - **hospital_service** - Which clinical team (Emergency Medicine, Cardiology, Trauma Surgery, etc.)
    - **pat_service_c** - Numeric code for hospital_service

    ## Bed Request Tracking (often empty)
    - **pend_id** - Pending bed request ID
    - **bed_request_created_dttm** - When bed was requested
    - **bed_request_first_readyto_plan_dttm** - Ready for bed planning
    - **bed_request_first_preassigned_dttm** - Bed pre-assigned
    - **bed_request_first_assigned_dttm** - Bed officially assigned
    - **bed_request_first_bed_ready_dttm** - Bed physically ready
    - **bed_request_first_readyto_move_dttm** - Patient ready to move

    ## Other
    - **tc_admit_flag** - Flag indicating admission event
    - **referring_location** - Where patient was referred from

    ---

    **Total: 37 columns**
    """)
    return


@app.cell
def _(df):
    # Look at first few rows with all columns visible
    df.head(10)
    return


@app.cell
def _(df):
    # Event types, what kinds of events do we have?
    print("Unique Event Types:")
    print(df['event_type'].value_counts())
    print(f"\nTotal unique event types: {df['event_type'].nunique()}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Why are there way more discharges than admissions?

    Looking at event types:
    - Admissions: 229,440
    - Discharges: 983,931

    That's 4-5x more discharges than admissions, which doesn't make sense. You can't discharge more people than you admit.

    What's probably happening:
    - "Discharge" might include ED patients who go home without being admitted
    - Outpatient visits that end also get counted as discharges
    - We need to check if this is true

    Let's look at what patient classes are getting "discharged" vs "admitted" to figure out what's actually going on.
    """)
    return


@app.cell
def _(df):
    # What patient classes have Discharge events?
    discharge_events = df[df['event_type'] == 'Discharge']
    discharge_classes = discharge_events['patient_class'].value_counts()
    print("Discharges by Patient Class:")
    print(f"Total unique classes: {len(discharge_classes)}\n")
    for class_name, class_count in discharge_classes.items():
        print(f"{class_name}: {class_count:,}")
    return


@app.cell
def _(df):
    # What about Admissions?
    admission_events = df[df['event_type'] == 'Admission']
    admission_classes = admission_events['patient_class'].value_counts()
    print("Admissions by Patient Class:")
    print(f"Total unique classes: {len(admission_classes)}\n")
    for admit_class, admit_count in admission_classes.items():
        print(f"{admit_class}: {admit_count:,}")
    return


@app.cell
def _(mo):
    mo.md("""
    Most "Discharges" are outpatients. Looking at the discharge and admission breakdown by patient class, we found that the majority of the 983K discharges are from outpatient visits - people coming in for radiology, infusion therapy, procedures, clinic visits, etc. They come in, get their service, and leave the same day, which gets recorded as a "discharge" even though they were never "admitted" in the traditional sense. This explains why there are way more discharges than admissions overall. However, there are still weird discrepancies even within inpatient and ED encounters - for example, 38K inpatient discharges but only 3K inpatient admissions. This suggests there might be data quality issues like duplicate discharge records, missing admission records, or the event types mean something different than we expect. We need to investigate these discrepancies before we can reliably model patient flow.
    """)
    return


@app.cell
def _(events_per_encounter):
    events_per_encounter
    return


@app.cell
def _(PHI_COLS, df):
    # Checking - how many events per encounter?
    events_per_encounter = df.groupby('pat_enc_csn_id')['event_type'].value_counts().unstack(fill_value=0)
    print("Sample of events per encounter (first 10):")
    print(events_per_encounter.reset_index().loc[:, ~events_per_encounter.reset_index().columns.isin(PHI_COLS)].head(10))
    print("\nSummary stats:")
    print(events_per_encounter.describe())
    return (events_per_encounter,)


@app.cell
def _(df):
    # Checking - do encounters have multiple discharges?
    discharges_per_encounter = df[df['event_type'] == 'Discharge'].groupby('pat_enc_csn_id').size()
    print(f"Encounters with discharges: {len(discharges_per_encounter):,}")
    print(f"Encounters with >1 discharge: {(discharges_per_encounter > 1).sum():,}")
    print(f"\nMax discharges in one encounter: {discharges_per_encounter.max()}")
    return


@app.cell
def _(df):
    # Checking - do encounters have multiple admissions?
    admissions_per_encounter = df[df['event_type'] == 'Admission'].groupby('pat_enc_csn_id').size()
    print(f"Encounters with admissions: {len(admissions_per_encounter):,}")
    print(f"Encounters with >1 admission: {(admissions_per_encounter > 1).sum():,}")
    print(f"\nMax admissions in one encounter: {admissions_per_encounter.max()}")
    return


@app.cell
def _(mo):
    mo.md("""
    No duplicate events, but not all encounters have admissions. We discovered that each encounter has exactly 1 discharge, no duplicates, clean data. Each encounter has exactly 1 admission (when it has one) - no duplicates, clean data. BUT only 23% of encounters have an "Admission" event - this is the key!

    Looking at the summary stats:

    - Mean discharges per encounter: 0.999 (almost everyone)
    - Mean admissions per encounter: 0.233 (only 23%!)
    - This explains why 983K discharges vs 229K admissions

    What does this mean? Basically, outpatient visits get a "Discharge" when they leave, but never had an "Admission". ED visits that go home "Discharge" from ED without hospital "Admission". True inpatient stays should have both "Admission" AND "Discharge".

    Some wild outliers:

    - One encounter has 364 Census events (daily snapshots over a year)
    - One encounter has 68 transfers!

    The issue is that "Admission" and "Discharge" as event types mean different things depending on the encounter type.

    For our patient flow model, we don't need to worry about these administrative labels. Instead, we need to focus on the Transfer events which show actual physical movement between units. Let's look at what transfer patterns exist in the data.
    """)
    return


@app.cell
def _(df):
    # Patient classes, what types of patients?
    print("Unique Patient Classes:")
    all_patient_classes = df['patient_class'].value_counts()
    print(f"Total unique patient classes: {len(all_patient_classes)}")
    for patient_class, count in all_patient_classes.items():
        print(f"{patient_class}: {count:,}")
    return


@app.cell
def _(df):
    # Department names, where are patients located?
    all_departments = df['department_name'].value_counts()
    print(f"Total unique departments: {len(all_departments)}")
    for dept_name, dept_count in all_departments.items():
        print(f"{dept_name}: {dept_count:,}")
    return


@app.cell
def _(df):
    # Department groups, what are the higher level groupings?
    print("Unique Department Groupings:")
    print(df['department_group'].value_counts())
    print(f"\nTotal unique department groups: {df['department_group'].nunique()}")
    return


@app.cell
def _(df):
    # Hospital services, what service lines?
    all_services = df['hospital_service'].value_counts()
    print(f"Total unique hospital services: {len(all_services)}")
    for service_name, service_count in all_services.items():
        print(f"{service_name}: {service_count:,}")
    return


@app.cell
def _(df):
    # Transfer types
    all_transfer_types = df['transfer_type'].value_counts()
    print(f"Total unique transfer types: {len(all_transfer_types)}")
    for transfer_name, transfer_count in all_transfer_types.items():
        print(f"{transfer_name}: {transfer_count:,}")
    return


@app.cell
def _(df):
    # Look at transfers between department_groups (the tiers)
    transfer_data = df[df['event_type'].isin(['Transfer In', 'Transfer Out'])]

    # What tiers do patients transfer between?
    print("Transfers by department group:")
    print(transfer_data['department_group'].value_counts())
    return


@app.cell
def _(df):
    # Base patient class
    print("Unique Base Patient Classes:")
    print(df['base_patient_class'].value_counts())
    print(f"\nTotal unique base patient classes: {df['base_patient_class'].nunique()}")
    return


@app.cell
def _(df):
    # Do patients change class during an encounter?
    class_changes = df.groupby('pat_enc_csn_id')['patient_class'].nunique()
    print(f"Encounters with class changes: {(class_changes > 1).sum():,}")
    print(f"Total encounters: {df['pat_enc_csn_id'].nunique():,}")
    return


@app.cell
def _(mo):
    mo.md("""
    Looking at the transfer data, we can see clear patterns of patient movement:

    **Transfer Types:**
    - Inpatient to Inpatient: 26,139 (most common - patients moving between units)
    - ED to Inpatient: 16,344 (patients getting admitted from ED)
    - ED to ED: 14,184 (transfers between emergency departments)
    - Trauma: 11,336 (trauma activations)
    - Specialty transfers: ECMO, Stroke, STEMI, Transplant, Burn (time-sensitive care)

    **Transfer Activity by Tier:**
    - OB: 50,241 transfers (obstetrics - lots of movement)
    - MedSurg: 34,207 (general medical/surgical floors - Tier 2)
    - PCU: 27,195 (progressive care - Tier 2)
    - ICU: 24,187 (intensive care - Tier 3)
    - IMU: 16,756 (intermediate medical - Tier 2)
    - NICU: 3,658 (neonatal intensive care - Tier 3)

    We have good transfer data across all three tiers. The transfer events will let us build a transition matrix showing how patients move between ED, regular floors, and ICU across the different hospitals.

    Next step is to filter the data to focus on encounters with actual patient flow (those with transfers), then build the transition matrix.
    """)
    return


@app.cell
def _(df):
    # How many events does each encounter have?
    encounter_event_counts = df.groupby('pat_enc_csn_id').size()

    print("Events per encounter:")
    print(f"Min: {encounter_event_counts.min()}")
    print(f"Max: {encounter_event_counts.max()}")
    print(f"Mean: {encounter_event_counts.mean():.2f}")
    print(f"Median: {encounter_event_counts.median():.0f}")
    print(f"\nEncounters with only 1 event: {(encounter_event_counts == 1).sum():,}")
    print(f"Percentage: {(encounter_event_counts == 1).sum() / len(encounter_event_counts) * 100:.1f}%")
    return


@app.cell
def _(df):
    # Check encounters with NO transfers (arrive and leave only)
    encounters_with_transfers = df[df['event_type'].isin(['Transfer In', 'Transfer Out'])]['pat_enc_csn_id'].unique()
    total_encounters = df['pat_enc_csn_id'].nunique()
    no_transfer_encounters = total_encounters - len(encounters_with_transfers)

    print(f"Total encounters: {total_encounters:,}")
    print(f"Encounters WITH transfers: {len(encounters_with_transfers):,}")
    print(f"Encounters WITHOUT transfers: {no_transfer_encounters:,}")
    print(f"Percentage without transfers: {no_transfer_encounters / total_encounters * 100:.1f}%")
    return


@app.cell
def _(df, events_per_encounter):
    # What are these single-event encounters?
    single_event_encounters = df[df['pat_enc_csn_id'].isin(events_per_encounter[events_per_encounter == 1].index)]
    print("Single event encounters by event type:")
    print(single_event_encounters['event_type'].value_counts())
    print("\nSingle event encounters by patient class:")
    print(single_event_encounters['patient_class'].value_counts().head(10))
    return


@app.cell
def _(df):
    # Check linked Transfer Out → Transfer In pairs
    # Use linked_transfer_out_event_id to connect them

    # Get Transfer Out events with their department
    transfer_out = df[df['event_type'] == 'Transfer Out'][['event_id', 'pat_enc_csn_id', 'department_name', 'department_group', 'effective_time']]
    transfer_out = transfer_out.rename(columns={'event_id': 'transfer_out_id', 'department_name': 'from_dept', 'department_group': 'from_group', 'effective_time': 'out_time'})

    # Get Transfer In events with their department
    transfer_in = df[df['event_type'] == 'Transfer In'][['event_id', 'pat_enc_csn_id', 'department_name', 'department_group', 'effective_time', 'linked_transfer_out_event_id']]
    transfer_in = transfer_in.rename(columns={'department_name': 'to_dept', 'department_group': 'to_group', 'effective_time': 'in_time'})

    # Merge them
    transfers = transfer_in.merge(transfer_out, left_on='linked_transfer_out_event_id', right_on='transfer_out_id', suffixes=('', '_out'))

    print(f"Total linked transfers: {len(transfers):,}")
    print(f"\nTransfers within SAME department: {(transfers['from_dept'] == transfers['to_dept']).sum():,}")
    print(f"Transfers to DIFFERENT department: {(transfers['from_dept'] != transfers['to_dept']).sum():,}")
    return


@app.cell
def _(mo):
    mo.md("""
    Issue 1: Most encounters have only 1 event (76.6%)
    - These are mostly outpatient visits (radiology, infusion, procedures)
    - Also ED visits that discharge home
    - They have 1 event and leave - no patient flow to model, at least for right now while we build the simple model.

    Issue 2: 91.5% of encounters have NO transfers
    - Only 83,423 encounters (8.5%) actually have transfer events between units
    - BUT this doesn't mean the other 91.5% are irrelevant!
    - Many are still valid patient flow: ED → Discharge, ICU → Discharge (these are important transitions)
    - The real question is what TYPE of encounters to include (ED/ICU/Floor vs Outpatient), not whether they have transfers
    - We'll decide which encounter types to keep after looking at the data more carefully

    Issue 3: Same Department Bed Changes
    - Out of 207,300 total transfers:
      - 82,058 (40%) are within the SAME department - just bed changes!
      - 125,242 (60%) are to DIFFERENT departments - actual patient flow!
      - We need to filter these out.

    What This Means for Data Cleaning:

    1. Filter by encounter TYPE (keep ED/ICU/Floor, remove outpatient), not by whether they have transfers
    2. Only use transfers to DIFFERENT departments (125,242 transfers)
    3. Ignore same-department bed shuffles (82,058 transfers)

    This will give us clean patient flow data for the transition matrix.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    What Encounters Do We Have?

    Before filtering, let's see all the encounter types and decide which ones are relevant for ED → Floor → ICU flow modeling.
    """)
    return


@app.cell
def _(df):
    # Look at combinations of patient_class and department_group
    encounter_summary = df.groupby(['patient_class', 'department_group']).size().reset_index(name='count')
    encounter_summary = encounter_summary.sort_values('count', ascending=False)

    print("Patient Class + Department Group combinations:")
    print(encounter_summary.to_string(index=False))
    return


@app.cell
def _(df):
    # Also - which patient classes appear in ED, ICU, MedSurg?
    ed_classes = df[df['department_name'].str.contains('EMERGENCY', na=False)]['patient_class'].value_counts()
    print("\nPatient classes in ED:")
    print(ed_classes)
    return


@app.cell
def _(df):
    icu_classes = df[df['department_group'] == 'ICU']['patient_class'].value_counts()
    print("\nPatient classes in ICU:")
    print(icu_classes)
    return


@app.cell
def _(df):
    floor_classes = df[df['department_group'].isin(['MedSurg', 'PCU', 'IMU'])]['patient_class'].value_counts()
    print("\nPatient classes on Floors (MedSurg/PCU/IMU):")
    print(floor_classes)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Data Filtering Plan: What We Are Keeping vs Removing and Why

    Keeping patients who occupy beds and can move between units:
    - Inpatient Admission
    - Emergency Department - Encounter
    - Trauma Inpatient Admission
    - Observation
    - Trauma Observation Admission
    - Hospital Inpatient Surgery
    - Inpatient Psychiatry
    - Newborn
    - Surgical OB Patient
    - Outpatient in a Bed (they're in ICU/MedSurg/PCU - physical beds!)
    - Outpatient Labor and Delivery - Encounter (in OB units)

    Removing same-day outpatient services (no bed, no movement):
    - Outpatient Radiology - Encounter
    - Outpatient Infusion - Encounter
    - Outpatient Radiation Oncology - Series
    - Hospital Outpatient Procedure
    - Hospital Outpatient Surgery
    - Outpatient Clinic Visit - Encounter
    - Outpatient Physical Therapy - Series
    - Outpatient Other - Encounter
    - Lab Referred Specimen
    - Telemedicine - Encounter
    - All other "Outpatient [service] - Series/Encounter"
    - Trauma Outpatient (unless in a bed)

    This will give us clean patient flow data for the 9×10 transition matrix.
    """)
    return


@app.cell
def _(df):
    # Filter by patient_class to identify which encounters involve physical bed occupancy in hospital units
    # Does this patient_class represent someone who occupies a physical bed in a hospital unit (ED/Floor/ICU) and could potentially transfer between units?
    keep_classes = [
        'Inpatient Admission',
        'Emergency Department - Encounter',
        'Trauma Inpatient Admission',
        'Observation',
        'Trauma Observation Admission',
        'Hospital Inpatient Surgery',
        'Inpatient Psychiatry',
        'Newborn',
        'Surgical OB Patient',
        'Outpatient in a Bed',
        'Outpatient Labor and Delivery - Encounter'
    ]

    # Filter the data
    df_filtered = df[df['patient_class'].isin(keep_classes)].copy()

    print(f"Original data: {len(df):,} rows, {df['pat_enc_csn_id'].nunique():,} encounters")
    print(f"Filtered data: {len(df_filtered):,} rows, {df_filtered['pat_enc_csn_id'].nunique():,} encounters")
    print(f"Removed: {len(df) - len(df_filtered):,} rows ({(len(df) - len(df_filtered))/len(df)*100:.1f}%)")
    return df_filtered, keep_classes


@app.cell
def _(df, keep_classes):
    # What patient classes did we REMOVE?
    data_removed = df[~df['patient_class'].isin(keep_classes)]
    classes_removed = data_removed['patient_class'].value_counts()
    print("Removed Patient Classes:")
    print(f"Total classes removed: {len(classes_removed)}\n")
    for removed_class_name, removed_class_count in classes_removed.items():
        print(f"{removed_class_name}: {removed_class_count:,}")
    return


@app.cell
def _(df_filtered):
    # What patient classes did we KEEP?
    print("\nKept Patient Classes:")
    print(df_filtered['patient_class'].value_counts())
    return


@app.cell
def _(df_filtered):
    # Check: Do filtered encounters have transfers?
    filtered_encounters_with_transfers = df_filtered[df_filtered['event_type'].isin(['Transfer In', 'Transfer Out'])]['pat_enc_csn_id'].nunique()
    print(f"\nFiltered encounters: {df_filtered['pat_enc_csn_id'].nunique():,}")
    print(f"Filtered encounters with transfers: {filtered_encounters_with_transfers:,}")
    print(f"Percentage with transfers: {filtered_encounters_with_transfers / df_filtered['pat_enc_csn_id'].nunique() * 100:.1f}%")
    return


@app.cell
def _(df_filtered):
    # What department_groups remain?
    print("\nDepartment Groups in Filtered Data:")
    print(df_filtered['department_group'].value_counts())
    return


@app.cell
def _(df_filtered):
    # How many filtered encounters have only 1 event?
    filtered_events_per_encounter = df_filtered.groupby('pat_enc_csn_id').size()

    print("Events per encounter in filtered data:")
    print(f"Min: {filtered_events_per_encounter.min()}")
    print(f"Mean: {filtered_events_per_encounter.mean():.2f}")
    print(f"Median: {filtered_events_per_encounter.median():.0f}")
    print(f"\nEncounters with only 1 event: {(filtered_events_per_encounter == 1).sum():,}")
    print(f"Percentage: {(filtered_events_per_encounter == 1).sum() / len(filtered_events_per_encounter) * 100:.1f}%")
    return (filtered_events_per_encounter,)


@app.cell
def _(df_filtered, filtered_events_per_encounter):
    # What ARE these single-event filtered encounters?
    single_event_filtered = df_filtered[df_filtered['pat_enc_csn_id'].isin(
        filtered_events_per_encounter[filtered_events_per_encounter == 1].index
    )]

    print("\nSingle Event Encounters (after filtering):")
    print("By event type:")
    print(single_event_filtered['event_type'].value_counts())
    print("\nBy patient class:")
    print(single_event_filtered['patient_class'].value_counts().head())
    print("\nBy department group:")
    print(single_event_filtered['department_group'].value_counts().head())
    return


@app.cell
def _(mo):
    mo.md(r"""
    These data quality checks looks good, the filter worked as intended. Now moving on to other data quality checks.
    """)
    return


@app.cell
def _(df_filtered):
    # Missing critical fields
    print("Missing Values in Key Fields:")
    print(f"Missing effective_time: {df_filtered['effective_time'].isna().sum():,}")
    print(f"Missing department_name: {df_filtered['department_name'].isna().sum():,}")
    print(f"Missing department_group: {df_filtered['department_group'].isna().sum():,}")
    print(f"Missing event_type: {df_filtered['event_type'].isna().sum():,}")
    return


@app.cell
def _(df_filtered):
    # Do all encounters have a discharge event?
    encounters_with_discharge = df_filtered[df_filtered['event_type'] == 'Discharge']['pat_enc_csn_id'].nunique()
    total_filtered_encounters = df_filtered['pat_enc_csn_id'].nunique()

    print(f"\nDischarge Events:")
    print(f"Total encounters: {total_filtered_encounters:,}")
    print(f"Encounters with Discharge: {encounters_with_discharge:,}")
    print(f"Encounters without Discharge: {total_filtered_encounters - encounters_with_discharge:,}")
    print(f"Percentage missing discharge: {(total_filtered_encounters - encounters_with_discharge) / total_filtered_encounters * 100:.1f}%")
    return


@app.cell
def _(df_filtered):
    # Calculate rough LOS per encounter
    encounter_los = df_filtered.groupby('pat_enc_csn_id')['effective_time'].agg(['min', 'max'])
    encounter_los['los_days'] = (encounter_los['max'] - encounter_los['min']).dt.total_seconds() / 86400

    print(f"\nLength of Stay Distribution:")
    print(encounter_los['los_days'].describe())
    print(f"Encounters with LOS > 180 days: {(encounter_los['los_days'] > 180).sum():,}")
    return (encounter_los,)


@app.cell
def _(df_filtered, encounter_los):
    # How many events do the LOS=0 encounters have?
    zero_los_encs = encounter_los[encounter_los['los_days'] == 0].index
    zero_los_events = df_filtered[df_filtered['pat_enc_csn_id'].isin(zero_los_encs)]

    print(f"Encounters with LOS=0: {len(zero_los_encs):,}")
    print(f"\nEvents per encounter for LOS=0:")
    print(zero_los_events.groupby('pat_enc_csn_id').size().value_counts().sort_index())

    print(f"\nEvent types for LOS=0:")
    print(zero_los_events['event_type'].value_counts())

    print(f"\nPatient class for LOS=0:")
    print(zero_los_events['patient_class'].value_counts().head(10))
    return (zero_los_encs,)


@app.cell
def _(df, zero_los_encs):
    # Seeing what patient classes were in these encounters before filtering
    orphan_in_original = df[df['pat_enc_csn_id'].isin(zero_los_encs)]
    print(orphan_in_original.groupby('pat_enc_csn_id').size().describe())
    print(orphan_in_original['patient_class'].value_counts().head(10))
    return


@app.cell
def _(mo):
    mo.md(r"""
    1,720 encounters (1.1% of 153,878) have only a single event in df_filtered, resulting in LOS = 0.
    These are not filtering artifacts — they had only ~1 event in the original data too (mean 1.08).
    They are incomplete records (orphan discharges, registration-only entries) and contribute no
    transitions. Dropping them to keep only encounters with 2+ events.
    """)
    return


@app.cell
def _(df_filtered):
    # Drop single-event encounters
    event_counts = df_filtered.groupby('pat_enc_csn_id').size()
    multi_event_encs = event_counts[event_counts > 1].index
    df_clean = df_filtered[df_filtered['pat_enc_csn_id'].isin(multi_event_encs)]
    print(f"Remaining encounters: {df_clean['pat_enc_csn_id'].nunique():,}")
    print(f"Remaining events: {len(df_clean):,}")
    return (df_clean,)


@app.cell
def _(df_filtered):
    # Events out of order?
    # Check if seq_num_in_enc matches chronological order
    sample_encounters = df_filtered.groupby('pat_enc_csn_id').filter(lambda x: len(x) > 1).groupby('pat_enc_csn_id').head(20)
    sample = sample_encounters.groupby('pat_enc_csn_id').apply(
        lambda x: (x.sort_values('effective_time')['seq_num_in_enc'].diff() < 0).any()
    )
    print(f"\nEvent Ordering:")
    print(f"Encounters checked: {len(sample)}")
    print(f"Encounters with seq_num out of order: {sample.sum()}")
    return


@app.cell
def _(df, df_filtered):
    # Do encounters have mutliple patient classes?
    encounters_with_multi_classes = df.groupby('pat_enc_csn_id')['patient_class'].nunique()
    print(f"Encounters with multiple patient classes (BEFORE filter): {(encounters_with_multi_classes > 1).sum():,}")

    # After filtering, do they still have multiple?
    filtered_multi_classes = df_filtered.groupby('pat_enc_csn_id')['patient_class'].nunique()
    print(f"Encounters with multiple patient classes (AFTER filter): {(filtered_multi_classes > 1).sum():,}")
    return


@app.cell
def _(df_filtered):
    # Example of an encounter that has multiple patient classes
    sample_multi_class = df_filtered.groupby('pat_enc_csn_id').filter(lambda x: x['patient_class'].nunique() > 1).groupby('pat_enc_csn_id').first().index[0]

    print(f"Example Encounter with Multiple Classes: {sample_multi_class}:")
    sample_events = df_filtered[df_filtered['pat_enc_csn_id'] == sample_multi_class][
        ['event_type', 'effective_time', 'patient_class', 'department_name', 'department_group', 'seq_num_in_enc']
    ].sort_values('effective_time')

    print(f"\nTotal events in this encounter: {len(sample_events)}")
    print("\n")
    for idx, row in sample_events.iterrows():
        print(f"Event: {row['event_type']:15} | Time: {row['effective_time']} | Class: {row['patient_class']:30} | Dept: {row['department_name']:30} | Group: {row['department_group']}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Looking at encounters with multiple patient classes, we found that it's completely normal for patient_class to change during a hospital stay. For example, a patient might start as "Hospital Inpatient Surgery" in a perioperative area, then get updated to "Inpatient Admission" once they transfer to an actual inpatient unit. This explains why we have 40,731 encounters with multiple patient classes after filtering. The 556,863 missing department_group values are from events in perioperative areas, holding areas, recovery rooms, and other pre-admission locations that don't fit into the ED/Floor/ICU tier system. These aren't data errors - they're just areas we don't need to model. For the transition matrix, we'll only use events where department_group exists, which represents when patients are actually in one of our three tiers (ED, Floor, ICU).
    """)
    return


@app.cell
def _(df_filtered):
    # Check if timestamps are in proper chronological order
    timestamp_order_issues = df_filtered.groupby('pat_enc_csn_id').apply(
        lambda x: (x.sort_values('effective_time')['effective_time'].diff().dt.total_seconds() < 0).any()
    )

    print(f"Encounters with timestamps going BACKWARDS: {timestamp_order_issues.sum()}")
    return


@app.cell
def _(df_filtered):
    # Find an encounter with seq_num issues
    encounters_multi_events = df_filtered.groupby('pat_enc_csn_id').filter(lambda x: len(x) > 1)
    seq_check = encounters_multi_events.groupby('pat_enc_csn_id').apply(
        lambda x: (x.sort_values('seq_num_in_enc')['effective_time'].diff().dt.total_seconds() < 0).any()
    )
    problem_encounter = seq_check[seq_check].index[0]

    print(f"Encounter with Sequencing Issue: {problem_encounter}:\n")
    return (problem_encounter,)


@app.cell
def _(df, problem_encounter):
    # Before filtering, all events from original data
    print("Before Filtering (Original Data):")
    before_filter = df[df['pat_enc_csn_id'] == problem_encounter][
        ['seq_num_in_enc', 'effective_time', 'event_type', 'patient_class', 'department_name']
    ].sort_values('seq_num_in_enc')

    for before_idx, before_row in before_filter.iterrows():
        print(f"seq={before_row['seq_num_in_enc']:.0f} | {before_row['effective_time']} | {before_row['event_type']:15} | {before_row['patient_class'][:30]:30}")
    return


@app.cell
def _(df_filtered, problem_encounter):
    # After filtering, only kept events
    print("\n\nAfter Filtering (Kept Events):")
    after_filter = df_filtered[df_filtered['pat_enc_csn_id'] == problem_encounter][
        ['seq_num_in_enc', 'effective_time', 'event_type', 'patient_class', 'department_name']
    ].sort_values('seq_num_in_enc')

    for after_idx, after_row in after_filter.iterrows():
        print(f"seq={after_row['seq_num_in_enc']:.0f} | {after_row['effective_time']} | {after_row['event_type']:15} | {after_row['patient_class'][:30]:30}")
    return


@app.cell
def _(df_filtered, pd):
    # How widespread is Nan problem in sequencing?
    nan_seq = df_filtered[df_filtered['seq_num_in_enc'].isna()]
    print(f"Total events with NaN seq_num: {len(nan_seq):,}")
    print(f"Total encounters with at least one NaN seq: {nan_seq['pat_enc_csn_id'].nunique():,}")
    print(f"Out of total encounters: {df_filtered['pat_enc_csn_id'].nunique():,}")
    print(f"Percentage of encounters affected: {nan_seq['pat_enc_csn_id'].nunique() / df_filtered['pat_enc_csn_id'].nunique() * 100:.1f}%")

    print("\n--- Event type breakdown for NaN seq events ---")
    print(nan_seq['event_type'].value_counts())

    print("\n--- Patient class breakdown for NaN seq events ---")
    print(nan_seq['patient_class'].value_counts().head(10))

    print("\n--- Department group breakdown for NaN seq events ---")
    print(nan_seq['department_group'].value_counts().head(10))

    # Looking at 5 example encounters
    nan_encs = nan_seq['pat_enc_csn_id'].unique()[:5]

    for enc in nan_encs:
        enc_data = df_filtered[df_filtered['pat_enc_csn_id'] == enc].sort_values('effective_time')
        print(f"\n{'='*80}")
        print(f"Encounter: {enc} ({len(enc_data)} events)")
        print(f"{'='*80}")
        for _, r in enc_data.iterrows():
            seq = r['seq_num_in_enc']
            seq_str = f"seq={int(seq)}" if pd.notna(seq) else "seq=NaN"
            print(f"  {seq_str:>10} | {r['effective_time']} | {str(r['event_type']):20s} | {str(r['patient_class']):30s} | {str(r['department_group'])}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The seq_num_in_enc field is scoped to the Inpatient Admission patient class, not the full encounter.
    When a patient starts in the ED and converts to inpatient, the initial ED Admission event receives no
    sequence number (NaN). This is systematic, not a data quality issue. Since all NaN cases follow this
    pattern and timestamps are in correct chronological order, we will use effective_time to order events
    and extract transitions — no resequencing needed.
    """)
    return


@app.cell
def _(df_clean):
    # Where is ED hiding?
    ed_events = df_clean[df_clean['patient_class'].str.contains('Emergency', na=False)]
    print(f"ED events: {len(ed_events):,}")
    print(f"\nED department_group values:")
    print(ed_events['department_group'].value_counts(dropna=False))

    print(f"\nED department_name samples:")
    print(ed_events['department_name'].value_counts().head(10))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ED events (306,495) have NaN for department_group in the source data — the field was never populated
    for emergency departments. This accounts for the majority of the ~556K missing department_group values
    identified earlier. ED events are still present in df_clean and can be identified via patient_class
    ("Emergency Department - Encounter") or department_name prefix (LJ/HC/EC EMERGENCY DEPT). We will
    fill in "ED" as the department_group for these events before building the transition matrix.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tier Mapping

    Our simulation models patient flow between 9 states: 3 care tiers (ED, ICU, Floor) × 3 campuses
    (La Jolla, Hillcrest, East Campus), plus discharge home. The raw data has ~10 granular department
    groups (MedSurg, PCU, IMU, OB, etc.) that need to be collapsed into these 3 tiers. Mapping: ED
    events (identified via patient_class/department_name since department_group is NaN for ED), ICU stays
    as ICU, and everything else (MedSurg, OB, PCU, IMU, MedSTele, Behavioral Health, Nursery, NICU) as
    Floor. Events with no tier (perioperative/holding) will be skipped in transition extraction so
    transitions connect around them (e.g., ED → periop → Floor becomes ED → Floor).
    """)
    return


@app.cell
def _(df_clean, pd):
    # Create tier mapping
    def assign_tier(row):
        if row['patient_class'] == 'Emergency Department - Encounter' or \
           (pd.isna(row['department_group']) and 'EMERGENCY' in str(row['department_name'])):
            return 'ED'
        elif row['department_group'] == 'ICU':
            return 'ICU'
        elif pd.notna(row['department_group']):
            return 'Floor'
        else:
            return None  # perioperative/holding - no tier

    df_clean_tier = df_clean.copy()
    df_clean_tier['tier'] = df_clean_tier.apply(assign_tier, axis=1)

    print("Tier counts:")
    print(df_clean_tier['tier'].value_counts(dropna=False))

    print("\nTier x Location:")
    print(pd.crosstab(df_clean_tier['tier'], df_clean_tier['location']))
    return (df_clean_tier,)


@app.cell
def _(df_clean_tier):
    # Check None tier events
    no_tier = df_clean_tier[df_clean_tier['tier'].isna()]
    print(f"Events with no tier: {len(no_tier):,}")
    print(f"Encounters affected: {no_tier['pat_enc_csn_id'].nunique():,}")
    print(f"\ndepartment_name for no-tier events:")
    for dept, cnt in no_tier['department_name'].value_counts().items():
        print(f"  {dept}: {cnt:,}")
    return


@app.cell
def _(df_clean_tier, pd):
    # Create state column: tier + location
    df_clean_tier['state'] = df_clean_tier['tier'] + '_' + df_clean_tier['location']

    # Sort by encounter and time, drop no-tier events
    df_sorted = df_clean_tier[df_clean_tier['tier'].notna()].sort_values(['pat_enc_csn_id', 'effective_time'])

    # Extract transition pairs
    transition_pairs = []
    for enc_id, group in df_sorted.groupby('pat_enc_csn_id'):
        states = group['state'].values
        # Remove consecutive duplicates
        path = [states[0]]
        for st in states[1:]:
            if st != path[-1]:
                path.append(st)
        # Add discharge as final state
        path.append('Discharge')
        # Collect pairs
        for i in range(len(path) - 1):
            transition_pairs.append((path[i], path[i+1]))

    transitions_df = pd.DataFrame(transition_pairs, columns=['from_state', 'to_state'])
    print(f"Total transitions: {len(transitions_df):,}")
    print(f"\nTransition counts:")
    for pair, n_trans in transitions_df.value_counts().head(20).items():
        print(f"  {pair[0]} -> {pair[1]}: {n_trans:,}")
    return (transitions_df,)


@app.cell
def _(pd, transitions_df):
    # Define states
    from_states = [
        'ED_La Jolla', 'ED_Hillcrest', 'ED_East Campus',
        'Floor_La Jolla', 'Floor_Hillcrest', 'Floor_East Campus',
        'ICU_La Jolla', 'ICU_Hillcrest', 'ICU_East Campus'
    ]
    to_states = from_states + ['Discharge']

    # Build count matrix
    count_matrix = pd.DataFrame(0, index=from_states, columns=to_states)
    for (f, t), num in transitions_df.value_counts().items():
        if f in from_states and t in to_states:
            count_matrix.loc[f, t] = num

    print("Count Matrix:")
    print(count_matrix)

    # Convert to probabilities (each row sums to 1)
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    print("\nProbability Matrix:")
    print(prob_matrix.round(4))

    # Verify rows sum to 1
    print(f"\nRow sums: {prob_matrix.sum(axis=1).values}")
    return (prob_matrix,)


@app.cell
def _(df_clean_tier):
    # Build state blocks with duration
    df_states = df_clean_tier[df_clean_tier['tier'].notna()].copy()
    df_states['state'] = (df_states['tier'] + '_' + df_states['location']).str.upper().str.replace(' ', '_')
    df_states = df_states.sort_values(['pat_enc_csn_id', 'effective_time'])

    # Detect state changes and assign block IDs
    df_states['prev_state'] = df_states.groupby('pat_enc_csn_id')['state'].shift()
    df_states['is_new'] = (df_states['state'] != df_states['prev_state']).astype(int)
    df_states['block'] = df_states.groupby('pat_enc_csn_id')['is_new'].cumsum()

    # Collapse to one row per block with time_in, time_out, duration
    state_blocks = (
        df_states.groupby(['pat_enc_csn_id', 'block', 'state'])
        .agg(time_in=('effective_time', 'min'), time_out=('effective_time', 'max'))
        .reset_index()
        .sort_values(['pat_enc_csn_id', 'block'])
    )
    state_blocks['duration_hours'] = (state_blocks['time_out'] - state_blocks['time_in']).dt.total_seconds() / 3600

    # Build trajectory library
    trajectory_library = (
        state_blocks
        .sort_values(['pat_enc_csn_id', 'block'])
        .groupby('pat_enc_csn_id')
        .apply(lambda g: list(zip(g['state'], g['duration_hours'])))
        .to_dict()
    )

    # Preview a few
    # for eid, traj in list(trajectory_library.items())[:5]:
    #     traj_str = ' → '.join([f"{state} ({dur:.1f}h)" for state, dur in traj])
    #     print(f"{eid}: {traj_str}")

    print(f"\nTotal trajectories: {len(trajectory_library):,}")
    return state_blocks, trajectory_library


@app.cell
def _(state_blocks):
    # Get durations per state from state_blocks
    for state_name, grp in state_blocks.groupby('state'):
        durations = grp['duration_hours']
        print(f"{state_name}: n={len(grp):,}, mean={durations.mean():.1f}h, median={durations.median():.1f}h, std={durations.std():.1f}h")
    return


@app.cell
def _(np, state_blocks, stats):
    distributions = {
        'lognorm': stats.lognorm,
        'gamma': stats.gamma,
        'expon': stats.expon,
        'weibull_min': stats.weibull_min
    }

    fit_results = {}
    for sname, sgrp in state_blocks.groupby('state'):
        dur_hrs = sgrp['duration_hours'].dropna()
        dur_hrs = dur_hrs[dur_hrs > 0]

        print(f"\n{'='*60}")
        print(f"Fitting: {sname} (n={len(dur_hrs):,})")
        print(f"{'='*60}")

        best_name = None
        best_sse = float('inf')
        best_params = None

        for dist_name, dist in distributions.items():
            params = dist.fit(dur_hrs)
            pdf_values = dist.pdf(sorted(dur_hrs), *params)
            sse = np.sum((np.histogram(dur_hrs, bins=50, density=True)[0] - 
                           dist.pdf(np.histogram(dur_hrs, bins=50)[1][:-1] + 
                           np.diff(np.histogram(dur_hrs, bins=50)[1])/2, *params))**2)
            print(f"  {dist_name}: SSE={sse:.6f}, params={tuple(round(p,4) for p in params)}")

            if sse < best_sse:
                best_sse = sse
                best_name = dist_name
                best_params = params

        fit_results[sname] = {'dist': best_name, 'params': best_params, 'sse': best_sse}
        print(f"  → Best: {best_name}")
    return


@app.cell
def _(df_clean_tier):
    boarded = df_clean_tier[
        (df_clean_tier['department_name'].str.contains('EMERGENCY', na=False)) & 
        (df_clean_tier['patient_class'] != 'Emergency Department - Encounter')
    ]
    print(f"Potential boarded events: {len(boarded):,}")
    print(f"\nTier breakdown:")
    for tier_val, n in boarded['tier'].value_counts(dropna=False).items():
        print(f"  {tier_val}: {n:,}")
    return (boarded,)


@app.cell
def _(boarded, state_blocks):
    # Compare ED durations: true ED patients vs boarded
    ed_blocks = state_blocks[state_blocks['state'].str.startswith('ED_')]

    # Tag which encounters have boarded events
    boarded_encs = boarded['pat_enc_csn_id'].unique()
    ed_blocks_tagged = ed_blocks.copy()
    ed_blocks_tagged['is_boarded'] = ed_blocks_tagged['pat_enc_csn_id'].isin(boarded_encs)

    print("ED durations for encounters WITH boarding:")
    print(ed_blocks_tagged[ed_blocks_tagged['is_boarded']]['duration_hours'].describe())

    print("\nED durations for encounters WITHOUT boarding:")
    print(ed_blocks_tagged[~ed_blocks_tagged['is_boarded']]['duration_hours'].describe())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Boarded Patients (Known Limitation)

    148,234 ED events (~33% of all ED events) are boarded patients — physically in the ED but already
    admitted as inpatients. These inflate ED duration distributions significantly: median 14.7h with
    boarding vs 3.3h without. For v1 of the simulation, boarded time is counted as ED occupancy. Future
    iterations could split these encounters at the patient_class change point to separate true ED time
    from boarding time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Arrival Rates: Weekday vs Weekend

    The simulation needs to know how many patients enter the system per day at each campus. Patient arrival
    patterns differ between weekdays and weekends (e.g., fewer elective admissions on weekends, different ED
    volumes). We calculate mean daily arrivals by campus (La Jolla, Hillcrest, East Campus) split by
    weekday vs weekend. These rates will drive patient generation in the simulation — each simulated day
    samples from the appropriate arrival distribution to determine how many new patients enter each campus.
    """)
    return


@app.cell
def _(df_clean_tier):
    # Get first event per encounter (arrival) with tier
    arrivals = df_clean_tier[df_clean_tier['tier'].notna()].sort_values('effective_time').groupby('pat_enc_csn_id').first().reset_index()
    arrivals['day_of_week'] = arrivals['effective_time'].dt.dayofweek
    arrivals['is_weekend'] = arrivals['day_of_week'].isin([5, 6])
    arrivals['date'] = arrivals['effective_time'].dt.date

    # Daily arrival counts by campus and weekday/weekend
    daily = arrivals.groupby(['location', 'is_weekend', 'date']).size().reset_index(name='daily_arrivals')

    print("Mean daily arrivals by campus and day type:")
    for (loc, wknd), sub in daily.groupby(['location', 'is_weekend']):
        day_type = 'Weekend' if wknd else 'Weekday'
        print(f"  {loc} ({day_type}): mean={sub['daily_arrivals'].mean():.1f}, std={sub['daily_arrivals'].std():.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # UCSD Health DES — EDA & Data Preparation Summary

    ## Data Cleaning Pipeline

    **Starting point:** 2.1M ADT events across 985K patient encounters (2025).

    **Step 1 — Filter to bed-occupying patients:** Removed non-bed-occupying patient classes (outpatient radiology, infusion, therapy, etc.), reducing to 1.15M rows across 153,878 encounters.

    **Step 2 — Drop single-event encounters:** 1,720 encounters (1.1%) had only a single event, producing no usable transitions. These were incomplete records in the source system (orphan discharges, registration-only entries), not artifacts of our filtering. Dropped to yield 152,158 encounters / 1,151,386 events.

    **Step 3 — Tier mapping:** Collapsed ~10 granular department groups into 3 care tiers for the simulation's 9-state model (3 tiers × 3 campuses). ED events had NaN for `department_group` in the source data (never populated for emergency departments) and were identified via `patient_class` and `department_name`. Mapping: ED → ED, ICU → ICU, everything else (MedSurg, OB, PCU, IMU, MedSTele, Behavioral Health, Nursery, NICU) → Floor. 100,633 events with no tier (perioperative/holding areas) are skipped in transition extraction so transitions connect around them (e.g., ED → periop → Floor becomes ED → Floor).

    **Step 4 — Timestamp-based ordering:** The `seq_num_in_enc` field is scoped to the Inpatient Admission patient class, not the full encounter. ED Admission events receive NaN sequence numbers. Since timestamps are in correct chronological order, all event ordering uses `effective_time` — no resequencing needed.

    ---

    ## 1. Transition Matrix (9×10)

    Transition probabilities between 9 states (ED/Floor/ICU × La Jolla/Hillcrest/East Campus) plus Discharge. Built by sorting each encounter by `effective_time`, extracting the sequence of states (dropping NaN-tier events and consecutive duplicates), and counting state-to-state transitions. 185,866 total transitions extracted.

    |  | ED_LJ | ED_HC | ED_EC | Fl_LJ | Fl_HC | Fl_EC | ICU_LJ | ICU_HC | ICU_EC | Disch |
    |---|---|---|---|---|---|---|---|---|---|---|
    | **ED_LJ** | 0.000 | 0.001 | 0.000 | 0.163 | 0.008 | 0.015 | 0.027 | 0.001 | 0.001 | 0.785 |
    | **ED_HC** | 0.001 | 0.000 | 0.000 | 0.001 | 0.173 | 0.023 | 0.001 | 0.028 | 0.001 | 0.774 |
    | **ED_EC** | 0.002 | 0.006 | 0.000 | 0.000 | 0.005 | 0.112 | 0.000 | 0.001 | 0.019 | 0.854 |
    | **Fl_LJ** | 0.001 | 0.000 | 0.000 | 0.000 | 0.002 | 0.001 | 0.042 | 0.000 | 0.000 | 0.955 |
    | **Fl_HC** | 0.000 | 0.001 | 0.000 | 0.010 | 0.000 | 0.002 | 0.002 | 0.037 | 0.000 | 0.949 |
    | **Fl_EC** | 0.000 | 0.000 | 0.000 | 0.004 | 0.011 | 0.000 | 0.000 | 0.000 | 0.027 | 0.957 |
    | **ICU_LJ** | 0.000 | 0.000 | 0.000 | 0.559 | 0.001 | 0.000 | 0.000 | 0.001 | 0.001 | 0.438 |
    | **ICU_HC** | 0.000 | 0.001 | 0.000 | 0.003 | 0.743 | 0.000 | 0.017 | 0.000 | 0.005 | 0.230 |
    | **ICU_EC** | 0.000 | 0.000 | 0.002 | 0.006 | 0.005 | 0.648 | 0.032 | 0.018 | 0.000 | 0.289 |

    **Key patterns:** ED → Discharge is dominant (77–85%), reflecting most ED patients going home. Floor → Discharge (95–96%) shows most floor patients discharge directly. ICU → Floor same campus (56–74%) is the primary step-down pathway. Cross-campus transfers exist but are rare.

    ---

    ## 2. Duration Distributions (Lognormal Fits)

    Length of stay per state block, fit using scipy. Lognormal wins 8 of 9 states (exponential for ED East Campus). All states are heavily right-skewed (mean >> median) due to long-stay outliers.

    | State | Best Fit | Params (shape, loc, scale) | Mean (h) | Median (h) |
    |---|---|---|---|---|
    | ED_East Campus | expon | (0.017, 3.67) | 3.7 | 2.5 |
    | ED_Hillcrest | lognorm | (0.777, -0.658, 5.644) | 7.0 | 4.8 |
    | ED_La Jolla | lognorm | (1.255, -0.107, 6.123) | 13.7 | 4.8 |
    | Floor_East Campus | lognorm | (1.001, -1.784, 59.882) | 110.5 | 53.1 |
    | Floor_Hillcrest | lognorm | (1.186, -3.259, 59.236) | 112.1 | 61.9 |
    | Floor_La Jolla | lognorm | (1.177, -3.345, 52.999) | 99.8 | 53.8 |
    | ICU_East Campus | lognorm | (0.946, -2.649, 48.313) | 74.6 | 42.8 |
    | ICU_Hillcrest | lognorm | (1.002, -2.176, 59.649) | 102.2 | 51.6 |
    | ICU_La Jolla | lognorm | (1.120, -0.956, 46.637) | 93.2 | 43.4 |

    ---

    ## 3. Trajectory Library

    150,337 patient trajectories with state and duration at each stop. Each encounter maps to a sequence of `(state, duration_hours)` tuples. Example: `ED_La Jolla (4.5h) → Floor_La Jolla (48.2h) → Discharge`. Combines the transition logic with the duration data for full patient path reconstruction.

    ---

    ## 4. Arrival Rates (Weekday vs Weekend)

    Mean daily arrivals by campus and day type, calculated from the first event of each encounter.

    | Campus | Weekday (mean ± std) | Weekend (mean ± std) |
    |---|---|---|
    | East Campus | 92.4 ± 10.5 | 83.3 ± 10.5 |
    | Hillcrest | 144.7 ± 19.9 | 125.1 ± 10.3 |
    | La Jolla | 198.2 ± 26.2 | 148.9 ± 14.5 |

    All campuses show 10–25% fewer arrivals on weekends, consistent with reduced elective admissions. La Jolla has the largest absolute drop.

    ---

    ## Known Limitation: Boarded Patients

    148,234 ED events (~33% of all ED events) are boarded patients — physically in the ED but already admitted as inpatients. These inflate ED duration distributions significantly: median 14.7h with boarding vs 3.3h without. For v1 of the simulation, boarded time is counted as ED occupancy. Future iterations could split these encounters at the patient_class change point to separate true ED time from boarding time.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Saving all tables to be imported to another notebook for DES
    """)
    return


@app.cell
def _():
    # df_clean_tier.to_csv('ADT_cleaned_with_tiers.csv')
    return


@app.cell
def _(prob_matrix):
    # prob_matrix.to_csv('probability_matrix.csv')
    prob_matrix
    return


@app.cell
def _(pd):

    fitted_dist_data = [
        ["ED_East Campus", "expon", (0.017, 3.67), 3.7, 2.5],
        ["ED_Hillcrest", "lognorm", (0.777, -0.658, 5.644), 7.0, 4.8],
        ["ED_La Jolla", "lognorm", (1.255, -0.107, 6.123), 13.7, 4.8],
        ["Floor_East Campus", "lognorm", (1.001, -1.784, 59.882), 110.5, 53.1],
        ["Floor_Hillcrest", "lognorm", (1.186, -3.259, 59.236), 112.1, 61.9],
        ["Floor_La Jolla", "lognorm", (1.177, -3.345, 52.999), 99.8, 53.8],
        ["ICU_East Campus", "lognorm", (0.946, -2.649, 48.313), 74.6, 42.8],
        ["ICU_Hillcrest", "lognorm", (1.002, -2.176, 59.649), 102.2, 51.6],
        ["ICU_La Jolla", "lognorm", (1.120, -0.956, 46.637), 93.2, 43.4],
    ]

    fitted_dist_df = pd.DataFrame(
        fitted_dist_data,
        columns=["state", "best_fit", "params)", "mean", "median"]
    )

    # fitted_dist_df.to_csv('states_fitted_distribution_df.csv')
    # fitted_dist_df
    return


@app.cell
def _(pd):
    arrival_rates_data = [
        ["East Campus", 92.4, 10.5, 83.3, 10.5],
        ["Hillcrest", 144.7, 19.9, 125.1, 10.3],
        ["La Jolla", 198.2, 26.2, 148.9, 14.5],
    ]

    df_week = pd.DataFrame(
        arrival_rates_data,
        columns=[
            "campus",
            "weekday_mean",
            "weekday_std",
            "weekend_mean",
            "weekend_std",
        ]
    )

    # df_week.to_csv('weekday_weekend_arrival_rates_df.csv')
    df_week
    return


@app.cell
def _(trajectory_library):
    type(trajectory_library)
    return


@app.cell
def _():
    # import json
    # with open('trajectory_library.json', 'w') as json_file:
    #     json.dump(trajectory_library, json_file, indent=4)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
