<h1 style="text-align:center; color:#4F81BD;">🏥 SimHospital Quarter 2: Migrating to UCSD Health Data</h1>

This project represents **Quarter 2** of *SimHospital*, a multi-stage initiative to develop data-driven tools for hospital operational planning and patient flow optimization using **Discrete-Event Simulation (DES)**.

This project began using MIMIC to build the skeleton of our hospital simulation, and we were able to switch to UCSD Health data and apply the same logics.

---

<h2 style="color:#4F81BD;">Data Access</h2>
Due to HIPAA restrictions, the UCSD Health data used in this project cannot be shared. Only aggregated results, tables, and simulation outputs are included in this repository.

If you would like to run the simulation yourself with real-world clinical data, you can instead use the publicly available MIMIC-IV Emergency Department (MIMIC-ED) dataset. Because of data-use agreements and file size limitations, the dataset must be downloaded separately and cannot be hosted on GitHub.

The MIMIC-ED folder should be placed outside the project directory so that the two are sibling folders and file paths resolve correctly.
📁 [Download MIMIC_ED Folder](https://drive.google.com/drive/folders/1R39eyLbLz9ccqoQCbLDfq12LXLs3ZFt9?usp=share_link)

**MIMIC-ED should be placed outside the SimHospital folder so that the two are sibling directories and file paths in the project function correctly.**

<h2 style="color:#4F81BD;">Folder Structure</h2>

```
/MIMIC-ED                     <-- data folder (outside of SimHospital)
  |  
  ├── raw                     <-- raw data
  ├── cleaned                 <-- initial preprocessing/cleaning
  └──processed               <-- transformed and working dataframes

/SimHospital
  │
  ├── src                      <-- Main source folder
  │   ├── notebooks            <-- Jupyter notebooks (EDA, visualizations, modeling)
  │   ├── simulation           <-- Python files with simulation builds
  │   └── r_script             <-- R scripts (DES, simulation, modeling)
  │
  |
  ├── results                  <-- Outputs (plots, tables, metrics)
  |
  |
  ├── docs                     <-- Documents (ex. old readme.md versions)   
  |
  ├── requirements.txt     
  └── README.md
  
```
---

<h2 style="color:#4F81BD;">Notebook Overview</h2>



From Quarter 1:
These files are our EDA and simulation building using MIMIC Data.

| File | Language | Description |
|------|----------|-------------|
| [ucsd_health_simulation.html](src/notebooks/ucsd_health_simulation.html) | Python | DES file with UCSD health data, modeled using multiple arrival rates in order to infer proper resource capacities |
| [mimic_simulation.py](src/simulation/mimic_simulation.py) | Python | Multi-hospital model with MIMIC data, synthesized 3 different hospitals (to resemble UCSD's system) and experimented with dynamic routing based on whether or not the patient is marked as high acuity |
| [ucsd_health_eda.html](src/notebooks/ucsd_health_eda.html) | Python | Exploratory data analysis of UCSD Health data, includes aggregates and tables (no PHI)|



From Quarter 1:
| File | Language | Description |
|----------|-------------|-------------|
| [01_clean_mimic_ed.ipynb](src/notebooks/mimic_eda/01_clean_mimic_ed.ipynb) | Python | Loads the raw MIMIC-IV ED extract, inspects the schema, and produces a cleaned encounter-level table (`mimicel_clean.csv`) with one row per ED stay and standardized arrival/triage/depart timestamps. This dataset is the basis for estimating arrival rates, door-to-triage times, and length-of-stay distributions for the baseline DES model. |
| [02_activity_sequence_analysis.ipynb](src/notebooks/mimic_eda/02_activity_sequence_analysis.ipynb) | Python|  Uses a 5% patient sample to explore ED activity sequences. Deduplicates the activity log, builds an interactive patient-journey lookup tool, and computes transition probabilities and mean inter-activity times between key ED steps (Enter ED → Triage → Vital signs → Med reconciliation/dispensations → Discharge). |
| [03_build_sim_input_tables.ipynb](src/notebooks/mimic_eda/03_build_sim_input_tables.ipynb) | Python| Processes and normalizes the cleaned activity log into four analysis-ready datasets—`ed_stays`, `ed_activity_log`, `ed_diagnoses`, and `ed_medications`—and saves them as CSV files. Includes data quality validation, deduplication, and standardization. These four datasets are the direct inputs to the discrete-event simulation model. |
| [04_training_df_aggregates.ipynb](src/notebooks/mimic_eda/04_training_df_aggregates.ipynb) | Python| Aggregates and prepares the cleaned ED datasets into a training dataframe for modeling in R. Includes summary statistics, probability aggregates, and visualizations that inform simulation branching logic. |
[05_R_model_v1.ipynb](src/notebooks/archive/05_R_model_v1.ipynb) | R| Feature engineering and creates baseline model, XGBoost only model, and ensemble model. Includes performance evaluation and comparisons. |
| [hospital_sim_v1.ipynb](src/notebooks/mimic_eda/hospital_sim_v1.ipynb)| R| Implements a baseline ED simulation: patients arrive, length-of-stay (LOS) is sampled from ed_stays.csv, and admitted patients’ LOS is predicted using the R model.|



---

<h2 style="color:#4F81BD;">Environment Setup</h2>

**To reproduce results locally, first clone this repository into your desired directory/environment**
```bash
git clone https://github.com/your-username/SimHospital.git
cd SimHospital
```
**Then install the dependencies from `requirements.txt`**
```bash
pip install -r requirements.txt
```
**Launch Jupyter and open the notebook:**
```bash
jupyter notebook src/notebooks/01_clean_mimic_ed.ipynb
```
<h2 style="color:#4F81BD;">Authors</h2>

<p style="font-size:16px; line-height:1.6;">
<b>Nadine Orriss</b> — B.S. Data Science, UC San Diego (Class of 2026)<br>
<b>Kayanne Tran</b> — B.S. Data Science, UC San Diego (Class of 2026)<br><br>

<b>Faculty Mentor:</b> Dr. Karandeep Singh, MD, MMSc — Joan and Irwin Jacobs Chancellor’s Endowed Chair in Digital Health Innovation; Associate Professor of Biomedical Informatics, UC San Diego; Chief Health AI Officer, UC San Diego Health<br>
<b>Project Mentor:</b> Dr. Aaron Boussina — Assistant Professor of AI & Digital Health, UC San Diego Health
</p>

<details>
<summary><h2 style="color:#4F81BD;">Project Status</h2></summary>

**Current Stage:** Data Cleaning & Metric Extraction (Phase 1)<br>
**Next Steps:** Develop core DES model modules and validate against UCSD Health aggregates<br>
**Goal:** Build a scalable, data-driven simulation framework for emergency and hospital-wide operations

</details>


<p style="text-align:center; font-style:italic;"> Last updated November 2025 · SimHospital (Project 1)</p>
