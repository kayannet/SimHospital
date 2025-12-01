<h1 style="text-align:center; color:#4F81BD;">🏥 SimHospital (Project 1): Parameterizing an Emergency Department DES Using MIMIC-IV Data</h1>

This project represents **Phase 1** of *SimHospital*, a multi-stage initiative to build a hospital-scale **Discrete-Event Simulation (DES)** framework for patient flow modeling and operational decision support.  
The current phase focuses on developing a **data-driven baseline model** of an Emergency Department (ED) using the publicly available **MIMIC-IV ED dataset**.  
This baseline model provides empirical parameters—such as wait times, length of stay, arrival patterns, and disposition ratios—that will later be used to calibrate and validate the hospital-level simulator with UCSD Health aggregate data.

By grounding the DES in de-identified MIMIC data first, we ensure that the workflow is **reproducible, ethically compliant, and generalizable** before scaling to institutional data access at UC San Diego Health.

---

<h2 style="color:#4F81BD;">Data Access</h2>

Due to data-use agreements, the MIMIC-IV ED dataset cannot be hosted publicly. The files are also too large for GitHub, so they must be downloaded and stored locally outside of the project root folder.
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
  │   ├── notebooks            <-- Python notebooks (EDA, visualizations, modeling)
  │   └── r_script             <-- R scripts (DES, simulation, modeling)
  |
  ├── results                  <-- Outputs (plots, tables, metrics)
  |
  ├── requirements.txt     
  └── README.md
  
```
---

<h2 style="color:#4F81BD;">Notebook Overview</h2>

| Notebook | Language | Description |
|----------|-------------|-------------|
| [01_clean_mimic_ed.ipynb](src/notebooks/01_clean_mimic_ed.ipynb) | Python | Loads the raw MIMIC-IV ED extract, inspects the schema, and produces a cleaned encounter-level table (`mimicel_clean.csv`) with one row per ED stay and standardized arrival/triage/depart timestamps. This dataset is the basis for estimating arrival rates, door-to-triage times, and length-of-stay distributions for the baseline DES model. |
| [02_activity_sequence_analysis.ipynb](src/notebooks/02_activity_sequence_analysis.ipynb) | Python|  Uses a 5% patient sample to explore ED activity sequences. Deduplicates the activity log, builds an interactive patient-journey lookup tool, and computes transition probabilities and mean inter-activity times between key ED steps (Enter ED → Triage → Vital signs → Med reconciliation/dispensations → Discharge). |
| [03_build_sim_input_tables.ipynb](src/notebooks/03_build_sim_input_tables.ipynb) | Python| Processes and normalizes the cleaned activity log into four analysis-ready datasets—`ed_stays`, `ed_activity_log`, `ed_diagnoses`, and `ed_medications`—and saves them as CSV files. Includes data quality validation, deduplication, and standardization. These four datasets are the direct inputs to the discrete-event simulation model. |
| [04_training_df_aggregates.ipynb](src/notebooks/04_training_df_aggregates.ipynb) | Python| Aggregates and prepares the cleaned ED datasets into a training dataframe for modeling in R. Includes summary statistics, probability aggregates, and visualizations that inform simulation branching logic. |
| [hospital_sim_v1.ipynb](src/notebooks/hospital_sim_v1.ipynb)| R| Implements a baseline ED simulation: patients arrive, length-of-stay (LOS) is sampled from ed_stays.csv, and admitted patients’ LOS is predicted using the R model.|



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
