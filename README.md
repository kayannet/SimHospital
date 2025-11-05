<h1 style="text-align:center; color:#4F81BD;">🏥 SimHospital (Project 1): Parameterizing an Emergency Department DES Using MIMIC-IV Data</h1>

This project represents **Phase 1** of *SimHospital*, a multi-stage initiative to build a hospital-scale **Discrete-Event Simulation (DES)** framework for patient flow modeling and operational decision support.  
The current phase focuses on developing a **data-driven baseline model** of an Emergency Department (ED) using the publicly available **MIMIC-IV ED dataset**.  
This baseline model provides empirical parameters—such as wait times, length of stay, arrival patterns, and disposition ratios—that will later be used to calibrate and validate the hospital-level simulator with UCSD Health aggregate data.

By grounding the DES in de-identified MIMIC data first, we ensure that the workflow is **reproducible, ethically compliant, and generalizable** before scaling to institutional data access at UC San Diego Health.

---

<h2 style="color:#4F81BD;">Data Access</h2>

Due to data-use agreements, the MIMIC-IV ED dataset cannot be hosted publicly.  
You can download the complete data folder here:

📁 [Download MIMIC_ED Folder](https://drive.google.com/drive/folders/your-folder-id?usp=share_link)

After downloading, the folder structure should look like this:
```
MIMIC_ED/
├── raw/
│ └── mimicel.csv
├── cleaned/
│ └── mimicel_clean.csv
└── README_data.txt
```
Once downloaded, place the MIMIC_ED folder in the project root directory so that relative paths in the notebooks work correctly.

---

<h2 style="color:#4F81BD;">Notebook Overview</h2>

| Notebook | Description |
|-----------|--------------|
| `01_clean_mimic_ed.ipynb` | Cleans, inspects, and extracts key operational metrics (wait times, length of stay, arrival rates, and disposition ratios) from the MIMIC-IV ED dataset to parameterize the baseline DES model. |

> Future notebooks (e.g., simulation modeling, validation, and scenario testing) will be added as the project progresses.

---

<h2 style="color:#4F81BD;">Environment Setup</h2>

To reproduce results locally, create a new virtual environment and install dependencies:

```bash
pip install duckdb pandas matplotlib seaborn jupyter
```
**Optional (for interactive visualization):**
```bash
pip install plotly
```
**Then launch Jupyter and open the notebook:**
```bash
jupyter notebook notebooks/01_clean_mimic_ed.ipynb
```
<h2 style="color:#4F81BD;">Authors</h2>

<p style="font-size:16px; line-height:1.6;">
<b>Nadine Orriss</b> — B.S. Data Science, UC San Diego (Class of 2026)<br>
<b>Kayanne Tran</b> — B.S. Data Science, UC San Diego<br><br>

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
