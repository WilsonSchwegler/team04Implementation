# Data-Driven MLB Pitch Planning System Team04 - Wilson Schwegler, Noah Betoshana

This repository contains our final project for an interactive MLB pitch-planning system. Our application focuses on early count pitch sequencing. After a user selects a pitcher, batter, and first pitch, the system recommends a second pitch type and location and explains that recommendation through coordinated visual views.

The system combines:
- an Explore page for browsing pitcher and hitter archetypes before opening the matchup view,
- a trained pitch recommendation pipeline built from MLB Statcast data,
- an interactive frontend for trajectory and bucket based inspection,
- and batter specific or handedness level strike zone surfaces.

## 1. Required Downloads

Before running the project, download the following files and place them in the listed locations.

1. `p1` planner table
- Download: [Google Drive file](https://drive.google.com/file/d/1Dk2myQxyFsZ85tqkAh71YtpQPVwJ9A4A/view?usp=sharing)
- Place in: `models/p1/tables/`

2. First-two-pitch source CSV
- Download: [Google Drive file](https://drive.google.com/file/d/1JQ-hZLTLp6xxCuoVtDmBX8ilIhRKyRC1/view?usp=drive_link)
- Place in: `models/`

3. `v2` model assets folder
- Download: [Google Drive folder](https://drive.google.com/drive/folders/1QANUEmDCyW_E1SH7cPXnQUadzUVeDIr1?usp=drive_link)
- Place in: `models/v2/`
- *Download should be a folder named `tables`*

4. 2025 pitch level CSV
- Download: [Google Drive file](https://drive.google.com/file/d/1LcEBJRt0bU55yyEVygGQDJH7HfskYI9P/view?usp=sharing)
- Place in: `data/`



## 2. Conda Environment Setup

We recommend using Conda.

Create and activate the environment:

```bash
conda create -n ecs273-pitch python=3.10 -y
conda activate ecs273-pitch
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies once:

```bash
cd frontend
npm install
cd ..
```

## 3. Run The Backend

Start the backend from the repository root:

```bash
conda activate ecs273-pitch
python -m uvicorn backend.main:app --reload --port 8000
```

Important:
- the backend can take some time to fully start because it loads model artifacts, tables, caches, and runtime support files;
- wait until backend startup completes before expecting the frontend to work correctly;

## 4. Run The Frontend

In a second terminal, start the frontend:

```bash
conda activate ecs273-pitch
cd frontend
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## 5. Startup Note

The frontend depends on the backend API. If the frontend is started before the backend finishes loading, the app may initially show empty batter/pitcher lists or fail to load recommendations. If this happens:
- wait for backend startup to complete,
- reload the frontend page,
- then try again.

## 6. What The System Does

At a high level:
- the user selects a pitcher and batter,
- the user selects a first-pitch type and location,
- the backend estimates the likely first-pitch count transition,
- candidate second-pitch locations are generated from batter-side and pitcher-side historical bucket caches,
- each candidate is scored by an event-tree recommendation model,
- and the frontend visualizes recommended pitch shapes, target buckets, and explanatory views.

The frontend has two main experiences:
- `Explore`: an embedding view for browsing pitcher and hitter archetypes,
- `Matchup`: the main recommendation interface with 3D pitch trajectories, strike-zone context, pitch-two ranking, and bucket explanations.

## 7. Repository Structure

```text
Pitch-Sequencing-Optimization-and-Visualization-Tool/
├── README.md
├── requirements.txt
├── backend/
│   ├── main.py
│   └── v2_runtime.py
├── data/
│   ├── pitch_level_2025.csv
│   ├── pitch_type_averages_2025.csv
│   └── strike-zone and runtime support files
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── first_two_pitch_at_bats_2021_2025.csv
│   ├── p1/
│   │   ├── artifacts/
│   │   ├── tables/
│   │   └── modeling code for pitch 1
│   └── v2/
│       ├── artifacts/
│       ├── tables/
│       ├── location_grid.py
│       ├── cache_helpers.py
│       └── modeling code for pitch 2
```

## 10. Troubleshooting

- If the frontend loads but batter/pitcher lists are empty, the backend likely has not finished startup. Wait, then reload the page.
- If `POST /api/predict` fails, verify that all downloaded files were placed in the correct directories.
- If parquet loading fails, make sure `pyarrow` installed successfully from `requirements.txt`.
- If `npm run dev` fails because of missing frontend packages, run `npm install` again inside `frontend/`.
