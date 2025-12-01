# Reconstruction of Rail Incidents & Delay Propagation in UK

This toolkit integrates train schedules, incident and delay records, and passenger loading data from the UK’s [Rail Data Marketplace](https://raildata.org.uk/) to link together key variables of day-to-day rail operations. Our goal is to build a unified dataset that connects **system-level performance** (network-wide delays and cancellations) with **component-level events** (locations and durations of initial incidents, train movements, delay occurrences at stations, and passenger crowding).

This is one of the first efforts to generate a comprehensive dataset suitable for **validation of system-level rail models**, addressing a long-standing gap in **data availability for systems engineering**.

## What this toolkit invites innovation in

1. **Forecasting delay propagation** using AI and systems engineering to understand how local incidents ripple across the network.  
2. **Assessing and comparing station performance** under different demand levels and operational scenarios.  
3. **Exploring cause-and-effect relationships** between operational factors and delay severity.  
4. **Beyond** — we warmly welcome new ideas for new use cases and extensions.

## Research Project and contact
This repository is a result of the MSc research project by Marta Zarantonello, supervised by Dr Ji-Eun Byun, at the Research Division of Infrastructure & Environment, University of Glasgow. 
If you have any enquires, please contact [ji-eun.byun@glasgow.ac.uk](mailto:ji-eun.byun@glasgow.ac.uk).

## Data Setup

To use this tool, you need to download the NWR Historic Delay Attribution data from Network Rail, SWR Passenger Loadings and NWR Schedule data (or your organisation's portal). You can access these files from the Rail Data Marketplace (RDM) platform here: https://raildata.org.uk/. 
>  The data is not included in this repository due to licensing and size restrictions.

1. Create a `data/` folder in the project root if it doesn't exist.
2. Download the following files and save them in `data/`. Please do not create separate folders within the data folder.
> For delays, please search "NWR Historic Delay Attribution". Under "data files", you will find .zip files named, for example "202324.zip" for a complete set of one year data. Once you extract all, you will find data files named as :
   - `Transparency_23-24_P12.csv`
   - `Transparency_23-24_P13.csv`
   - `Transparency_24-25_P01.csv`
   - ...

"Transparency" refers to the initiative by the Rail Delivery Group (RDG) and train operators in Great Britain to publish Key Transparency Indicators (KTIs), that is, publicly available operational and performance data.
"23-24" stands for the year and "P01" is the month in which the data is located. These align with financial years, and therefore begin in the month of April. You could also find .zip files named "202425 data files 20250213.zip" or "Transparency 25-26 P01 20250516.zip", here, the date at the end of the name indicates the last entry in the data itself. 

> For SWR passenger loadings, please look up "SWR Passenger Loadings". Under "data files", you will find .xlsx files named:
   - `SWR Passenger Loadings - RY25 P10.xlsx`
   - `SWR Passenger Loadings - RY25 P11.xlsx`
   - ...

Here, "RY25" stands for the rail year that the data covers, with "P10" being the month within that year.

> For full schedules, please look up "NWR Schedule". Under "data files" you will find:
   - `CIF_ALL_FULL_DAILY_toc-full.json.gz`
   
This file contains "toc-full" which stands for Train Operating Companies (TOC) as a Full Extract in daily formats. The full extend of the data is weekly, meaning it contains all daily scheduled trains for a standard week in the year.

3. Inside the incidents.py, passenger_loadings.py and schedule.py you will find specifications for each file and how to modify their entries depending on the rail month or year.
4. The tool will automatically detect and load these files from the `data/` folder.
5. Please refer to the `reference/` folder for the only directly provided files, these include station reference files with latitude and longitude and description-related information.
6. **IMPORTANT** Here, the schedule file needs to be cleaned before it is pre-processed. To do so, please:
 - Run the `data/schedule_cleaning.py` file, where the function clean_schedule is present.
 - This will create the CIF_ALL_FULL_DAILY_toc-full_p4.pkl file
 - This is the cleaned version of the downloaded schedule file in .json.gz format. This data file has the suffix "p4" from it being the 4th section in the original schedule file.   
    The file is a newline-delimited JSON (NDJSON) file containing 5 types:
    1. JsonTimetableV1 - Header/metadata
    2. TiplocV1 - Location codes
    3. JsonAssociationV1 - Train associations
    4. JsonScheduleV1 - Schedule data (THIS IS WHAT WE EXTRACT)
    5. EOF - End of file marker
 The schedule.py file already contains the correct code for this cleaned file to be called properly in the following sections.

## Data Pre-Processing

After you have downloaded this data and saved it to the `data/` folder, you need to perform some pre-processing. This is a crucial step in this analysis as you want to match the scheduled trains with delays and passenger loadings. The script processes schedule data, applies delays, and saves the results as pandas DataFrames organized by day of the week for each station. Please note, that as of 11th November 2025, this script takes 1 full day to pre-process all the stations. To pre-process the data, you need to run:

> python -m preprocess.preprocessor

This can be run with different specifications for the user's needs. Below are defined all its possible usages:

1. To process All categories: python -m preprocess.preprocessor --all-categories
2. To process a single station: python -m preprocess.preprocessor <STANOX_CODE>
3. To process Category A stations only : python -m preprocess.preprocessor --category-A
4. To process Category B stations only: python -m preprocess.preprocessor --category-B
5. To process Category C1 stations only: python -m preprocess.preprocessor --category-C1
6. To process Category C2 stations only: python -m preprocess.preprocessor --category-C2

This script saves processed schedule and delay data to parquet files for railway stations by DFT category in a `processed_data/` folder.

## Requirements

To run the repository, you need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

## Tool Demos and Outputs

After you have downloaded and saved the raw data, and pre-processed it using the preprocessor tool, you can make use of the demos for the actual network analysis. To do so, you need to load the data you have just pre-processed. In the `outputs/` folder you can find two files:

- load_data.py
- utils.py

The load_data.py is a script that defines the function load_processed_data which is called at the start of every demo in the `demos/` folder. In the same way, the utils.py is a script that contains all needed functions that make the demos for this analysis possible. These functions will also be called at the beginning of every demo, according to their respective usage. 
In the `demos/` folder, you can find all 5 demos defined by this analysis. These demos are:

1. Aggregate View
2. Incident View
3. Time View
4. Train View
5. Station View

Each demo is concerned with a different aspect of network analysis and granularity of inspection.

## Testing

This project includes a comprehensive test suite to ensure code quality and reliability.

**82 tests** covering the preprocessor module functions (79% coverage).

### Run Tests
```bash
# Run all tests
pytest

# Run with details
pytest -v

# Run specific test file
pytest tests/test_preprocessor_utils.py
```

**All tests are passing ✅**

For detailed testing documentation, see [tests/README.md](tests/README.md).

### VS Code Integration
- Click the **Testing** icon (beaker) in the sidebar
- If tests don't appear: Reload window (`Ctrl+Shift+P` → "Reload Window")
- Click refresh button in Testing panel

**Tip:** If VS Code Test Explorer has issues, use the terminal - it always works!


