This repository stores projects related to the MSci in Research program within the Infrastructure & Environment division at the University of Glasgow. It will contain shared files with the professor supervising my project, Dr. Ji-Eun Byun, to promote collaboration between our ends and combine our efforts.
All files are considered as input in their form of download. The sources of these downloads are listed below.

## Data Setup

To use this tool, you need to download the NWR Historic Delay Attribution data from Network Rail, SWR Passenger Loadings and NWR Schedule data (or your organisation's portal). You can access these files from the Rail Data Marketplace (RDM) platform here: https://raildata.org.uk/. 
>  The data is not included in this repository due to licensing and size restrictions.

1. Create a `data/` folder in the project root if it doesn't exist.
2. Download the following files and save them in `data/`. Please do not create separate folders within the data folder.
> For delays:
   - `Transparency_23-24_P12.csv`
   - `Transparency_23-24_P13.csv`
   - `Transparency_24-25_P01.csv`
   - ...
> For SWR passenger loadings:
   - `SWR Passenger Loadings - RY25 P10.xlsx`
   - `SWR Passenger Loadings - RY25 P11.xlsx`
   - ...
> For full schedules:
   - `CIF_ALL_FULL_DAILY_toc-full.json.gz`
3. The tool will automatically detect and load these files from the `data/` folder.
4. Please refer to the `reference/` folder for the only directly provided files, these include station reference files with latitude and longitude and description-related information.
5. IMPORTANT! Here, the schedule file needs to be cleaned before it is pre-processed. To do so, please:
 - Run the schedule_cleaning.py file, where the function clean_schedule is      present.
 - This will create the CIF_ALL_FULL_DAILY_toc-full_p4.pkl file
 - This is the cleaned version of the downloaded schedule file in .json.gz format. The schedule.py file already contains the correct code for this cleaned file to be called properly in the following sections.

## Data Pre-Processing

After you have downloaded this data and saved it to the `data/` folder, you need to perform some pre-processing. This is a crucial step in this analysis as you want to match the scheduled trains with delays and passenger loadings. The script rocesses schedule data, applies delays, and saves the results as pandas DataFrames organized by day of the week for each station. Please note, that as of 11th November 2025, this script takes 1 full day to pre-process all the stations. To pre-process the data, you need to run:

> python -m preprocessor.main

This can be run with different specifications for the user's needs. Below are defined all its possible usages:

1. To process All categories: python -m preprocessor.main --all-categories
2. To process a single station: python -m preprocessor.main <STANOX_CODE>
3. To process Category A stations only : python -m preprocessor.main --category-A
4. To process Category B stations only: python -m preprocessor.main --category-B
5. To process Category C1 stations only: python -m preprocessor.main --category-C1
6. To process Category C2 stations only: python -m preprocessor.main --category-C2

This script saves processed schedule and delay data to parquet files for railway stations by DFT category in a `processed_data/` folder.

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
