This repository stores projects related to the MSci in Research program within the Infrastructure & Environment division at the University of Glasgow. It will contain shared files with the professor supervising my project, Dr. Ji-Eun Byun, to promote collaboration between our ends and combine our efforts.
All files are considered as input in their form of download. The sources of these downloads are listed below.

## Data Setup

To use this tool, you need to download the NWR Historic Delay Attribution data from Network Rail, SWR Passenger Loadings and NWR Schedule data (or your organisation's portal). You can access these files from the Rail Data Marketplace (RDM) platform here: https://raildata.org.uk/. 
>  The data is not included in this repository due to licensing and size restrictions.

1. Create a `data/` folder in the project root if it doesn't exist.
2. Download the following files and save them in `data/`. Please do not create separate folders within the data folder.
   For delays:
   - `Transparency_23-24_P12.csv`
   - `Transparency_23-24_P13.csv`
   - `Transparency_24-25_P01.csv`
   - ...
   For SWR passenger loadings:
   - `SWR Passenger Loadings - RY25 P10.xlsx`
   - `SWR Passenger Loadings - RY25 P11.xlsx`
   - ...
   For full schedules:
   - `CIF_ALL_FULL_DAILY_toc-full.json.gz`
3. The tool will automatically detect and load these files from the `data/` folder.
4. Please refer to the `reference/` and `track lines/` folders for the only directly provided files.
