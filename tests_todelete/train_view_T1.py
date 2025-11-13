# needed this function in train_view_T2.ipynb

import pandas as pd

def get_stanox_for_service(all_data, train_service_code, origin_code, destination_code):
    """
    For a given train service and OD pair, find the first train of the day (by earliest PLANNED_ORIGIN_GBTT_DATETIME),
    then order the unique STANOX codes for that train:
    - Use PLANNED_ORIGIN_GBTT_DATETIME for the origin,
    - PLANNED_DEST_GBTT_DATETIME for the destination,
    - PLANNED_CALLS for all other stations.
    Return the ordered list of STANOX codes for that train only, always including the origin as the first element.
    """

    # --- Ensure OD_PAIR exists ---
    if 'OD_PAIR' not in all_data.columns:
        all_data['OD_PAIR'] = (
            all_data['PLANNED_ORIGIN_LOCATION_CODE'].astype(str)
            + '_' +
            all_data['PLANNED_DEST_LOCATION_CODE'].astype(str)
        )

    od_pair = f"{origin_code}_{destination_code}"

    # --- Filter dataset for this service and OD pair ---
    subset = all_data[
        (all_data['OD_PAIR'] == od_pair)
        & (all_data['TRAIN_SERVICE_CODE'].astype(str) == str(train_service_code))
    ].copy()

    if subset.empty:
        message = f"🚫 No records found for train service {train_service_code} on OD pair {od_pair}."
        print(message)
        return message

    # --- Find the first train of the day (earliest PLANNED_ORIGIN_GBTT_DATETIME) ---
    subset['PLANNED_ORIGIN_GBTT_DATETIME'] = pd.to_datetime(subset['PLANNED_ORIGIN_GBTT_DATETIME'], format='%H%M', errors='coerce')
    first_origin_time = subset['PLANNED_ORIGIN_GBTT_DATETIME'].min()
    first_train = subset[subset['PLANNED_ORIGIN_GBTT_DATETIME'] == first_origin_time].copy()

    # --- For this train, get all unique STANOX codes and their times ---
    def get_time(row):
        stanox = str(row['STANOX'])
        if stanox == str(origin_code) and pd.notna(row.get('PLANNED_ORIGIN_GBTT_DATETIME', None)):
            return pd.to_datetime(row['PLANNED_ORIGIN_GBTT_DATETIME'], format='%H%M', errors='coerce')
        elif stanox == str(destination_code) and pd.notna(row.get('PLANNED_DEST_GBTT_DATETIME', None)):
            return pd.to_datetime(row['PLANNED_DEST_GBTT_DATETIME'], format='%H%M', errors='coerce')
        elif pd.notna(row.get('PLANNED_CALLS', None)) and str(row['PLANNED_CALLS']).isdigit() and len(str(row['PLANNED_CALLS'])) == 4:
            # Use arbitrary date for time-only values to allow sorting
            return pd.to_datetime(str(row['PLANNED_CALLS']), format='%H%M', errors='coerce')
        else:
            return pd.NaT

    # Keep only unique STANOX for this train, keeping the earliest time for each
    first_train['ORDER_TIME'] = first_train.apply(get_time, axis=1)
    ordered = first_train.groupby('STANOX', as_index=False)['ORDER_TIME'].min()
    ordered = ordered.sort_values('ORDER_TIME')
    stanox_list = ordered['STANOX'].astype(str).tolist()

    # Always include origin_code as the first element, even if not present in the data
    origin_str = str(origin_code)
    if origin_str in stanox_list:
        stanox_list.remove(origin_str)
    stanox_list = [origin_str] + stanox_list

    print(f"✅ Ordered STANOX for first train of the day for service {train_service_code} on OD pair {od_pair} (origin always first):")
    print(stanox_list)
    return stanox_list
