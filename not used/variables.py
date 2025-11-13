# this file is meant to define two main variables:
# 1. `selected_stanox`: A list of STANOX codes that are selected for analysis.
# 2. `platform_counts`: A dictionary mapping STANOX codes to their respective platform counts.

from collections import defaultdict
from collections import Counter

# Manually define platform counts for each station (STANOX code) https://www.railwaydata.co.uk/
platform_counts = {
    "72410": 16,
    "42140": 12, 
    "70261": 6,
    "32000": 14,  
    "72313": 2,
    "70030": 6, 
    "70411": 5,
    "30120": 8,
    "40320": 7,
    "70306": 8,
    "43314": 5,
    "70100": 5,
    "65630": 12,
    "71040": 11,
    "72007": 6,
    "36151": 10,
    "65043": 6,
    "33088": 2,
    "51009": 2,
    "72416": 4,
    "6347": 1,
    "69202": 4,
    "40320": 7,
    "35531": 4,
    "30021": 6,
}

# Group STANOX codes by platform counts
platform_groups = defaultdict(list)
for stanox, platform_count in platform_counts.items():
    platform_groups[platform_count].append(stanox)

# Distribute the selection across platform groups
benchmark_stanox = []
total_to_select = 20
for platform_count, stanox_list in platform_groups.items():
    # Calculate the number of stations to select from this group
    proportion = len(stanox_list) / len(platform_counts)
    num_to_select = max(1, round(proportion * total_to_select))  # Ensure at least one is selected
    benchmark_stanox.extend(stanox_list[:num_to_select])  # Select from the group

# If more than 20 were selected due to rounding, trim the list
benchmark_stanox = benchmark_stanox[:total_to_select]
benchmark_stanox_platform_counts = {stanox: platform_counts[stanox] for stanox in benchmark_stanox}

# print(f"Selected STANOX codes for analysis: {benchmark_stanox}")
# print(f"Platform counts for selected STANOX codes: {benchmark_stanox_platform_counts}")