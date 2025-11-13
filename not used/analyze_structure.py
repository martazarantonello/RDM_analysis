import gzip
import json
from collections import defaultdict

file_path = r'data/CIF_ALL_FULL_DAILY_toc-full.json.gz'

print("Analyzing the structure of the .gz file...")
print("This is a newline-delimited JSON file (NDJSON) - each line is a JSON object\n")

# Count different types of objects
object_types = defaultdict(int)
total_lines = 0

print("Scanning file to identify all object types...")
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        total_lines += 1
        try:
            obj = json.loads(line)
            # Get the top-level key (e.g., "TiplocV1", "ScheduleV1", etc.)
            if isinstance(obj, dict) and len(obj) > 0:
                key = list(obj.keys())[0]
                object_types[key] += 1
                
                # Print first occurrence of each type
                if object_types[key] == 1:
                    print(f"\nFound new type: {key}")
                    print(f"  Sample data: {json.dumps(obj, indent=2)[:300]}...")
        except json.JSONDecodeError:
            continue
        
        if i % 100000 == 0 and i > 0:
            print(f"  Processed {i:,} lines...")

print(f"\n{'='*60}")
print(f"Total lines in file: {total_lines:,}")
print(f"\nObject type breakdown:")
print(f"{'='*60}")

for obj_type, count in sorted(object_types.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_lines) * 100
    print(f"{obj_type:30s}: {count:10,} ({percentage:5.2f}%)")

print(f"\nTotal distinct object types: {len(object_types)}")

