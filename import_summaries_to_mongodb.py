import os
from dotenv import load_dotenv
import csv
import math
import re
from pathlib import Path
from pymongo import MongoClient, UpdateOne

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "visualizations"
MONGO_URI = os.environ["MONGODB_URI"]
DATABASE_NAME = "agriculture_telemetry"

# Stable keys used to identify each summary record.
KEY_CANDIDATES = {
    "crop_yield": [["crop"]],
    "geographic_summary": [["location", "crop"]],
    "irrigation_summary": [
        ["irrigation_status", "irrigation_method"],
        ["status", "method"],
        ["irrigation_status", "irrigation_type"],
    ],
    "temperature_summary": [["temperature_range"]],
    "soil_summary": [["crop", "soil_ph_range"], ["crop", "ph_range"]],
    "location_crop_soil": [["location", "crop", "soil_ph_range"]],
}


def normalize_header(value):
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def convert_value(value):
    value = value.strip()
    if value == "":
        return None

    try:
        number = float(value)
        if math.isfinite(number):
            return int(number) if number.is_integer() else number
        return value
    except ValueError:
        return value


def find_csv(folder):
    files = sorted(folder.glob("*.csv"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one CSV in {folder}, found {len(files)}"
        )
    return files[0]


def import_collection(db, folder_name, collection_name):
    folder = CSV_DIR / folder_name
    csv_file = find_csv(folder)

    with csv_file.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise RuntimeError(f"No CSV headers found in {csv_file}")

        headers = {
            original: normalize_header(original)
            for original in reader.fieldnames
        }
        rows = []
        for raw in reader:
            doc = {
                headers[key]: convert_value(value or "")
                for key, value in raw.items()
                if key is not None
            }
            if doc:
                rows.append(doc)

    if not rows:
        print(f"SKIPPED {collection_name}: no data rows")
        return

    candidates = KEY_CANDIDATES[collection_name]
    keys = next(
        (candidate for candidate in candidates
         if all(key in rows[0] for key in candidate)),
        None
    )

    # If the expected key names differ, use the entire row as the filter.
    # Identical rows will update rather than multiply on repeat imports.
    if keys is None:
        keys = list(rows[0].keys())
        print(f"NOTE {collection_name}: using all columns as the record key")

    operations = []
    for doc in rows:
        identity = {key: doc[key] for key in keys}
        operations.append(
            UpdateOne(identity, {"$set": doc}, upsert=True)
        )

    result = db[collection_name].bulk_write(operations, ordered=False)
    print(
        f"IMPORTED {collection_name}: CSV rows={len(rows)}, "
        f"upserted={result.upserted_count}, "
        f"updated={result.modified_count}, "
        f"matched={result.matched_count}"
    )
    print(f"  File: {csv_file.name}")
    print(f"  Record key: {keys}")


def main():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DATABASE_NAME]
        print(f"Connected to MongoDB database: {DATABASE_NAME}")

        datasets = [
            ("crop_yield", "crop_yield"),
            ("geographic", "geographic_summary"),
            ("irrigation", "irrigation_summary"),
            ("temperature", "temperature_summary"),
            ("soil", "soil_summary"),
            ("location_crop_soil", "location_crop_soil"),
        ]

        for folder, collection in datasets:
            import_collection(db, folder, collection)

        print("\nFinal MongoDB collection counts:")
        for _, collection in datasets:
            print(f"  {collection}: {db[collection].count_documents({})}")

        print("\nImport completed.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
