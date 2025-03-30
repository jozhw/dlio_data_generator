import csv
from pathlib import Path
from typing import Dict, List


class Saving:

    @staticmethod
    def save_to_csv(results: List[Dict], filepath: Path, chunk_size: int = 10000):

        # get key names of the dict to be field names
        fieldnames: List = list(results[0].keys())
        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for i in range(0, len(results), chunk_size):
                chunk = results[i : i + chunk_size]
                writer.writerows(chunk)

        print("Results have been saved to: {}".format(filepath))
