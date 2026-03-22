import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def download_data():
    """
    Downloads all available .csv files from a public repository for the years 1968-2024,
    combines them into a single DataFrame, and saves the result as one CSV file at data/01_raw/combined_matches.csv.

    Details:
    - Downloads data from the URL specified by the DATA_URL environment variable.
    - Concatenates yearly data into one combined file.
    - Creates the destination folder if it does not exist.
    - Saves the resulting CSV file in the data/01_raw directory.
    """
    load_dotenv()
    DATA_URL = os.getenv('DATA_URL')
    DESTINATION_PATH = Path("data/01_raw/combined_matches.csv")
    combined_df = pd.DataFrame()

    for year in range(1968, 2025, 1):
        url = DATA_URL + f"{year}.csv"
        df = pd.read_csv(url)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        logger.info(f'{year} year downloaded')
        
    combined_df.to_csv(DESTINATION_PATH)
    logger.info(f'{len(combined_df)} records downloaded to: {DESTINATION_PATH}')

if __name__ == "__main__":
    download_data()