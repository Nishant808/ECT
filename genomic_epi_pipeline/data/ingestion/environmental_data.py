"""
Environmental Data Ingestion.

Loads and harmonises meteorological and demographic data to be paired
with viral sequences by date and location.

Supported sources
-----------------
* Local CSV / Parquet files with columns: date, location, temperature,
  humidity, population_density (and any extras).
* NOAA Climate Data Online (CDO) via their public REST API.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
COL_DATE = "date"
COL_LOCATION = "location"
COL_TEMP = "temperature"
COL_HUMIDITY = "humidity"
COL_POP_DENSITY = "population_density"

REQUIRED_COLUMNS = [COL_DATE, COL_LOCATION, COL_TEMP, COL_HUMIDITY, COL_POP_DENSITY]


# ---------------------------------------------------------------------------
# Local file loader
# ---------------------------------------------------------------------------

class LocalEnvironmentalLoader:
    """
    Load environmental data from a local CSV or Parquet file.

    The file must contain at minimum the columns listed in ``REQUIRED_COLUMNS``.
    Additional columns are preserved.
    """

    def load(self, path: str) -> pd.DataFrame:
        """
        Load and validate an environmental data file.

        Parameters
        ----------
        path : str
            Path to a ``.csv`` or ``.parquet`` file.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with a ``datetime64`` ``date`` column.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Environmental data file not found: {path}")

        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        df = self._validate_and_clean(df)
        logger.info("Loaded %d environmental records from %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Environmental data missing required columns: {missing}")

        df = df.copy()
        df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")
        df = df.dropna(subset=[COL_DATE])

        # Clip physically implausible values
        df[COL_TEMP] = df[COL_TEMP].clip(-90, 60)
        df[COL_HUMIDITY] = df[COL_HUMIDITY].clip(0, 100)
        df[COL_POP_DENSITY] = df[COL_POP_DENSITY].clip(0)

        df = df.sort_values(COL_DATE).reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# NOAA CDO API loader
# ---------------------------------------------------------------------------

class NOAAEnvironmentalLoader:
    """
    Fetch daily climate summaries from the NOAA Climate Data Online API.

    Parameters
    ----------
    token : str
        NOAA CDO API token (free registration at https://www.ncdc.noaa.gov/cdo-web/token).
    """

    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

    def __init__(self, token: str):
        self.headers = {"token": token}

    def fetch(
        self,
        station_ids: List[str],
        start_date: str,
        end_date: str,
        datatypes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Fetch daily climate data for a list of station IDs.

        Parameters
        ----------
        station_ids : list of str
            NOAA station identifiers, e.g. ``["GHCND:USW00094728"]``.
        start_date : str
            ISO date string, e.g. ``"2020-01-01"``.
        end_date : str
            ISO date string, e.g. ``"2021-06-30"``.
        datatypes : list of str, optional
            NOAA datatype codes. Defaults to ``["TMAX", "TMIN", "PRCP"]``.

        Returns
        -------
        pd.DataFrame
        """
        if datatypes is None:
            datatypes = ["TMAX", "TMIN", "PRCP"]

        all_results: List[Dict] = []
        for station_id in station_ids:
            results = self._fetch_station(station_id, start_date, end_date, datatypes)
            all_results.extend(results)

        if not all_results:
            logger.warning("No NOAA data returned for the requested stations/dates.")
            return pd.DataFrame(columns=["date", "station", "datatype", "value"])

        df = pd.DataFrame(all_results)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------
    def _fetch_station(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        datatypes: List[str],
    ) -> List[Dict]:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": ",".join(datatypes),
            "limit": 1000,
            "units": "metric",
        }
        try:
            resp = requests.get(
                f"{self.BASE_URL}/data", headers=self.headers, params=params, timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except requests.RequestException as exc:
            logger.error("NOAA API request failed for station %s: %s", station_id, exc)
            return []


# ---------------------------------------------------------------------------
# Synthetic / mock generator (for testing and demos)
# ---------------------------------------------------------------------------

class SyntheticEnvironmentalGenerator:
    """
    Generate synthetic environmental data for testing.

    Produces realistic seasonal temperature and humidity curves with
    location-specific offsets and Gaussian noise.
    """

    LOCATION_OFFSETS: Dict[str, float] = {
        "USA": 0.0,
        "UK": -5.0,
        "Germany": -3.0,
        "France": 2.0,
        "Italy": 5.0,
        "Canada": -8.0,
        "Australia": 10.0,
        "Brazil": 12.0,
        "India": 15.0,
        "China": 3.0,
    }

    def generate(
        self,
        locations: List[str],
        start_date: str,
        end_date: str,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a synthetic environmental DataFrame.

        Parameters
        ----------
        locations : list of str
        start_date, end_date : str
            ISO date strings.
        freq : str
            Pandas date frequency string (``"D"`` = daily, ``"W"`` = weekly).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Columns: date, location, temperature, humidity, population_density.
        """
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start_date, end_date, freq=freq)
        rows: List[Dict] = []

        for location in locations:
            offset = self.LOCATION_OFFSETS.get(location, 0.0)
            pop_density = rng.uniform(5, 500)

            for date in dates:
                doy = date.day_of_year
                temp = (
                    15.0
                    + 10.0 * np.sin(2 * np.pi * doy / 365)
                    + offset
                    + rng.normal(0, 2.5)
                )
                humidity = float(
                    np.clip(
                        50.0
                        + 20.0 * np.sin(2 * np.pi * doy / 365 + np.pi / 4)
                        + rng.normal(0, 6),
                        5,
                        95,
                    )
                )
                rows.append(
                    {
                        COL_DATE: date,
                        COL_LOCATION: location,
                        COL_TEMP: round(float(temp), 2),
                        COL_HUMIDITY: round(humidity, 2),
                        COL_POP_DENSITY: round(float(pop_density), 1),
                    }
                )

        df = pd.DataFrame(rows).sort_values([COL_DATE, COL_LOCATION]).reset_index(drop=True)
        logger.info(
            "Generated %d synthetic environmental records (%d locations, %d dates)",
            len(df),
            len(locations),
            len(dates),
        )
        return df
