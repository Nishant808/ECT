"""
Sequence Fetcher: retrieves viral genomic sequences from public databases.

Supports NCBI Entrez (GenBank/RefSeq) and local FASTA files.
"""

import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Iterator
from dataclasses import dataclass, field

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


@dataclass
class SequenceRecord:
    """Normalised representation of a single viral sequence."""
    accession: str
    sequence: str
    collection_date: Optional[str]
    location: Optional[str]
    host: Optional[str]
    length: int = field(init=False)

    def __post_init__(self):
        self.length = len(self.sequence)


class NCBISequenceFetcher:
    """
    Fetch viral sequences from NCBI Entrez.

    Parameters
    ----------
    email : str
        Required by NCBI for API access.
    api_key : str, optional
        NCBI API key (increases rate limit from 3 to 10 req/s).
    batch_size : int
        Number of records to fetch per request.
    """

    def __init__(self, email: str, api_key: Optional[str] = None, batch_size: int = 200):
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, db: str = "nucleotide", max_records: int = 1000) -> List[str]:
        """
        Search NCBI and return a list of accession IDs.

        Parameters
        ----------
        query : str
            Entrez search query, e.g. ``"SARS-CoV-2[Organism] AND 2020[PDAT]"``.
        db : str
            NCBI database name.
        max_records : int
            Maximum number of IDs to return.

        Returns
        -------
        list of str
            Accession IDs.
        """
        logger.info("Searching NCBI %s: %s (max=%d)", db, query, max_records)
        handle = Entrez.esearch(db=db, term=query, retmax=max_records)
        record = Entrez.read(handle)
        handle.close()
        ids = record["IdList"]
        logger.info("Found %d records", len(ids))
        return ids

    def fetch(self, accession_ids: List[str], db: str = "nucleotide") -> List[SequenceRecord]:
        """
        Fetch full sequence records for a list of accession IDs.

        Parameters
        ----------
        accession_ids : list of str
        db : str

        Returns
        -------
        list of SequenceRecord
        """
        records: List[SequenceRecord] = []
        for batch_ids in self._batched(accession_ids):
            batch_records = self._fetch_batch(batch_ids, db)
            records.extend(batch_records)
            time.sleep(0.34)  # Respect NCBI rate limit (3 req/s without API key)
        logger.info("Fetched %d sequence records", len(records))
        return records

    def to_dataframe(self, records: List[SequenceRecord]) -> pd.DataFrame:
        """Convert a list of SequenceRecord objects to a DataFrame."""
        return pd.DataFrame([vars(r) for r in records])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_batch(self, ids: List[str], db: str) -> List[SequenceRecord]:
        id_str = ",".join(ids)
        handle = Entrez.efetch(db=db, id=id_str, rettype="gb", retmode="text")
        parsed: List[SequenceRecord] = []
        for bio_record in SeqIO.parse(handle, "genbank"):
            parsed.append(self._parse_genbank(bio_record))
        handle.close()
        return parsed

    @staticmethod
    def _parse_genbank(record: SeqRecord) -> SequenceRecord:
        features = {f.type: f for f in record.features}
        source = features.get("source")

        collection_date = None
        location = None
        host = None

        if source:
            qualifiers = source.qualifiers
            collection_date = qualifiers.get("collection_date", [None])[0]
            location = qualifiers.get("country", [None])[0]
            host = qualifiers.get("host", [None])[0]

        return SequenceRecord(
            accession=record.id,
            sequence=str(record.seq).upper(),
            collection_date=collection_date,
            location=location,
            host=host,
        )

    def _batched(self, ids: List[str]) -> Iterator[List[str]]:
        for i in range(0, len(ids), self.batch_size):
            yield ids[i : i + self.batch_size]


class FASTASequenceFetcher:
    """
    Load sequences from a local FASTA file.

    The FASTA header is expected to contain pipe-separated fields:
    ``>accession|date|location|host``
    Missing fields are set to ``None``.
    """

    def load(self, fasta_path: str) -> List[SequenceRecord]:
        """
        Parse a FASTA file and return SequenceRecord objects.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file.

        Returns
        -------
        list of SequenceRecord
        """
        path = Path(fasta_path)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        records: List[SequenceRecord] = []
        for bio_record in SeqIO.parse(str(path), "fasta"):
            parts = bio_record.id.split("|")
            accession = parts[0] if len(parts) > 0 else bio_record.id
            date = parts[1] if len(parts) > 1 else None
            location = parts[2] if len(parts) > 2 else None
            host = parts[3] if len(parts) > 3 else None

            records.append(
                SequenceRecord(
                    accession=accession,
                    sequence=str(bio_record.seq).upper(),
                    collection_date=date,
                    location=location,
                    host=host,
                )
            )

        logger.info("Loaded %d sequences from %s", len(records), fasta_path)
        return records

    def to_dataframe(self, records: List[SequenceRecord]) -> pd.DataFrame:
        return pd.DataFrame([vars(r) for r in records])
