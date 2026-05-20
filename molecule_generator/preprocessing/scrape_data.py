import csv
import logging
from typing import List
import requests


class PubChemSmilesScraper:
    """
    Scrapes SMILES strings from PubChem using the PUG REST API.

    Documentation here
    https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial

    Parameters
    ----------
    chunk_size : int
        Number of molecule IDs per API request. keep chunk small to avoidtimeout
    num_chunks : int
        Number of chunks to retrieve.
    """

    BASE_URL = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        "{}/property/CanonicalSMILES/CSV"
    )

    def __init__(self, chunk_size: int = 200, num_chunks: int = 10):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.total_molecules = chunk_size * num_chunks

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _build_url(self, start_cid: int) -> str:
        """
        Construct API request URL for a chunk of CIDs.
        """
        cid_range = ",".join(
            str(start_cid + i) for i in range(self.chunk_size)
        )
        return self.BASE_URL.format(cid_range)

    def _fetch_chunk(self, start_cid: int):

        cid_list = ",".join(
            str(start_cid + i)
            for i in range(self.chunk_size)
        )

        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
            "compound/cid/property/CanonicalSMILES/CSV"
        )

        try:
            response = requests.post(
                url,
                data={"cid": cid_list},
                headers={
                    "Content-Type":
                    "application/x-www-form-urlencoded"
                },
                timeout=60
            )

            response.raise_for_status()

        except requests.RequestException as e:
            self.logger.error(
                f"Request failed at CID {start_cid}: {e}"
            )
            return []

        rows = response.text.strip().split("\n")[1:]

        smiles = [
            row[1]
            for row in csv.reader(rows)
            if len(row) > 1
        ]

        return smiles

    def get_data(self) -> List[str]:
        """
        Retrieve SMILES strings across all chunks.
        """
        all_smiles = []

        for chunk_index in range(self.num_chunks):

            start_cid = chunk_index * self.chunk_size + 1

            self.logger.info(
                f"Fetching chunk {chunk_index + 1}/{self.num_chunks} "
                f"(CID {start_cid} → {start_cid + self.chunk_size - 1})"
            )

            smiles_chunk = self._fetch_chunk(start_cid)

            all_smiles.extend(smiles_chunk)

        return all_smiles

    def write_data(self, output_file: str = "pubchem_smiles.csv") -> None:
        """
        Retrieve SMILES data and write to CSV file.

        Parameters
        ----------
        output_file : str
            Output CSV filename.
        """

        smiles_data = self.get_data()

        if not smiles_data:
            self.logger.warning("No SMILES data retrieved.")
            return

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["CanonicalSMILES"])

            for smiles in smiles_data:
                writer.writerow([smiles])

        self.logger.info(
            f"Saved {len(smiles_data)} SMILES strings to {output_file}"
        )