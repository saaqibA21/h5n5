import time
from typing import List, Tuple
from Bio import Entrez, SeqIO
from tqdm import tqdm

def _setup_entrez(email: str, api_key: str):
    if not email:
        raise ValueError("NCBI_EMAIL is required in your .env (or environment).")
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def search_genbank_ids(query: str, retmax: int = 500) -> List[str]:
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

def fetch_fasta_by_ids(ids: List[str]) -> List[Tuple[str, str]]:
    """
    Returns list of (accession, sequence_str).
    """
    seqs = []
    for batch in _chunked(ids, 100):
        handle = Entrez.efetch(
            db="nucleotide",
            id=",".join(batch),
            rettype="fasta",
            retmode="text"
        )
        for rec in SeqIO.parse(handle, "fasta"):
            accession = rec.id.split("|")[0]
            seqs.append((accession, str(rec.seq).upper()))
        handle.close()
        time.sleep(0.34)  # be polite to NCBI
    return seqs

def download_h5n5_and_negative(
    email: str,
    api_key: str,
    segment_keyword: str,
    h5n5_max: int,
    negative_max: int
):
    """
    Binary classification:
      +1 = H5N5
      0  = other Influenza A (NOT H5N5), but still H5Nx to make it challenging/realistic.
    """
    _setup_entrez(email, api_key)

    # H5N5 HA sequences
    q_pos = (
        f'("Influenza A virus"[Organism]) AND (H5N5[All Fields]) '
        f'AND ({segment_keyword}[Title])'
    )

    # Negatives: H5 but NOT N5 (e.g., H5N1/H5N6/H5N8...)
    q_neg = (
    f'("Influenza A virus"[Organism]) '
    f'AND (H5N1[All Fields] OR H5N6[All Fields] OR H5N8[All Fields])'
)



    pos_ids = search_genbank_ids(q_pos, retmax=h5n5_max)
    neg_ids = search_genbank_ids(q_neg, retmax=negative_max)

    pos = fetch_fasta_by_ids(pos_ids)
    neg = fetch_fasta_by_ids(neg_ids)

    return pos, neg
