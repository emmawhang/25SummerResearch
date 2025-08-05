from Bio import Entrez

Entrez.email = "whang@purdue.edu"

def fetch_pubmed_abstracts(query, max_results=100):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    id_list = record["IdList"]

    abstracts = []
    for pmid in id_list:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract = handle.read().strip()
        abstracts.append(abstract)
        handle.close()
    return abstracts