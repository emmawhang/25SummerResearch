from Bio import Entrez
import json
import time

Entrez.email = "whang@purdue.edu"

def fetch_abstracts(query, max_results=100, batch_size=50):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    ids = Entrez.read(handle)["IdList"]
    abstracts = []

    # Fetch in batches
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start:start+batch_size]
        try:
            fetch = Entrez.efetch(db="pubmed", id=",".join(batch_ids), rettype="abstract", retmode="text")
            batch_abstracts = fetch.read().strip().split("\n\n")
            # Filter out empty abstracts
            abstracts.extend([ab for ab in batch_abstracts if ab.strip()])
            fetch.close()
        except Exception as e:
            print(f"Batch fetch failed: {e}")
            time.sleep(1)  # brief pause before retrying next batch
            continue

    return abstracts

def main():
    abstracts = fetch_abstracts("cancer treatment", max_results=500, batch_size=50)
    with open("data/pubmed_abstracts.json", "w") as f:
        json.dump(abstracts, f)

if __name__ == "__main__":
    main()
