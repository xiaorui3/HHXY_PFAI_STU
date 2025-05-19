import json
import os
from tqdm import tqdm

file_path = 'proteins1.json'

file_size = os.path.getsize(file_path)

chunk_size = 102400000*4

with tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading file") as pbar:

    with open(file_path, 'r') as file:
        file_content = ""
        while True:

            chunk = file.read(chunk_size)
            if not chunk:
                break
            file_content += chunk
            pbar.update(len(chunk.encode('utf-8')))

data = json.loads(file_content)

processed_data = []

for index, item in tqdm(enumerate(data), total=len(data), desc="Processing proteins"):
    literature_text = ""
    if 'literature' in item:
        if isinstance(item['literature'], dict):
            for lit_index, lit in item['literature'].items():
                literature_text += f"- Literature {lit_index}: Title \"{lit.get('name', 'Unknown')}\", Abstract \"{lit.get('abstract', 'Unknown')}\"\n"
        elif isinstance(item['literature'], str):
            literature_text = f"Literature information: {item['literature']}\n"
        else:
            literature_text = "Incorrect literature information format.\n"
    else:
        literature_text = "No related literature provided.\n"

    document = f"""The name of this protein is {item.get('Protein names', 'Unknown')}, with the entry number {item.get('Entry', 'Unknown')} in the database. This protein consists of {item.get('Length', 'Unknown')} amino acids, and the specific amino acid sequence is {item.get('Sequence', 'Unknown')}. Its main function is {item.get('Function', 'Unknown')}, and it originates from the organism {item.get('Organism', 'Unknown')}. Structurally, it is represented as {item.get('Subunit structure', 'Unknown')}.

Tissue specificity: {item.get('Tissue specificity', 'Not provided')}. Involvement in disease: {item.get('Involvement in disease', 'Not provided')}. Post-translational modification: {item.get('Post-translational modification', 'Not provided')}.

The protein domain is described as {item.get('Domain[CC]', 'Unknown')}, located at {item.get('Domain[FT]', 'Unknown')}. Expression induction factors are {item.get('Induction', 'Unknown')}. Related literature DOI is {item.get('DOI ID', 'Unknown')}, and the PubMed ID is {item.get('PubMed ID', 'Unknown')}. This entry was created on {item.get('Date of creation', 'Unknown')}.

Related literature includes:
{literature_text}

The catalytic activity of this protein is described as {item.get('Catalytic activity', 'Unknown')}, and the gene encoding it is {item.get('Gene Names', 'Unknown')}."""

    processed_data.append({"text": document})

processed_json = json.dumps(processed_data, indent=2)


with open('processed_data.json', 'w') as outfile:
    json.dump(processed_data, outfile, indent=4)