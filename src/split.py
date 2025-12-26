from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from ingest import load_documents_from_folder

def split_documents(documents):
    #diviser les documents en chunks
    splitter=CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.split_documents(documents)
    return chunks

if __name__=="__main__":
    project_root=Path(__file__).parent.parent
    docs_path=project_root/"docs"

    print("chargement ...")
    documents=load_documents_from_folder(docs_path)
    print(len(documents))
    print("division en chunks en cours ...")
    chunks=split_documents(documents)
    print(f"{len(chunks)} chunks créés avec succèes"  )