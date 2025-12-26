import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader



def load_documents_from_folder(folder_path):
    #charger tous les documents d'un dossier soit txt ou pdf
    documents=[]
    for filename in os.listdir(folder_path):
        filepath=os.path.join(folder_path,filename)
        try:
            if filename.endswith('.pdf'):
                loader=PyPDFLoader(filepath)
                documents.extend(loader.load())
                print("fichier .pdf loader avec succèes")
            elif filename.endswith('.txt'):
                loader=TextLoader(filepath,encoding='utf-8')
                documents.extend(loader.load())
                print("fichier .txt loader avec succèes")
        except Exception as e:
            print(f'Erreur lors du chargement de {filename}: {e}')
            return 
    return documents
            

if __name__=="__main__":
    project_root=Path(__file__).parent.parent
    docs_path=project_root/"docs"

    print("chargement des documents ...")
    documents=load_documents_from_folder(docs_path)
    print(f"{len(documents)} documents chargés avec succèes")
        
