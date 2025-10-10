import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

class MedicalDataStore:
    
    def __init__(self,
                 persist_dir: str = "../medical_chroma_db",
                 embedding_model: str = "../embedding_model/all-MiniLM-L6-v2/",
                 collection: str = "medical_info"
                 ) -> None:

        """
        Initialize the MedicareDataStore with embedding and Chroma settings.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vector_store = Chroma(
            collection_name=collection,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def store_in_vectordb(self, csv_path: str):

        """
        Load CSV, convert to LangChain Document format, chunk, and add to Chroma vector DB.
        """

        # Load the data
        df = pd.read_csv(csv_path)

        # Ensure the columns exist
        if 'disease' not in df.columns or 'combined_text' not in df.columns:
            raise ValueError("CSV must contain 'disease' and 'combined_text' columns.")

        # Convert to LangChain Document objects
        documents = []
        for _, row in df.iterrows():
            metadata = {"disease": row["disease"]}
            doc = Document(page_content=row["combined_text"], metadata=metadata)
            documents.append(doc)

        print(f"✅ Loaded {len(documents)} disease records from {csv_path}")

        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        print(f"🧩 Created {len(chunks)} text chunks for embedding.")

        # Add to vector DB
        self.vector_store.add_documents(chunks)
        # self.vector_store.persist()
        # print(f"💾 Data persisted to {self.vector_store._persist_directory}")

    def similarity_search(self, query: str, k: int = 3):
        """
        Retrieve the top-k most relevant chunks for a query.
        """
        results = self.vector_store.similarity_search(query, k=k)
        # print("🔍 Top results:")
        # for r in results:
        #     print(f"🧠 Disease: {r.metadata['disease']}")
        #     print(f"Snippet: {r.page_content[:200]}...\n")
        return results


if __name__ == "__main__":

    # Initialize store
    store = MedicalDataStore(
        persist_dir="../medical_chroma_db",
        embedding_model="../embedding_model/all-MiniLM-L6-v2/",  # Local model
        collection="medical_info"
    )

    # Load and embed the data
    store.store_in_vectordb("../data/symp_scan_rag.csv")

    # Test query
    query = """Patient reports a throbbing headache with sensitivity to light and nausea,
            onset not precisely determined. The headache is characterized as 10/10 in severity and is located
            diffusely throughout the head. Patient reports nausea alongside the headache, with the onset coinciding
            with the headache's initiation. Clinical examination reveals associated symptoms of photophobia and
            significant discomfort. Further evaluation of the patient’s cardiovascular and pulmonary systems is
            warranted given the acute nature of the symptoms."""
    
    results = store.similarity_search(query)
    
    print("🔍 Top results:")
    for r in results:
        print(f"🧠 Disease: {r.metadata['disease']}")
        print(f"Snippet: {r.page_content[:200]}...\n")
