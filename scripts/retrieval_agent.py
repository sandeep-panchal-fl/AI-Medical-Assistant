from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class MedicalDataRetrieval:
    
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

    def retrieve_data(self, query: str, k: int = 1):

        """
        Retrieve the top-k most relevant chunks for a query.
        """

        print("____________________________________\n")
        print("=== Retrieving Medical Data ===")
        results = self.vector_store.similarity_search(query, k=k)

        print("=== Medical Data Retrieved ===")
        return results[0].page_content if len(results) == 1 else results


if __name__ == "__main__":

    # Initialize store
    store = MedicalDataRetrieval(
        persist_dir="../medical_chroma_db",
        embedding_model="../embedding_model/all-MiniLM-L6-v2/",  # Local model
        collection="medical_info"
    )

    # Test query
    query = """Patient reports a throbbing headache with sensitivity to light and nausea,
            onset not precisely determined. The headache is characterized as 10/10 in severity and is located
            diffusely throughout the head. Patient reports nausea alongside the headache, with the onset coinciding
            with the headache's initiation. Clinical examination reveals associated symptoms of photophobia and
            significant discomfort. Further evaluation of the patient’s cardiovascular and pulmonary systems is
            warranted given the acute nature of the symptoms."""
    
    results = store.retrieve_data(query)
    
    print("🔍 Top results:")
    for r in results:
        print(f"🧠 Disease: {r.metadata['disease']}")
        print(f"Snippet: {r.page_content}")
