from retriever.retriever import MedicalRetriever


retriever = MedicalRetriever()
# Test the same query with and without UMLS
query = "What are the early signs of a myocardial infarction?"

# Variant A: Standard
results_base = retriever.retrieve(query, use_ontology=False)

# Variant B: Enhanced
results_umls = retriever.retrieve(query, use_ontology=True)

# Comparison logic: Did the UMLS version bring more relevant chunks to the top?