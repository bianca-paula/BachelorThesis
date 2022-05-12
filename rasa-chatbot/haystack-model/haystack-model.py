import wikipedia
from pprint import pprint
from haystack.nodes import FARMReader
from haystack.utils import convert_files_to_dicts,clean_wiki_text
from haystack.pipelines import ExtractiveQAPipeline
import re
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores.faiss import FAISSDocumentStore

import warnings
warnings.filterwarnings("ignore")

def get_wikipedia_articles(data_dir, topics=[]):
    for topic in topics:
        article = wikipedia.page(pageid=topic).content
        with open(f"{data_dir}/{topic}.txt", "w", encoding="utf-8") as f:
            f.write(article)

document_store = FAISSDocumentStore(faiss_index_factory_str="Flat",sql_url= "sqlite:///../faiss_document_store.db")
topics = [36808, 57330, 3997, 512662, 625404, 249930, 60575]
data_dir= "../actions/data"
get_wikipedia_articles(data_dir, topics=topics)
docs = convert_files_to_dicts(dir_path=data_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.write_documents(docs)
retriever = EmbeddingRetriever(
    document_store=document_store,
   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
   model_format="sentence_transformers"
)
document_store.update_embeddings(retriever)
document_store.save("../testfile_path.index")
