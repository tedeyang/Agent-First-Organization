import os
import argparse
import pickle
from pathlib import Path
import logging
import zipfile
import tempfile

from arklex.utils.loader import Loader

logger = logging.getLogger(__name__)


def build_rag(folder_path, rag_docs):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filepath = os.path.join(folder_path, "documents.pkl")
    loader = Loader()
    docs = []
    if Path(filepath).exists():
        logger.warning(f"Loading existing documents from {os.path.join(folder_path, 'documents.pkl')}! If you want to recrawl, please delete the file or specify a new --output-dir when initiate Generator.")
        docs = pickle.load(open(os.path.join(folder_path, "documents.pkl"), "rb"))
    else:
        for doc in rag_docs:
            source = doc.get("source")
            logging.info(f"Crawling {source}")
            num_docs = doc.get("num") if doc.get("num") else 1
            if doc.get('type') == 'url':
                    num_docs = doc.get("num") if doc.get("num") else 1
                    urls = loader.get_all_urls(source, num_docs)
                    crawled_urls = loader.to_crawled_url_objs(urls)
                    docs.extend(crawled_urls)
                    
            elif doc.get('type') == 'file':
                # check if the source is a file or a directory
                if os.path.isfile(source):
                    if source.lower().endswith('.zip'):
                        # Handle zip file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            with zipfile.ZipFile(source, 'r') as zip_ref:
                                zip_ref.extractall(temp_dir)
                            # Process all files in the extracted directory
                            file_list = []
                            for root, _, files in os.walk(temp_dir):
                                for file in files:
                                    file_list.append(os.path.join(root, file))
                            docs.extend(loader.to_crawled_local_objs(file_list))
                    else:
                        docs.extend(loader.to_crawled_local_objs([source]))
                elif os.path.isdir(source):
                    file_list = [os.path.join(source, f) for f in os.listdir(source)]
                    docs.extend(loader.to_crawled_local_objs(file_list))
                else:
                    raise FileNotFoundError(f"Source path '{source}' does not exist or is not accessible")

            elif doc.get('type') == 'text':
                docs.extend(loader.to_crawled_text([source]))
            else:
                # TODO: how to handle when type is not provided
                raise Exception("type must be one of [url, file, text] and it must be provided")

        logging.info(f"Content: {[doc.content for doc in docs]}")
        Loader.save(filepath, docs)

    logging.info(f"crawled sources: {[c.source for c in docs]}")
    chunked_docs = Loader.chunk(docs)
    filepath_chunk = os.path.join(folder_path, "chunked_documents.pkl")
    Loader.save(filepath_chunk, chunked_docs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_url", required=True, type=str, help="base url to crawl")
    parser.add_argument("--folder_path", required=True, type=str, help="location to save the documents")
    parser.add_argument("--max_num", type=int, default=10, help="maximum number of urls to crawl")
    args = parser.parse_args()

    build_rag(folder_path=args.folder_path, docs=[{"source": args.base_url, "num": args.max_num}])