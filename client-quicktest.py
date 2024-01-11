import pinecone
import numpy as np
import os
import argparse
import traceback

from dotenv import load_dotenv

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENVIRONMENT = os.environ['PINECONE_ENVIRONMENT']
INDEX_NAME = "quicktest"

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

def version():
    print(f"Pinecone version(): {pinecone.version()}")

def list():
    print(f"Pinecone list_indexes(): {pinecone.list_indexes()}")

def create():
    pinecone.create_index(INDEX_NAME, dimension=128, metric="cosine", pods=1, replicas=1, pod_type="p2.x4")
    print(f"Pinecone describe_index(): {pinecone.describe_index(INDEX_NAME)}")
    
def insert():
    index = pinecone.Index(INDEX_NAME)
    test_vector = np.random.uniform(-1, 1, size=128).tolist()
    upsert_response = index.upsert(vectors=[{'id':'1', 'values':test_vector}])
    print(f"Pinecone upsert() response: {upsert_response}")

def query():
    index = pinecone.Index(INDEX_NAME)
    test_vector = np.random.uniform(-1, 1, size=128).tolist()
    query_response = index.query(vector=test_vector, top_k=10, include_values=True, include_metadata=True)
    print(f"Pinecone query() response: {query_response}")

def delete():
    pinecone.delete_index(INDEX_NAME)
    print(f"Pinecone list_indexes after delete: {pinecone.list_indexes()}")

def main(args):
    if args.all_commands:
        try:
            version()
            list()
            create()
            insert()
            query()
            delete()
        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python client quicktest.")
    parser.add_argument('-all', '--all_commands', action='store_true', help='Run all pinecone commands.')
    args = parser.parse_args()
    
    main(args)
