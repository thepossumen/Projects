import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

def calculate_cosine_similarity(embedding1, embedding2):
    # Normalize each vector to unit length and calculate cosine similarity
    return F.cosine_similarity(embedding1, embedding2).item()

def embed_query(query, tokenizer, model):
    """
    Given a query, tokenier and model, creates an embedding return as tensor
    """

    # tokenize text, map to integers
    encoded_query = tokenizer([query], padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(input_ids=encoded_query.input_ids, attention_mask=encoded_query.attention_mask)
    
    return output.last_hidden_state.mean(dim=1)
    
def similarity_search(query, stored_embeddings, cleaned_documents_df, k, model_name, section=None):
    """
    ...
    """
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # create query embedding
    query_embedding = embed_query(query, tokenizer, model)


    ### section embeddings (depends on how they are stored)
    similarities = []
    for course_code, section_embeddings in stored_embeddings.items():
        # temporary way to get only the learning outcomes embedding
        embedding = section_embeddings[section]
        similarity = calculate_cosine_similarity(query_embedding, embedding)
        similarities.append((course_code, similarity))

    ### document embeddings
    # get similarity between queried embedding and all the stored embeddings
    # similarities = []
    # for course_code, embedding in stored_embeddings.items():
    #     similarity = calculate_cosine_similarity(query_embedding, embedding)
    #     similarities.append((course_code, similarity))
    
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    # Get the top k course codes
    top_k_course_codes = [course_code for course_code, _ in sorted_similarities[:k]]

    retrieved_course_code_text = {}

    for code in top_k_course_codes:
        matched_row = cleaned_documents_df[cleaned_documents_df['course_codes'] == code]

        # extract text from learning outcomes
        cleaned_text = matched_row[section].iloc[0]

        retrieved_course_code_text[code] = cleaned_text
    
    return retrieved_course_code_text

if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    query = 'List me some courses that teaches languages and neuro, languages'
    k = 1

    doc_embeddings_path = '/home/notna/scripts/dit247-nlp/project/data/document_embeddings.pickle'

    section_embeddings_path = '/home/notna/scripts/dit247-nlp/project/data/section_embeddings.pickle'

    cleaned_docs_path = '/home/notna/scripts/dit247-nlp/project/data/cleaned_documents.pickle'

    # load embeddings
    with open(section_embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    # load cleaned docs
    with open(cleaned_docs_path, 'rb') as f:
        cleaned_docs = pickle.load(f)

    # run similarity search
    top_k_docs = similarity_search(query, embeddings, cleaned_docs, k, model_name, section='Course content')

    print(top_k_docs)