import os
from pypdf import PdfReader
import pickle
import re
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# list of titles (section headers) in the order they appear in the documents
titles = [
    "Confirmation",
    "Position in the educational system",
    "Entry requirements",
    "Learning outcomes",
    "Course content",
    "Form of teaching",
    "Assessment",
    "Grades",
    "Course evaluation",
    "Additional information"
]

num_titles = len(titles)

def scrape_pdf(pdf_path, course_code):
    """
    Extract text from pdf...
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    
    # separate full text into sections
    sections = {}
    for i, title in enumerate(titles):
        # each section starts with the title
        start_index = text.find(title)
        # end current section at the next title 
        end_index = text.find(titles[i + 1]) if i + 1 < len(titles) else len(text)
        section_text = text[start_index:end_index].strip()
        sections[title] = f'course code: {course_code}, '+section_text

    return sections

def clean_text(text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = text.replace('\n', ' ')  # Remove newlines
        text = text.strip()  # Trim leading and trailing whitespace
        text = text.lower() # make text into lowercase
        return text

def preprocess_pdf(pdf_directory, save_path=None):
    """
    Preprocess pdfs to be ready for embedding model. 
    """

    all_documents = []
    course_codes = []

    # iterate over all pdfs
    print(f'Extracting text from the documents...')
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            # extract course code
            course_code = filename[0:6]
            document_sections = scrape_pdf(pdf_path, course_code)
            all_documents.append(document_sections)
            # save the course code
            course_codes.append(course_code)
    
    # create df where each row is a document/pdf, and the columns are the text in each section
    df = pd.DataFrame(all_documents)
    # clean the text
    print(f'Cleaning the text...')
    df_cleaned = df.applymap(clean_text)
    # add course codes as a column
    df_cleaned['course_codes'] = course_codes

    # if we want to save the cleaned pdfs before creating embeddings
    if save_path:
        print(f'Saving cleaned text data to: {save_path}')
        with open(save_path+'cleaned_documents.pickle', 'wb') as f:
            pickle.dump(df_cleaned, f, pickle.HIGHEST_PROTOCOL)
    
    return df_cleaned

def create_section_embeddings(model_name, df_docs, save_path=None):
    """
    Creates embedding for each section in the document.
    """
    
    # init model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    def create_embedding(text_batch):
        # convert series into list of strings
        text_list = text_batch.tolist()
        # tokenize text, map to integers
        encoded_text = tokenizer(text_list, padding='max_length', max_length=512, truncation=True, return_tensors="pt") ## parameters?

        with torch.no_grad():
            output = model(input_ids=encoded_text.input_ids, attention_mask=encoded_text.attention_mask)
        
        return output.last_hidden_state.mean(dim=1)

    embeddings = []

    print(f'Creating embeddings...')
    # iterate over each document at a time, this will process each document as a batch (we might want to have a larger batch size, in that case we would rework this a little bit)
    for _, document in tqdm(df_docs.iterrows(), total=df_docs.shape[0]):
        # exclude course code column
        sections = document[:-1]
        # create embeddings for each document
        section_embeddings = create_embedding(sections)
        # this list will contain all the section embeddings, where the first 10 belongs to the first document, next 10 belongs to the second... 
        embeddings.extend(section_embeddings)

    # init dict to store section embeddings  
    d_section_embeddings = {}

    # iterate over the list of embeddings, map embeddings to course codes and section titles
    for index, course_code in enumerate(df_docs['course_codes']):
        # calculate the start and end index for the embeddings of this course
        start_idx = index * num_titles
        end_idx = start_idx + num_titles

        # map section titles to their embedding
        sections_embeddings = dict(zip(titles, embeddings[start_idx:end_idx]))

        # Add this to your dictionary
        d_section_embeddings[course_code] = sections_embeddings
    
    if save_path:
        print(f'Saving section embeddings to: {save_path}')
        with open(save_path+'section_embeddings.pickle', 'wb') as f:
            pickle.dump(d_section_embeddings, f, pickle.HIGHEST_PROTOCOL)

    return d_section_embeddings

def create_document_embeddings(section_embeddings, save_path=None):
    """
    Create document embeddings based on the section embeddings.
    """

    # init dict to store document embeddings, where key is the course code and value the averaged section embeddings
    d_document_embeddings = {}

    for course_code, sections in section_embeddings.items():
        # Calculate the sum of all section embeddings
        sum_embeddings = sum(sections.values())

        # average section embeddings to get document embedding
        # here we can probably do something else and better???? e.g. weighted average...
        avg_embedding = sum_embeddings / len(sections)

        # Store the average embedding in the dictionary
        d_document_embeddings[course_code] = avg_embedding
 
    if save_path:
        print(f'Saving document embeddings to: {save_path}')
        with open(save_path+'document_embeddings.pickle', 'wb') as f:
            pickle.dump(d_document_embeddings, f, pickle.HIGHEST_PROTOCOL)
    
    return d_document_embeddings

if __name__ == '__main__':

    pdf_path = '/home/notna/scripts/dit247-nlp/project/data/pdfs'
    save_path = '/home/notna/scripts/dit247-nlp/project/data/'

    cleaned_docs = preprocess_pdf(pdf_path, save_path=save_path)

    model_name = 'bert-base-uncased'
    section_embeddings = create_section_embeddings(model_name, cleaned_docs, save_path=save_path)

    document_embeddings = create_document_embeddings(section_embeddings, save_path=save_path)