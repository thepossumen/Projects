import os
from pypdf import PdfReader
import pickle

# List of titles (section headers) in the order they appear in the documents
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

# Function to extract and split text from a single PDF
def process_pdf(pdf_path):
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
        sections[title] = section_text

    return sections

if __name__ == '__main__':
    # path to pdfs
    pdf_directory = "/home/notna/scripts/dit247-nlp/project/data/pdfs"
    
    all_documents = []

    # iterate over all pdfs
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            document_sections = process_pdf(pdf_path)
            all_documents.append(document_sections)

    # save
    path = '/home/notna/scripts/dit247-nlp/project/data/'
    with open(path+'documents.pickle', 'wb') as f:
        pickle.dump(all_documents, f, pickle.HIGHEST_PROTOCOL)