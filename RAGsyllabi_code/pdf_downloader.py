### test script for downloading pdfs

import requests
from bs4 import BeautifulSoup
import os
import re

max_downloads = 2
download_count = 0

# base url
pdf_base_url = 'https://kursplaner.gu.se/pdf/kurs/en/'

# Directory to save the downloaded PDFs
download_directory = '/home/notna/scripts/dit247-nlp/project/data'

# create directory if it does not exist
os.makedirs(download_directory, exist_ok=True)

def download_pdf(course_code, directory):
    """
    Download a PDF given the course code.
    """
    pdf_url = pdf_base_url + course_code
    pdf_response = requests.get(pdf_url)
    filename = os.path.join(directory, course_code + '.pdf')

    if pdf_response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(pdf_response.content)
            print(f"Downloaded {course_code}.pdf")
    else:
        print(f"Failed to download {course_code}.pdf")

# test codes
test_course_codes = ['KBT086', 'FFT810', 'FAB820']

for course_code in test_course_codes:
        download_pdf(course_code, download_directory)

print(f"Finished downloading PDFs.")