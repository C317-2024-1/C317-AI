import os
import re

from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)


def process_pdfs_in_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            output_path = os.path.join(output_directory, filename.replace('.pdf', '.txt'))
            save_text_to_file(text, output_path)
            print(f'Processed {filename}')


def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def process_txt_files_in_directory(input_directory, output_directory=None):
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(input_directory, filename)
            with open(txt_path, 'r', encoding='utf-8') as file:
                text = file.read()

            cleaned_text = clean_text(text)

            if output_directory:
                output_path = os.path.join(output_directory, filename)
            else:
                output_path = txt_path

            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

            print(f'Processed {filename}')


# Extrai o texto dos PDF
process_pdfs_in_directory('PDFs', 'textos')

# Trata os textos gerados
process_txt_files_in_directory('textos')
