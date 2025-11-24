from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file) 
    text = ""

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text
