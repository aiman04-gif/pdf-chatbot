from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    text = ""

    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)

        for i in range(len(reader.pages)):
            page = reader.pages[i]
            page_text = page.extract_text()
            if page_text:         # avoid None
                text += page_text

    return text