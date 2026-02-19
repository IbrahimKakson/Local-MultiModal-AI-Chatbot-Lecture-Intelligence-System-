import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.api.schemas import DocumentChunk

class PDFService:
    def __init__(self):
        # Configure the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, file_bytes: bytes, filename: str) -> list[DocumentChunk]:
        """
        Extracts text from a PDF file (bytes) and chunks it.
        Returns a list of DocumentChunk objects.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            chunks = []
            
            full_text = ""
            # 1. Extract text from each page
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            # 2. Split text into chunks
            if not full_text.strip():
                return []

            split_texts = self.text_splitter.split_text(full_text)
            
            for i, text_chunk in enumerate(split_texts):
                chunk = DocumentChunk(
                    text=text_chunk,
                    page_number=0, # Placeholder, as we are splitting full text
                    source=filename,
                    metadata={"chunk_index": i}
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return []
