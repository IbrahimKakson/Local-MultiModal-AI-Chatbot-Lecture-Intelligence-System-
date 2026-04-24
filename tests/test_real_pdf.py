from app.services.pdf_service import PDFService
import os

def test_real_pdf():
    # --- Instructions ---
    # 1. Place a real PDF file in the same folder as this script.
    # 2. Change the 'pdf_path' variable below to match the name of your file.
    # For example, if your file is "syllabus.pdf", change it to: pdf_path = "syllabus.pdf"
    
    pdf_path = "data/uploads/lecture.pdf" 
    
    # --------------------

    if not os.path.exists(pdf_path):
        print(f"[ERROR] Could not find the file '{pdf_path}'.")
        print(f"Please make sure you have a file named '{pdf_path}' in the main project folder.")
        return

    print(f"[START] Found '{pdf_path}'. Starting extraction...")
    
    try:
        # 1. Read the file as raw bytes
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            
        # 2. Run our service
        service = PDFService()
        chunks = service.extract_text_from_pdf(pdf_bytes, pdf_path)
        
        # 3. Print the results
        print(f"[SUCCESS] Extracted {len(chunks)} text chunks.")
        
        if len(chunks) > 0:
            print("\n--- Preview of Chunk 1 ---")
            print(chunks[0].text[:500] + "...") # Print first 500 characters
            
            if len(chunks) > 1:
                print("\n--- Preview of Chunk 2 ---")
                print(chunks[1].text[:500] + "...")
                
    except Exception as e:
        print(f"[ERROR] An error occurred while processing the PDF: {e}")

if __name__ == "__main__":
    test_real_pdf()
