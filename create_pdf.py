from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_invoice():
    pdf_filename = "sample_invoice.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "COMMERCIAL INVOICE")
    
    # Invoice Metadata
    c.setFont("Helvetica", 10)
    c.drawString(50, 700, "Invoice Number: #10492")
    c.drawString(50, 685, "Date: October 24, 2025")
    
    # Exporter / Importer
    c.drawString(50, 650, "Exporter: Shenzhen Industrial Components Ltd.")
    c.drawString(50, 635, "Origin: China")
    c.drawString(300, 650, "Importer: USA Electronics Logistics Inc.")
    c.drawString(300, 635, "Destination: Port of Long Beach, CA")
    
    # Line Items
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 580, "Item Description")
    c.drawString(300, 580, "Declared HTS")
    c.drawString(420, 580, "Qty")
    c.drawString(480, 580, "Total Value")
    
    c.line(50, 572, 550, 572)
    
    c.setFont("Helvetica", 10)
    c.drawString(50, 550, "Custom Molded Plastic Enclosures")
    c.drawString(50, 535, "for Industrial Circuit Boards")
    c.drawString(300, 550, "3926.90.9988")
    c.drawString(420, 550, "5,000 units")
    c.drawString(480, 550, "$45,000 USD")
    
    c.save()
    print(f"Successfully generated {pdf_filename} in your project folder!")

if __name__ == "__main__":
    create_sample_invoice()