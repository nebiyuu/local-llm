import pytest
import fitz


def extract_text_from_pdf(pdf_path):
    """Testable version of PDF extraction."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()

    return full_text


class TestExtractTextFromPdf:
    """Tests for the extract_text_from_pdf function."""

    def test_extracts_text_from_valid_pdf(self, sample_pdf_path):
        """Test that text is correctly extracted from a valid PDF."""
        text = extract_text_from_pdf(sample_pdf_path)
        assert isinstance(text, str)
        assert len(text) > 0
        assert "This is page" in text

    def test_handles_single_page_pdf(self, sample_pdf_single_page):
        """Test extraction from a single-page PDF."""
        text = extract_text_from_pdf(sample_pdf_single_page)
        assert "Single page test content" in text

    def test_handles_multipage_pdf(self, sample_pdf_path):
        """Test that all pages are included in extraction."""
        text = extract_text_from_pdf(sample_pdf_path)
        assert "page 1" in text
        assert "page 2" in text
        assert "page 3" in text

    def test_raises_on_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(Exception):
            extract_text_from_pdf("/nonexistent/path/to/file.pdf")

    def test_raises_on_corrupted_file(self, tmp_path):
        """Test that an error is raised for corrupted PDF files."""
        corrupted_pdf = tmp_path / "corrupted.pdf"
        corrupted_pdf.write_bytes(b"NOT A PDF FILE")
        with pytest.raises(Exception):
            extract_text_from_pdf(str(corrupted_pdf))


class TestExtractTextFromPdfContent:
    """Tests for content accuracy in PDF extraction."""

    def test_text_order_preserved(self, sample_pdf_path):
        """Test that text from different pages maintains order."""
        text = extract_text_from_pdf(sample_pdf_path)
        page1_pos = text.find("page 1")
        page2_pos = text.find("page 2")
        page3_pos = text.find("page 3")
        assert page1_pos < page2_pos < page3_pos

    def test_newlines_handled(self, sample_pdf_path):
        """Test that newlines are preserved in extracted text."""
        text = extract_text_from_pdf(sample_pdf_path)
        assert "\n" in text

    def test_special_characters(self, tmp_path):
        """Test extraction of text with special characters."""
        special_pdf = tmp_path / "special.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?\n"
            "Unicode: 你好 مرحبا 👋🎉"
        )
        doc.save(str(special_pdf))
        doc.close()

        text = extract_text_from_pdf(str(special_pdf))
        assert "Special chars" in text
        assert "Unicode" in text