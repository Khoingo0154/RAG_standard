from ingestion.services.chunker import _find_page_range, _is_valid_chunk, _clean_text


class TestFindPageRange:
    def test_single_page(self):
        boundaries = [(0, 100, 1)]
        start, end = _find_page_range(10, 50, boundaries)
        assert start == 1
        assert end == 1

    def test_cross_two_pages(self):
        boundaries = [(0, 100, 1), (100, 200, 2)]
        start, end = _find_page_range(80, 120, boundaries)
        assert start == 1
        assert end == 2

    def test_middle_page_of_three(self):
        boundaries = [(0, 10, 1), (10, 20, 2), (20, 30, 3)]
        start, end = _find_page_range(12, 18, boundaries)
        assert start == 2
        assert end == 2

    def test_no_matching_boundary(self):
        boundaries = []
        start, end = _find_page_range(0, 10, boundaries)
        assert start == 1
        assert end == 1

    def test_exact_boundary(self):
        boundaries = [(0, 50, 1), (50, 100, 2)]
        start, end = _find_page_range(50, 70, boundaries)
        assert start == 2
        assert end == 2


class TestEdgeCases:
    def test_clean_text_empty_string(self):
        assert _clean_text("") == ""

    def test_clean_text_whitespace_only(self):
        assert _clean_text("   \n  \t  ") == ""

    def test_is_valid_chunk_exactly_fifty(self):
        text = "A" * 50
        assert _is_valid_chunk(text) is True

    def test_is_valid_chunk_forty_nine(self):
        text = "A" * 49
        assert _is_valid_chunk(text) is False
