from pathlib import Path

from atlast.chunking import chunk_document
from atlast.documents import load_document


def test_chunking_is_deterministic_and_preserves_section_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sections.txt"
    path.write_text("one two three four five\fsix seven eight", encoding="utf-8")
    document = load_document(path)

    first = chunk_document(document, chunk_size=3, overlap=1)
    second = chunk_document(document, chunk_size=3, overlap=1)

    assert first == second
    assert [chunk.text for chunk in first] == [
        "one two three",
        "three four five",
        "six seven eight",
    ]
    assert [chunk.metadata["section"] for chunk in first] == [1, 1, 2]
    assert [chunk.metadata["chunk_index"] for chunk in first] == [0, 1, 2]
