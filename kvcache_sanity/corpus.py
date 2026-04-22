from pathlib import Path
from kvcache_sanity.models import Document

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


def load_documents(corpus_dir: Path | None = None) -> dict[str, Document]:
    """Load all .txt documents from the corpus directory.

    Document format: first line must be '# Title', remainder is body content.
    """
    if corpus_dir is None:
        corpus_dir = CORPUS_DIR

    documents: dict[str, Document] = {}
    for path in sorted(corpus_dir.glob("*.txt")):
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8").strip()
        lines = raw.split("\n")

        if lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title = doc_id
            body = raw

        # Rough token estimate: ~4 chars per token
        approx_tokens = len(body) // 4

        documents[doc_id] = Document(
            id=doc_id,
            title=title,
            content=body,
            approximate_tokens=approx_tokens,
        )

    return documents
