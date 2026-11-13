from importlib import resources
from pathlib import Path
from kvcache_sanity.models import Document
from kvcache_sanity.paths import user_data_dir

CORPUS_DIR = resources.files("kvcache_sanity") / "data" / "corpus"


def _load_dir(corpus_dir: Path) -> dict[str, Document]:
    """Load all .txt documents from a single directory.

    Document format: first line must be '# Title', remainder is body content.
    """
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


def load_documents(corpus_dir: Path | None = None) -> dict[str, Document]:
    """Load corpus documents.

    With an explicit corpus_dir, that directory is used exclusively (full
    override). Otherwise, loads the bundled corpus and merges in the user
    corpus directory (see paths.user_data_dir()) if it exists — user
    documents win on a matching doc_id.
    """
    if corpus_dir is not None:
        return _load_dir(corpus_dir)

    documents = _load_dir(CORPUS_DIR)

    user_corpus_dir = user_data_dir() / "corpus"
    if user_corpus_dir.is_dir():
        documents.update(_load_dir(user_corpus_dir))

    return documents
