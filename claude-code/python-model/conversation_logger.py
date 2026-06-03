import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _daily_conversations_path(date: Optional[datetime] = None) -> Path:
    d = date or datetime.now(timezone.utc)
    name = f'conversations-{d.strftime("%Y-%m-%d")}.jsonl'
    return DATA_DIR / name


def append_conversation(model: str, user: str, assistant: str, metadata: Optional[dict] = None) -> Path:
    """Append a single conversation turn to today's conversations file.

    The written record contains: timestamp, model, user, assistant, text (combined)
    """
    rec = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'model': str(model),
        'user': user,
        'assistant': assistant,
        'text': f"user: {user}\nassistant: {assistant}",
    }
    if metadata:
        rec['meta'] = metadata

    path = _daily_conversations_path()
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return path


def list_conversation_files():
    return sorted(DATA_DIR.glob('conversations-*.jsonl'))


def combine_conversations(output_path: Optional[Path] = None) -> Path:
    """Combine all daily conversation files into a single JSONL training file.

    Returns the path to the combined file.
    """
    if output_path is None:
        output_path = DATA_DIR / 'combined_training.jsonl'

    with open(output_path, 'w', encoding='utf-8') as out:
        for p in list_conversation_files():
            with open(p, 'r', encoding='utf-8') as src:
                for line in src:
                    out.write(line)

    return output_path


def normalize_conversations_for_training(output_path: Optional[Path] = None) -> Path:
    """Create a normalized JSONL training file with {'prompt':..., 'response':...} per line.

    Reads daily conversation files and writes normalized entries.
    """
    if output_path is None:
        output_path = DATA_DIR / 'combined_training_normalized.jsonl'

    with open(output_path, 'w', encoding='utf-8') as out:
        for p in list_conversation_files():
            with open(p, 'r', encoding='utf-8') as src:
                for line in src:
                    try:
                        rec = json.loads(line)
                        prompt = rec.get('user', '')
                        response = rec.get('assistant', '')
                        norm = {'prompt': prompt, 'response': response}
                        out.write(json.dumps(norm, ensure_ascii=False) + '\n')
                    except Exception:
                        # skip malformed lines
                        continue

    return output_path


def archive_conversations(older_than_days: int = 30, archive_dir: Optional[Path] = None) -> int:
    """Move conversation files older than `older_than_days` into an archive folder.

    Returns the number of files archived.
    """
    from datetime import datetime, timedelta

    if archive_dir is None:
        archive_dir = DATA_DIR / 'archive'
    archive_dir.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    moved = 0
    for p in list_conversation_files():
        # filenames are conversations-YYYY-MM-DD.jsonl; parse date
        name = p.stem  # conversations-YYYY-MM-DD
        try:
            date_part = name.split('-', 1)[1]
            file_date = datetime.strptime(date_part, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except Exception:
            # fallback to mtime
            mtime = datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
            file_date = mtime

        if file_date < cutoff:
            target = archive_dir / p.name
            p.rename(target)
            moved += 1

    return moved
