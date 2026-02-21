from pathlib import Path


def load_mapping_from_file(mapping_file: Path) -> dict[str, str]:
    """Load modality mapping from a file.

    Args:
        mapping_file (Path): Path to the mapping file.
    Returns:
        dict[str, str]: The modality mapping.
    """
    try:
        with open(mapping_file, "r") as f:
            mapping = {}
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                mapping[key.strip()] = value.strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to load mapping from {mapping_file}") from exc
    return mapping
