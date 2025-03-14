from pathlib import Path


def get_template(file):
    with open(Path(__file__).parent / "templates" / file) as f:
        return f.read()


from seidr.dev import SEIDR as SEIDR
