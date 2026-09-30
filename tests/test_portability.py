import json
import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_python_files_compile(self):
        for path in (ROOT / "scripts").glob("*.py"):
            py_compile.compile(str(path), doraise=True)

    def test_profiles_are_valid_json(self):
        for path in (ROOT / "configs").glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("processing", data)

    def test_no_machine_specific_paths(self):
        forbidden = (
            "/home/" + "antonio",
            "/media/" + "antonio",
            "Hard " + "Disk2",
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix in {".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for value in forbidden:
                self.assertNotIn(value, text, f"{value!r} found in {path}")


if __name__ == "__main__":
    unittest.main()
