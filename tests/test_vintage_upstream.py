import json
import tempfile
import unittest
from pathlib import Path

from gpt_oss.vintage.registry import get_profile
from gpt_oss.vintage.engine import VintageEngine
from gpt_oss.vintage.corpus import prepare_corpus
from gpt_oss.vintage.retrieval import HistLMRetriever


class VintageUpstreamTests(unittest.TestCase):
    def test_registry_distinguishes_generation_and_retrieval(self):
        self.assertEqual(get_profile("talkie-1930-it").task, "generation")
        self.assertEqual(get_profile("histlm-bert-1760-1900").task, "retrieval")
        self.assertFalse(get_profile("vintage-llm-340m").instruction_tuned)

    def test_retrieval_model_cannot_be_used_as_generator(self):
        with self.assertRaises(ValueError):
            VintageEngine("histlm-bert-1760-1900")

    def test_corpus_rejects_post_cutoff_material(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            rows = [
                {"text": "An old passage", "year": 1890, "source": "A"},
                {"text": "A later passage", "year": 1901, "source": "B"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_corpus(source, root / "out", cutoff_year=1899)

    def test_corpus_deduplicates_and_splits_by_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            rows = [
                {"text": "Passage one.", "year": 1880, "source": "Book A"},
                {"text": "Passage one.", "year": 1880, "source": "Book A"},
                {"text": "Passage two.", "year": 1881, "source": "Book B"},
                {"text": "Passage three.", "year": 1882, "source": "Book C"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = prepare_corpus(source, root / "out", validation_fraction=0.34, seed=7)
            self.assertEqual(manifest["unique_documents"], 3)
            self.assertEqual(manifest["deduplicated_documents"], 1)
            self.assertFalse(set(manifest["train_sources"]) & set(manifest["validation_sources"]))
            self.assertTrue((root / "out" / "manifest.json").is_file())

    def test_histlm_retriever_filters_years_without_loading_model(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus = Path(temp) / "passages.jsonl"
            rows = [
                {"passage_id": "a", "text": "Steam engine passage", "year": 1880, "source": "Book A"},
                {"passage_id": "b", "text": "Modern passage", "year": 1901, "source": "Book B"},
            ]
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            retriever = HistLMRetriever(corpus, start_year=1760, end_year=1900)
            self.assertEqual(len(retriever._passages), 1)
            self.assertEqual(retriever._passages[0].passage_id, "a")


if __name__ == "__main__":
    unittest.main()
