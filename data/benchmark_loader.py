"""
Benchmark Loader for D3 Framework.

Unified interface for loading MT-Bench, AlignBench, and AUTO-J benchmarks
as described in Section 4.1 of the D3 paper.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkSample:
    """A single benchmark sample for pairwise evaluation."""
    question_id: str
    question: str
    answer1: str
    answer2: str
    human_label: int  # 0 = answer1 better, 1 = answer2 better
    category: Optional[str] = None
    model1: Optional[str] = None
    model2: Optional[str] = None
    metadata: Optional[Dict] = None


class BenchmarkLoader:
    """Base class for benchmark loaders."""
    
    def __init__(self, name: str):
        self.name = name
        self.samples: List[BenchmarkSample] = []
        
    def load(self, path: str) -> List[BenchmarkSample]:
        """Load benchmark data from path."""
        raise NotImplementedError
    
    def get_samples(self, category: Optional[str] = None, limit: Optional[int] = None) -> List[BenchmarkSample]:
        """Get samples, optionally filtered by category."""
        samples = self.samples
        if category:
            samples = [s for s in samples if s.category == category]
        if limit:
            samples = samples[:limit]
        return samples
    
    def get_categories(self) -> List[str]:
        """Get unique categories in the benchmark."""
        return list(set(s.category for s in self.samples if s.category))
    
    def __len__(self) -> int:
        return len(self.samples)


class MTBenchLoader(BenchmarkLoader):
    """
    Loader for MT-Bench dataset.
    
    Per the paper: "80 multi-turn conversational questions testing 
    general-purpose helpfulness and instruction-following"
    """
    
    def __init__(self):
        super().__init__("MT-Bench")
    
    def load(self, path: str) -> List[BenchmarkSample]:
        """
        Load from preprocessed Excel file.
        
        Expected columns:
        - Question: The question text
        - Response_A: First response
        - Response_B: Second response
        - Model_A_Score: Binary (1 = win)
        - Model_B_Score: Binary (1 = win)
        """
        try:
            import pandas as pd
            df = pd.read_excel(path)
            
            self.samples = []
            for idx, row in df.iterrows():
                # Human label: 1 if Model_A wins, 0 if Model_B wins
                if pd.notna(row.get('Model_A_Score')) and pd.notna(row.get('Model_B_Score')):
                    human_label = 1 if row['Model_A_Score'] > row['Model_B_Score'] else 0
                else:
                    human_label = 0
                
                sample = BenchmarkSample(
                    question_id=str(idx),
                    question=str(row.get('Question', '')),
                    answer1=str(row.get('Response_A', '')),
                    answer2=str(row.get('Response_B', '')),
                    human_label=human_label,
                    category=str(row.get('Category', 'general')) if 'Category' in row else None
                )
                self.samples.append(sample)
                
            print(f"Loaded {len(self.samples)} samples from MT-Bench")
            return self.samples
            
        except ImportError:
            print("pandas required for Excel loading. Install with: pip install pandas openpyxl")
            return []
    
    def load_from_huggingface(self) -> List[BenchmarkSample]:
        """Load directly from HuggingFace datasets."""
        try:
            from datasets import load_dataset
            
            ds = load_dataset("lmsys/mt_bench_human_judgments", split="train")
            
            self.samples = []
            for idx, item in enumerate(ds):
                sample = BenchmarkSample(
                    question_id=str(idx),
                    question=item.get('question', ''),
                    answer1=item.get('answer_a', ''),
                    answer2=item.get('answer_b', ''),
                    human_label=1 if item.get('winner', '') == 'model_a' else 0,
                    model1=item.get('model_a', ''),
                    model2=item.get('model_b', '')
                )
                self.samples.append(sample)
            
            print(f"Loaded {len(self.samples)} samples from HuggingFace MT-Bench")
            return self.samples
            
        except ImportError:
            print("datasets library required. Install with: pip install datasets")
            return []


class AlignBenchLoader(BenchmarkLoader):
    """
    Loader for AlignBench dataset.
    
    Per the paper: "683 alignment-focused questions covering helpfulness, 
    harmlessness, and ethical reasoning (professionally translated to English)"
    """
    
    def __init__(self):
        super().__init__("AlignBench")
    
    def load(self, path: str) -> List[BenchmarkSample]:
        """
        Load AlignBench from JSON format.
        
        Expected format (after preprocessing):
        [
            {
                "id": "align_001",
                "question": "...",
                "answer_a": "...",
                "answer_b": "...",
                "human_preference": "a" or "b",
                "category": "helpfulness/harmlessness/ethics"
            }
        ]
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.samples = []
            for item in data:
                human_label = 1 if item.get('human_preference', 'a') == 'a' else 0
                
                sample = BenchmarkSample(
                    question_id=str(item.get('id', len(self.samples))),
                    question=item.get('question', ''),
                    answer1=item.get('answer_a', ''),
                    answer2=item.get('answer_b', ''),
                    human_label=human_label,
                    category=item.get('category'),
                    metadata=item.get('metadata', {})
                )
                self.samples.append(sample)
            
            print(f"Loaded {len(self.samples)} samples from AlignBench")
            return self.samples
            
        except FileNotFoundError:
            print(f"AlignBench file not found at {path}")
            print("Download from: https://github.com/THUDM/AlignBench")
            return []
    
    @staticmethod
    def get_download_instructions() -> str:
        return """
        AlignBench Download Instructions:
        1. Visit: https://github.com/THUDM/AlignBench
        2. Download the dataset files
        3. Preprocess to pairwise comparison format
        4. Save as JSON with expected schema
        """


class AutoJLoader(BenchmarkLoader):
    """
    Loader for AUTO-J dataset.
    
    Per the paper: "58 real-world scenarios with 3,436 pairwise comparisons 
    spanning creative writing, technical explanation, and diverse task domains"
    """
    
    def __init__(self):
        super().__init__("AUTO-J")
    
    def load(self, path: str) -> List[BenchmarkSample]:
        """
        Load AUTO-J from JSON format.
        
        Expected format (after preprocessing):
        [
            {
                "id": "autoj_001",
                "scenario": "creative_writing",
                "question": "...",
                "response_1": "...",
                "response_2": "...",
                "gpt4_preference": 1 or 2,
                "human_preference": 1 or 2
            }
        ]
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.samples = []
            for item in data:
                # Use human preference if available, otherwise GPT-4 judgment
                pref = item.get('human_preference', item.get('gpt4_preference', 1))
                human_label = 1 if pref == 1 else 0
                
                sample = BenchmarkSample(
                    question_id=str(item.get('id', len(self.samples))),
                    question=item.get('question', ''),
                    answer1=item.get('response_1', ''),
                    answer2=item.get('response_2', ''),
                    human_label=human_label,
                    category=item.get('scenario'),
                    metadata={
                        'gpt4_preference': item.get('gpt4_preference'),
                        'human_preference': item.get('human_preference')
                    }
                )
                self.samples.append(sample)
            
            print(f"Loaded {len(self.samples)} samples from AUTO-J")
            return self.samples
            
        except FileNotFoundError:
            print(f"AUTO-J file not found at {path}")
            print("Download from: https://github.com/GAIR-NLP/auto-j")
            return []
    
    @staticmethod
    def get_download_instructions() -> str:
        return """
        AUTO-J Download Instructions:
        1. Visit: https://github.com/GAIR-NLP/auto-j
        2. Download the pairwise evaluation dataset
        3. Convert to JSON with expected schema
        """


def get_loader(benchmark_name: str) -> BenchmarkLoader:
    """Factory function to get appropriate loader."""
    loaders = {
        "mt-bench": MTBenchLoader,
        "mtbench": MTBenchLoader,
        "alignbench": AlignBenchLoader,
        "align-bench": AlignBenchLoader,
        "auto-j": AutoJLoader,
        "autoj": AutoJLoader
    }
    
    name_lower = benchmark_name.lower().replace("_", "-")
    if name_lower in loaders:
        return loaders[name_lower]()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(loaders.keys())}")


# Example usage
if __name__ == "__main__":
    print("=== Benchmark Loaders Test ===")
    
    # Test MT-Bench loader
    mt_loader = get_loader("mt-bench")
    print(f"Created {mt_loader.name} loader")
    
    # Test AlignBench loader
    align_loader = get_loader("alignbench")
    print(f"Created {align_loader.name} loader")
    print(align_loader.get_download_instructions())
    
    # Test AUTO-J loader
    autoj_loader = get_loader("auto-j")
    print(f"Created {autoj_loader.name} loader")