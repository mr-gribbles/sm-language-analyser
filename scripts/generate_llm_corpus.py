"""Generate LLM-written text corpus for AI detection training.

This script creates a corpus of LLM-generated academic-style text using various
prompting strategies to create diverse, realistic AI-generated content that
pairs well with scraped academic papers.
"""

import argparse
import json
import time
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import sys
import os

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.core_logic.llm_text_generator import generate_text_with_gemini
from src.core_logic.corpus_manager import save_record_to_corpus
from src import config


class LLMCorpusGenerator:
    """Generator for creating diverse LLM-written academic text."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the LLM corpus generator.
        
        Args:
            model_name: Name of the LLM model to use.
        """
        self.model_name = model_name
        self.generation_count = 0
    
    def get_academic_prompts(self) -> List[Dict[str, Any]]:
        """Get diverse prompts for generating academic-style text.
        
        Returns:
            List of prompt dictionaries with templates and metadata.
        """
        prompts = [
            {
                "category": "abstract_generation",
                "template": "Write an academic abstract for a research paper about {topic}. Include background, methodology, results, and conclusions. Make it 150-250 words.",
                "topics": [
                    "machine learning optimization algorithms",
                    "natural language processing for sentiment analysis",
                    "computer vision applications in medical imaging",
                    "deep reinforcement learning in robotics",
                    "neural network architectures for time series prediction",
                    "federated learning privacy preservation",
                    "graph neural networks for social network analysis",
                    "transformer models for code generation",
                    "adversarial attacks on image classification",
                    "quantum machine learning algorithms"
                ]
            },
            {
                "category": "introduction_generation",
                "template": "Write the introduction section of an academic paper about {topic}. Include motivation, problem statement, and contribution overview. Make it 200-300 words.",
                "topics": [
                    "automated theorem proving using neural networks",
                    "blockchain applications in supply chain management",
                    "edge computing for IoT data processing",
                    "explainable AI in healthcare decision making",
                    "multi-modal learning for autonomous vehicles",
                    "privacy-preserving machine learning techniques",
                    "sustainable AI and green computing",
                    "human-AI collaboration in creative tasks",
                    "bias detection and mitigation in NLP models",
                    "continual learning in dynamic environments"
                ]
            },
            {
                "category": "methodology_generation",
                "template": "Write a methodology section for a research paper on {topic}. Describe the approach, experimental setup, and evaluation metrics. Make it 250-350 words.",
                "topics": [
                    "few-shot learning for image classification",
                    "attention mechanisms in sequence-to-sequence models",
                    "meta-learning for rapid adaptation",
                    "self-supervised learning from unlabeled data",
                    "neural architecture search optimization",
                    "distributed training of large language models",
                    "active learning for data-efficient training",
                    "domain adaptation in computer vision",
                    "multi-task learning with shared representations",
                    "causal inference in machine learning"
                ]
            },
            {
                "category": "related_work",
                "template": "Write a related work section for a paper on {topic}. Compare different approaches and highlight gaps in current research. Make it 200-300 words.",
                "topics": [
                    "generative adversarial networks for data augmentation",
                    "transfer learning in low-resource scenarios",
                    "interpretable machine learning methods",
                    "robust optimization under uncertainty",
                    "online learning algorithms",
                    "dimensionality reduction techniques",
                    "ensemble methods for improved accuracy",
                    "anomaly detection in time series data",
                    "recommendation systems using deep learning",
                    "natural language generation evaluation metrics"
                ]
            },
            {
                "category": "discussion_generation",
                "template": "Write a discussion section for a research paper on {topic}. Analyze results, discuss implications, and mention limitations. Make it 200-300 words.",
                "topics": [
                    "ethical considerations in AI decision making",
                    "scalability challenges in distributed learning",
                    "robustness of neural networks to adversarial examples",
                    "interpretability vs accuracy trade-offs",
                    "data quality impact on model performance",
                    "computational efficiency of modern architectures",
                    "generalization capabilities across domains",
                    "fairness in algorithmic decision making",
                    "environmental impact of large-scale training",
                    "human factors in AI system design"
                ]
            }
        ]
        return prompts
    
    def get_writing_style_variations(self) -> List[str]:
        """Get different writing style instructions for variation.
        
        Returns:
            List of style instruction strings.
        """
        styles = [
            "Write in a formal academic tone with technical precision.",
            "Use a clear, accessible writing style suitable for a broad scientific audience.",
            "Adopt a concise, direct writing approach with minimal jargon.",
            "Write with detailed explanations and comprehensive coverage of concepts.",
            "Use an analytical writing style with critical evaluation of ideas.",
            "Employ a structured, methodical approach to presenting information.",
            "Write with emphasis on practical applications and real-world relevance.",
            "Use a comparative approach, contrasting different methods and approaches.",
            "Adopt an exploratory tone that discusses open questions and future directions.",
            "Write with focus on interdisciplinary connections and broader implications."
        ]
        return styles
    
    def generate_academic_text(self, prompt_info: Dict[str, Any], style: str) -> str:
        """Generate academic text using LLM.
        
        Args:
            prompt_info: Dictionary containing prompt template and topic.
            style: Writing style instruction.
            
        Returns:
            Generated academic text.
        """
        # Select random topic
        topic = random.choice(prompt_info["topics"])
        
        # Format the prompt
        base_prompt = prompt_info["template"].format(topic=topic)
        
        # Add style instruction
        full_prompt = f"{style}\n\n{base_prompt}"
        
        # Generate text using the LLM
        generated_text = generate_text_with_gemini(
            prompt=full_prompt,
            model_name=self.model_name
        )
        
        return generated_text
    
    def create_corpus_entry(self, generated_text: str, prompt_info: Dict[str, Any], 
                          topic: str, style: str) -> Dict[str, Any]:
        """Create a corpus entry for generated text.
        
        Args:
            generated_text: The LLM-generated text.
            prompt_info: Information about the prompt used.
            topic: The topic that was generated.
            style: The writing style used.
            
        Returns:
            Corpus entry dictionary.
        """
        self.generation_count += 1
        
        # Create a unique ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        entry_id = f"llm_generated_{timestamp}_{self.generation_count:04d}"
        
        # Extract title from the generated text (first line or sentence)
        lines = generated_text.split('\n')
        title = lines[0][:100] + "..." if len(lines[0]) > 100 else lines[0]
        
        corpus_entry = {
            'id': entry_id,
            'source': 'llm_generated',
            'original_content': {
                'raw_text': generated_text,
                'cleaned_text': generated_text,
                'title': title,
                'abstract': generated_text,  # For academic text, the whole text can serve as abstract
                'authors': ['AI Generated'],
                'categories': [f"generated.{prompt_info['category']}"],
                'paper_id': entry_id,
                'url': f"generated://{entry_id}",
                'published_date': datetime.now(timezone.utc).isoformat(),
            },
            'metadata': {
                'scraped_at': datetime.now(timezone.utc).isoformat(),
                'source_type': 'llm_generated',
                'language': 'en',
                'word_count': len(generated_text.split()),
                'char_count': len(generated_text),
                'generation_category': prompt_info['category'],
                'generation_topic': topic,
                'writing_style': style,
                'model_used': self.model_name
            },
            'processing_info': {
                'collected_at': datetime.now(timezone.utc).isoformat(),
                'client_version': '1.0',
                'data_quality': 'high',  # LLM-generated is considered high quality
                'generation_method': 'prompt_based'
            }
        }
        
        return corpus_entry
    
    def generate_corpus(self, num_texts: int, output_dir: str, output_file: str,
                       delay: float = 2.0) -> int:
        """Generate a corpus of LLM-written academic text.
        
        Args:
            num_texts: Number of texts to generate.
            output_dir: Output directory for the corpus.
            output_file: Output filename.
            delay: Delay between API calls.
            
        Returns:
            Number of texts successfully generated.
        """
        prompts = self.get_academic_prompts()
        styles = self.get_writing_style_variations()
        
        generated_count = 0
        
        print(f"Generating {num_texts} academic texts using {self.model_name}")
        print(f"Output: {output_dir}/{output_file}")
        
        for i in range(num_texts):
            try:
                # Select random prompt and style
                prompt_info = random.choice(prompts)
                style = random.choice(styles)
                topic = random.choice(prompt_info["topics"])
                
                print(f"Generating {i+1}/{num_texts}: {prompt_info['category']}")
                
                # Generate text
                generated_text = self.generate_academic_text(prompt_info, style)
                
                if generated_text and len(generated_text.strip()) > 50:
                    # Create corpus entry
                    corpus_entry = self.create_corpus_entry(
                        generated_text, prompt_info, topic, style
                    )
                    
                    # Save to corpus
                    save_record_to_corpus(corpus_entry, output_dir, output_file)
                    generated_count += 1
                    
                    print(f"Generated {len(generated_text.split())} words")
                else:
                    print(f"Generation failed or too short")
                
                # Rate limiting
                if i < num_texts - 1:  # Don't delay after the last generation
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error generating text {i+1}: {e}")
                continue
        
        print(f"Generated {generated_count}/{num_texts} texts successfully")
        return generated_count


def main():
    """Main function for LLM corpus generation."""
    parser = argparse.ArgumentParser(
        description="Generate LLM-written academic text corpus for AI detection training"
    )
    
    parser.add_argument(
        "--num-texts", "-n",
        type=int,
        default=100,
        help="Number of texts to generate (default: 100)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model to use (default: gemini-1.5-flash)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="corpora/llm_generated",
        help="Output directory (default: corpora/llm_generated)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output filename (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output_file:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        args.output_file = f"llm_generated_academic_{timestamp}.jsonl"
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = LLMCorpusGenerator(model_name=args.model)
    
    # Generate corpus
    try:
        generated_count = generator.generate_corpus(
            num_texts=args.num_texts,
            output_dir=args.output_dir,
            output_file=args.output_file,
            delay=args.delay
        )
        
        if generated_count > 0:
            print(f"Corpus generation complete")
            print(f"Generated {generated_count} texts saved to: {args.output_dir}/{args.output_file}")
        else:
            print(f"No texts were generated successfully")
            
    except KeyboardInterrupt:
        print(f"Generation interrupted by user")
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
