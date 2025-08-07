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
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
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
            # Computer Science & Technical Topics
            {
                "category": "cs_abstract_generation",
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
                "category": "cs_methodology_generation",
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
            
            # Psychology Topics
            {
                "category": "psychology_abstract_generation",
                "template": "Write an academic abstract for a psychology research paper about {topic}. Include background, methodology, results, and conclusions. Make it 150-250 words.",
                "topics": [
                    "cognitive biases in decision-making processes",
                    "the impact of social media on adolescent mental health",
                    "neuroplasticity and learning in older adults",
                    "attachment styles and romantic relationship satisfaction",
                    "the effectiveness of mindfulness-based interventions for anxiety",
                    "working memory capacity and academic performance",
                    "cultural differences in emotional expression and regulation",
                    "the role of sleep in memory consolidation",
                    "behavioral interventions for addiction recovery",
                    "developmental psychology of moral reasoning in children"
                ]
            },
            {
                "category": "psychology_methodology_generation",
                "template": "Write a methodology section for a psychology study on {topic}. Describe participants, procedures, measures, and analysis plan. Make it 250-350 words.",
                "topics": [
                    "longitudinal study of personality development",
                    "experimental investigation of stereotype threat",
                    "cross-cultural comparison of parenting styles",
                    "neuroimaging study of emotion regulation",
                    "randomized controlled trial of therapy effectiveness",
                    "observational study of child social behavior",
                    "survey research on workplace stress and burnout",
                    "meta-analysis of cognitive behavioral therapy outcomes",
                    "qualitative study of trauma recovery experiences",
                    "psychometric validation of a new assessment tool"
                ]
            },
            
            # Philosophy Topics
            {
                "category": "philosophy_abstract_generation",
                "template": "Write an academic abstract for a philosophy paper about {topic}. Include the philosophical problem, argument, and conclusions. Make it 150-250 words.",
                "topics": [
                    "the nature of consciousness and the hard problem",
                    "moral responsibility in deterministic universes",
                    "epistemic justification and the Gettier problem",
                    "personal identity and psychological continuity",
                    "the ethics of artificial intelligence and automation",
                    "free will and moral accountability",
                    "the problem of evil and theodicy",
                    "virtue ethics and character development",
                    "the nature of time and temporal experience",
                    "distributive justice and economic inequality"
                ]
            },
            {
                "category": "philosophy_argument_generation",
                "template": "Write a philosophical argument about {topic}. Present the problem, develop your position, consider objections, and defend your view. Make it 300-400 words.",
                "topics": [
                    "the moral status of non-human animals",
                    "whether knowledge requires certainty",
                    "the relationship between mind and body",
                    "the foundations of moral obligation",
                    "the nature of aesthetic experience",
                    "political authority and the social contract",
                    "the problem of induction in scientific reasoning",
                    "environmental ethics and our duties to nature",
                    "the meaning of life and human purpose",
                    "the ethics of genetic enhancement"
                ]
            },
            
            # Business & Management Topics
            {
                "category": "business_abstract_generation",
                "template": "Write an academic abstract for a business research paper about {topic}. Include background, methodology, findings, and implications. Make it 150-250 words.",
                "topics": [
                    "the impact of remote work on organizational culture",
                    "sustainable business practices and financial performance",
                    "digital transformation strategies in traditional industries",
                    "consumer behavior in e-commerce environments",
                    "leadership styles and employee engagement",
                    "supply chain resilience during global disruptions",
                    "the role of artificial intelligence in customer service",
                    "corporate social responsibility and brand loyalty",
                    "innovation management in startup ecosystems",
                    "cross-cultural negotiations in international business"
                ]
            },
            {
                "category": "business_case_study_generation",
                "template": "Write a business case study analysis of {topic}. Include situation analysis, key challenges, strategic options, and recommendations. Make it 300-400 words.",
                "topics": [
                    "a company's digital marketing transformation",
                    "merger and acquisition integration challenges",
                    "crisis management during a product recall",
                    "entering emerging markets with cultural barriers",
                    "implementing sustainable manufacturing processes",
                    "managing organizational change during restructuring",
                    "developing new products for changing demographics",
                    "competitive strategy in disrupted industries",
                    "building strategic partnerships and alliances",
                    "managing stakeholder relationships during growth"
                ]
            },
            
            # Economics Topics
            {
                "category": "economics_abstract_generation",
                "template": "Write an academic abstract for an economics research paper about {topic}. Include research question, methodology, findings, and policy implications. Make it 150-250 words.",
                "topics": [
                    "the effects of minimum wage increases on employment",
                    "behavioral economics and consumer decision-making",
                    "the impact of automation on labor markets",
                    "monetary policy effectiveness during economic crises",
                    "income inequality and economic growth",
                    "the economics of climate change and carbon pricing",
                    "international trade and economic development",
                    "financial market volatility and investor behavior",
                    "the gig economy and traditional employment models",
                    "healthcare economics and policy reform"
                ]
            },
            {
                "category": "economics_analysis_generation",
                "template": "Write an economic analysis of {topic}. Include theoretical framework, empirical evidence, and policy recommendations. Make it 300-400 words.",
                "topics": [
                    "the economic impact of universal basic income",
                    "market failures in healthcare systems",
                    "the economics of education and human capital",
                    "fiscal policy responses to economic recessions",
                    "the role of central banks in financial stability",
                    "economic effects of immigration policies",
                    "competition policy in digital markets",
                    "the economics of renewable energy transition",
                    "urban economics and housing affordability",
                    "international development and poverty reduction"
                ]
            },
            
            # Interdisciplinary Topics
            {
                "category": "interdisciplinary_abstract_generation",
                "template": "Write an academic abstract for an interdisciplinary research paper about {topic}. Include multiple perspectives, methodology, and broader implications. Make it 150-250 words.",
                "topics": [
                    "the psychology of economic decision-making",
                    "ethical implications of business automation",
                    "philosophical foundations of psychological research",
                    "economic analysis of mental health interventions",
                    "business applications of behavioral psychology",
                    "the ethics of economic inequality",
                    "psychological factors in consumer economics",
                    "philosophical perspectives on business ethics",
                    "economic psychology of financial decision-making",
                    "the intersection of technology, ethics, and society"
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
        default="gemini-2.5-flash-lite",
        help="LLM model to use (default: gemini-2.5-flash-lite)"
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
