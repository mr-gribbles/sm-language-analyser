"""Generate LLM-written social media corpus for AI detection training.

This script creates a corpus of LLM-generated social media posts using various
prompting strategies to create diverse, realistic AI-generated content that
pairs well with scraped social media data.
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

from src.core.llm_text_generator import generate_text_with_gemini
from src.core.corpus_manager import save_record_to_corpus
from src import config


class SocialMediaCorpusGenerator:
    """Generator for creating diverse LLM-written social media posts."""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize the social media corpus generator.
        
        Args:
            model_name: Name of the LLM model to use.
        """
        self.model_name = model_name
        self.generation_count = 0
    
    def get_social_media_prompts(self) -> List[Dict[str, Any]]:
        """Get diverse prompts for generating social media posts.
        
        Returns:
            List of prompt dictionaries with templates and metadata.
        """
        prompts = [
            # Twitter/X Style Posts
            {
                "category": "twitter_opinion",
                "template": "You are posting your opinion about {topic} on Twitter. Write a single authentic tweet (under 280 characters) expressing your personal view.",
                "topics": [
                    "remote work vs office work",
                    "artificial intelligence and job security",
                    "climate change and individual responsibility",
                    "social media's impact on mental health",
                    "the rise of electric vehicles",
                    "cryptocurrency and traditional banking",
                    "online education vs traditional learning",
                    "the gig economy and worker rights",
                    "privacy concerns with tech companies",
                    "sustainable fashion and fast fashion"
                ]
            },
            {
                "category": "twitter_news_reaction",
                "template": "You just heard about {topic}. Write a single Twitter post (under 280 characters) reacting to this news with your immediate thoughts.",
                "topics": [
                    "major tech company announcing layoffs",
                    "a new environmental policy being announced",
                    "a breakthrough in medical research",
                    "changes in social media platform policies",
                    "new economic policy changes",
                    "a space exploration milestone",
                    "a cybersecurity breach at a major company",
                    "a new AI technology being released",
                    "climate summit outcomes",
                    "companies changing work-from-home policies"
                ]
            },
            
            # Reddit Style Posts
            {
                "category": "reddit_discussion",
                "template": "You want to start a discussion about {topic} on Reddit. Write a post that shares your thoughts and asks the community a genuine question. 100-300 words.",
                "topics": [
                    "the ethics of AI in creative fields",
                    "best practices for work-life balance",
                    "the future of urban transportation",
                    "sustainable living tips and tricks",
                    "mental health resources and support",
                    "the impact of social media algorithms",
                    "career advice for young professionals",
                    "the role of technology in education",
                    "environmental activism and individual action",
                    "financial planning for uncertain times"
                ]
            },
            {
                "category": "reddit_experience_sharing",
                "template": "You want to share your experience with {topic} on Reddit. Write a personal post telling your story authentically. 150-400 words.",
                "topics": [
                    "switching to a more sustainable lifestyle",
                    "dealing with workplace burnout",
                    "learning a new skill during the pandemic",
                    "navigating career changes in tech",
                    "mental health journey and self-care",
                    "building better financial habits",
                    "the challenges of remote work",
                    "overcoming social media addiction",
                    "finding work-life balance as a parent",
                    "the journey to healthier eating habits"
                ]
            },
            
            # LinkedIn Style Posts
            {
                "category": "linkedin_professional",
                "template": "You're sharing your professional thoughts about {topic} on LinkedIn. Write a thoughtful post that would resonate with your professional network. 100-200 words.",
                "topics": [
                    "the importance of continuous learning",
                    "building effective remote teams",
                    "navigating career transitions",
                    "the future of workplace diversity",
                    "leadership lessons from recent challenges",
                    "the role of mentorship in career growth",
                    "adapting to technological change",
                    "building professional networks online",
                    "the importance of emotional intelligence",
                    "sustainable business practices"
                ]
            },
            {
                "category": "linkedin_industry_insights",
                "template": "You're posting on LinkedIn about trends you've noticed in {topic}. Share your professional insights and analysis. 150-300 words.",
                "topics": [
                    "artificial intelligence in the workplace",
                    "the future of digital marketing",
                    "remote work technology solutions",
                    "sustainability in business operations",
                    "the evolution of customer service",
                    "data privacy and business compliance",
                    "the impact of automation on jobs",
                    "emerging trends in professional development",
                    "the changing landscape of entrepreneurship",
                    "innovation in financial technology"
                ]
            },
            
            # Instagram/Facebook Style Posts
            {
                "category": "instagram_lifestyle",
                "template": "You're posting on Instagram about {topic}. Share your personal experience or thoughts in a lifestyle-focused way. Include hashtags. 50-150 words.",
                "topics": [
                    "morning routine and productivity",
                    "sustainable fashion choices",
                    "healthy meal prep ideas",
                    "mindfulness and meditation practices",
                    "travel experiences and cultural insights",
                    "fitness and wellness journey",
                    "creative hobbies and self-expression",
                    "home organization and minimalism",
                    "supporting local businesses",
                    "environmental conservation efforts"
                ]
            },
            {
                "category": "facebook_community",
                "template": "You're posting in your local community Facebook group about {topic}. Write something helpful and community-focused. 100-250 words.",
                "topics": [
                    "local environmental initiatives",
                    "community events and activities",
                    "supporting small businesses locally",
                    "parenting tips and experiences",
                    "neighborhood safety and security",
                    "local government and civic engagement",
                    "community volunteering opportunities",
                    "local food and restaurant recommendations",
                    "educational resources for families",
                    "community health and wellness programs"
                ]
            },
            
            # General Social Media Content
            {
                "category": "motivational_post",
                "template": "You want to share something motivational about {topic} on social media. Write an inspiring post that encourages others. 50-150 words.",
                "topics": [
                    "overcoming challenges and setbacks",
                    "pursuing personal goals and dreams",
                    "the importance of self-care",
                    "building resilience and mental strength",
                    "embracing change and new opportunities",
                    "the value of learning from failure",
                    "finding purpose in daily life",
                    "building meaningful relationships",
                    "celebrating small wins and progress",
                    "maintaining optimism during difficult times"
                ]
            },
            {
                "category": "educational_content",
                "template": "You want to help people understand {topic} better. Create a social media post that explains it clearly and helpfully. 100-200 words.",
                "topics": [
                    "how climate change affects daily life",
                    "the basics of personal finance",
                    "understanding mental health and wellness",
                    "the impact of social media algorithms",
                    "sustainable living practices",
                    "the importance of digital privacy",
                    "how artificial intelligence works",
                    "the benefits of renewable energy",
                    "understanding cryptocurrency basics",
                    "the psychology of habit formation"
                ]
            }
        ]
        return prompts
    
    def get_social_media_style_variations(self) -> List[str]:
        """Get different social media writing style instructions for variation.
        
        Returns:
            List of style instruction strings.
        """
        styles = [
            "Write in a casual, conversational tone as if talking to a friend.",
            "Use an enthusiastic and energetic voice with emojis where appropriate.",
            "Adopt a professional but approachable tone suitable for LinkedIn.",
            "Write with a humorous and witty style that engages readers.",
            "Use an authentic, personal voice sharing genuine thoughts and experiences.",
            "Employ a motivational and inspiring tone that uplifts the audience.",
            "Write with a question-asking, community-engaging style.",
            "Use a storytelling approach that makes the content relatable.",
            "Adopt an informative but accessible tone for educational content.",
            "Write with a trendy, current style that reflects social media culture."
        ]
        return styles
    
    def generate_social_media_post(self, prompt_info: Dict[str, Any], style: str) -> str:
        """Generate social media post using LLM.
        
        Args:
            prompt_info: Dictionary containing prompt template and topic.
            style: Writing style instruction.
            
        Returns:
            Generated social media post.
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
        """Create a corpus entry for generated social media post.
        
        Args:
            generated_text: The LLM-generated social media post.
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
        
        # Extract title/preview from the generated text (first line or sentence, truncated)
        lines = generated_text.split('\n')
        first_line = lines[0] if lines else generated_text
        title = first_line[:100] + "..." if len(first_line) > 100 else first_line
        
        corpus_entry = {
            'id': entry_id,
            'source': 'llm_generated',
            'original_content': {
                'raw_text': generated_text,
                'cleaned_text': generated_text,
                'title': title,
                'content': generated_text,  # For social media posts, content is the main field
                'author': 'AI Generated',
                'platform': prompt_info['category'].split('_')[0],  # Extract platform from category
                'post_type': prompt_info['category'],
                'post_id': entry_id,
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
                'model_used': self.model_name,
                'platform_category': prompt_info['category'].split('_')[0]
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
        """Generate a corpus of LLM-written social media posts.
        
        Args:
            num_texts: Number of posts to generate.
            output_dir: Output directory for the corpus.
            output_file: Output filename.
            delay: Delay between API calls.
            
        Returns:
            Number of posts successfully generated.
        """
        prompts = self.get_social_media_prompts()
        styles = self.get_social_media_style_variations()
        
        generated_count = 0
        
        print(f"Generating {num_texts} social media posts using {self.model_name}")
        print(f"Output: {output_dir}/{output_file}")
        
        for i in range(num_texts):
            try:
                # Select random prompt and style
                prompt_info = random.choice(prompts)
                style = random.choice(styles)
                topic = random.choice(prompt_info["topics"])
                
                print(f"Generating {i+1}/{num_texts}: {prompt_info['category']}")
                
                # Generate post
                generated_text = self.generate_social_media_post(prompt_info, style)
                
                if generated_text and len(generated_text.strip()) > 10:  # Lower threshold for social media
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
                print(f"Error generating post {i+1}: {e}")
                continue
        
        print(f"Generated {generated_count}/{num_texts} posts successfully")
        return generated_count


def main():
    """Main function for social media corpus generation."""
    parser = argparse.ArgumentParser(
        description="Generate LLM-written social media corpus for AI detection training"
    )
    
    parser.add_argument(
        "--num-posts", "-n",
        type=int,
        default=100,
        help="Number of social media posts to generate (default: 100)"
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
        default="corpora/llm_generated_social_media",
        help="Output directory (default: corpora/llm_generated_social_media)"
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
        args.output_file = f"llm_generated_social_media_{timestamp}.jsonl"
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SocialMediaCorpusGenerator(model_name=args.model)
    
    # Generate corpus
    try:
        generated_count = generator.generate_corpus(
            num_texts=args.num_posts,
            output_dir=args.output_dir,
            output_file=args.output_file,
            delay=args.delay
        )
        
        if generated_count > 0:
            print(f"Social media corpus generation complete")
            print(f"Generated {generated_count} posts saved to: {args.output_dir}/{args.output_file}")
        else:
            print(f"No posts were generated successfully")
            
    except KeyboardInterrupt:
        print(f"Generation interrupted by user")
    except Exception as e:
        print(f"Error during generation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
