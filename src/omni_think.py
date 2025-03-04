import os
import sys
import logging
# Add repository root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tools.lm import OpenAIModel_dashscope
from src.tools.rm import GoogleSearchAli
from src.tools.mindmap import MindMap
from src.actions.outline_generation import OutlineGenerationModule
from src.dataclass.Article import Article
from src.actions.article_generation import ArticleGenerationModule
from src.actions.article_polish import ArticlePolishingModule

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_article(topic):
    """Generate a 300-word article based on the provided topic using OmniThink's tools."""
    # Initialize language model with LM_KEY
    kwargs = {
        'api_key': os.getenv("LM_KEY"),
        'temperature': 1.0,
        'top_p': 0.9,
    }
    try:
        lm = OpenAIModel_dashscope(model="deepseek-reasoner", max_tokens=400, **kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize language model: {e}")
        raise

    # Initialize search retriever with SEARCHKEY
    try:
        rm = GoogleSearchAli(k=5, api_key=os.getenv("SEARCHKEY"))
    except Exception as e:
        logger.error(f"Failed to initialize search retriever: {e}")
        raise

    # Parse topic string
    context = ""
    question = topic
    if "Context:" in topic and "Question:" in topic:
        context = topic.split("Context:")[1].split("Question:")[0].strip()
        question = topic.split("Question:")[1].split("Generate")[0].strip()

    try:
        mind_map = MindMap(retriever=rm, gen_concept_lm=lm, depth=2)
        generator = mind_map.build_map(question)
        layers = list(generator)  # Consume generator safely
        if not layers:
            logger.error("Mind map generation returned no layers.")
            raise ValueError("Mind map generation failed.")
        mind_map.prepare_table_for_retrieval()

        ogm = OutlineGenerationModule(lm)
        outline = ogm.generate_outline(topic=question, mindmap=mind_map)
        if not outline:
            logger.error("Outline generation failed.")
            raise ValueError("Outline generation failed.")

        article_with_outline = Article.from_outline_str(topic=question, outline_str=outline)
        ag = ArticleGenerationModule(retriever=rm, article_gen_lm=lm, retrieve_top_k=3, max_thread_num=10)
        article = ag.generate_article(topic=question, mindmap=mind_map, article_with_outline=article_with_outline)
        if not article:
            logger.error("Article generation failed.")
            raise ValueError("Article generation failed.")

        ap = ArticlePolishingModule(article_gen_lm=lm, article_polish_lm=lm)
        polished_article = ap.polish_article(topic=question, draft_article=article)
        if not polished_article:
            logger.error("Article polishing failed.")
            raise ValueError("Article polishing failed.")

        os.makedirs("results", exist_ok=True)
        with open("results/article.md", "w", encoding="utf-8") as f:
            f.write(f"# {question}\n\n{polished_article.to_string()}")
    except Exception as e:
        logger.error(f"Article generation failed: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python omni_think.py <topic>")
        sys.exit(1)
    topic = sys.argv[1]
    generate_article(topic)
