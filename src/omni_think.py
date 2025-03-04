import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python omni_think.py <topic>")
        exit(1)
    topic = sys.argv[1]
    generate_article(topic)
