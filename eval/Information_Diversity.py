import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from argparse import ArgumentParser


def calculate_snippet_similarities(snippets, model_path):
    model = SentenceTransformer(model_path)  # 可选择其他模型

    snippets = [" ".join(snippet) for snippet in snippets]
    snippet_embeddings = model.encode(snippets)
    
    similarity_matrix = cosine_similarity(snippet_embeddings)
    
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    
    similarities = similarity_matrix[upper_triangle_indices]
    
    mean_similarity = np.mean(similarities)
    
    return mean_similarity, similarities


def get_snippets(data):
    all_snippets = []
    for info in data['info']:
        all_snippets.append(info['snippets'])
    for child in data['children']:
        all_snippets  = all_snippets + get_snippets(data['children'][child])
    return all_snippets

def calculate_deepthink(path, model_path):
    all_dirversity = []
    files = os.listdir(path)
    for file in files:
        print(file)
        with open(os.path.join(path, file),'r') as f:
            all_snippets = []
            data = json.load(f)
            all_snippets = get_snippets(data)
            if len(all_snippets) == 1 or len(all_snippets) == 0:
                continue
            mean_similarity, similarities = calculate_snippet_similarities(all_snippets, model_path)
            diversity = 1 - mean_similarity
            all_dirversity.append(diversity)

    return (sum(all_dirversity)/len(all_dirversity)) 



def main(args):
    map_path = args.mappath
    model_path = args.model_path

    print(calculate_deepthink(map_path, model_path))

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--mappath', type=str, default='../results/map',
                        help='Directory to store the articles.')
    parser.add_argument('--model_path', type=str, default='./models/paraphrase-MiniLM-L6-v2',
                        help='Directory to store the model.')
    main(parser.parse_args())


