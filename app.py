from flask import Flask, request, jsonify
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# BERT Model
bert_summarizer = pipeline("summarization")

# T5 Model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# BART Model
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extractive_summarization(text, num_sentences=3):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(text)])
    sentence_similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sentence_scores = sentence_similarity_matrix.diagonal()

    filtered_text = [sentence for sentence in text if sentence]
    if len(sentence_scores) != len(text):
        print("Sentence scores and text lists have different lengths.")

    ranked_sentences = sorted(((score, sentence) for score, sentence in zip(sentence_scores, text)), reverse=True)
    summary_sentences = ranked_sentences[:num_sentences]
    summary_sentences = sorted(summary_sentences, key=lambda x: text.index(x[1]))
    summary = ' '.join(sentence for score, sentence in summary_sentences)

    return summary

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    method = data['method']
    
    if method == 'BERT':
        summary = bert_summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    elif method == 'T5':
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    elif method == 'BART':
        summary = bart_summarizer(text, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
    elif method == 'TF-IDF':
        summary = extractive_summarization(text.split('. '))  # Assuming sentences are separated by '. '
    else:
        return jsonify({'error': 'Invalid summarization method'}), 400
    
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
