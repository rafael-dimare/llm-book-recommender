#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import pipeline

import gradio as gr

PERSIST_DIR = '../data/db_books_embeddings'


def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:

    tone_emotion_dict = dict(
        Happy = 'joy',
        Surprising = 'surprise',
        Angry = 'anger',
        Suspensful = 'fear',
        Sad = 'sadness'
    )
    
    recommendations = db_books.similarity_search(query, k=initial_top_k)
    books_ids = [doc.id for doc in recommendations]
    book_recs = books.query("isbn13.isin(@books_ids)").copy()
    
    if category != "All":
        book_recs.query("simple_categories == @category", inplace=True)    # What about missing categories??

    book_recs.sort_values(tone_emotion_dict.get(tone, 'title_and_subtitle'),
                          ascending=False, inplace=True)

    return book_recs.iloc[:final_top_k, :]


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_description = " ".join(description.split()[:20]) + "..."

        if row['authors'] is np.nan:
            authors_str = "[Unkown author]"
            
        else:
            authors_list = row['authors'].split(";")
            if len(authors_list) == 1:
                authors_str = authors_list[0]
            elif len(authors_list) == 2:
                authors_str = f"{authors_list[0]} and {authors_list[1]}"
            else:
                authors_str = f"{', '.join(authors_list[:-1])}, and {authors_list[-1]}"
    
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
    
        results.append((row['large_thumbnail'], caption))

    return results



# ------------------- Load books database
books = pd.read_csv("../data/books_for_dashboard.csv",
                    dtype={'isbn13': str, 'isbn': str,})

books['large_thumbnail'] = np.where(books['thumbnail'].isna(),
                                    "../data/cover_not_found.png",
                                    books['thumbnail'] + "&wfife=w800")


# ------------------- Load vector databse for semantic recommendations
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = Chroma(
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIR
)



categories = ["All"] + sorted(books['simple_categories'].dropna().unique())
tones = ["All"] + ['Happy', 'Surprising', 'Angry', 'Suspensful', 'Sad']


# ------------------- Start of the `gradio` dashboard

def shutdown_gracefully():
    dashboard.close()

print("Starting dashboard...")
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "What kind of book would you like to read?",
                                placeholder = "e.g., A story about mystery")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

    shutdown_button = gr.Button("Shutdown Dashboard", variant="stop")
    shutdown_button.click(fn=shutdown_gracefully, inputs=None, outputs=None)

    # dashboard.launch(inline=True)
    dashboard.launch(server_name="0.0.0.0", server_port=7864)