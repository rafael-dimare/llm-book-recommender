{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "dcd2ddb2-2f52-432f-9680-2843cfd37bef",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/rhernand/.local/lib/python3.9/site-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n",
      "/home/rhernand/.local/lib/python3.9/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import os\n",
    "\n",
    "from langchain_community.document_loaders import TextLoader\n",
    "from langchain_text_splitters import CharacterTextSplitter\n",
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from langchain_chroma import Chroma\n",
    "\n",
    "from transformers import pipeline\n",
    "\n",
    "import gradio as gr\n",
    "\n",
    "PERSIST_DIR = '../data/db_books_embeddings'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4c8665e7-2c2e-4473-aab1-35516bd9ae64",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load books database\n",
    "books = pd.read_csv(\"../data/books_for_dashboard.csv\",\n",
    "                    dtype={'isbn13': str, 'isbn': str,})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "da680fae-62af-4500-82db-3c7c54c8a85e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/rhernand/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:182: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:109.)\n",
      "  return torch._C._cuda_getDeviceCount() > 0\n",
      "/home/rhernand/.local/lib/python3.9/site-packages/huggingface_hub/file_download.py:945: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n",
      "/home/rhernand/.local/lib/python3.9/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successfully loaded database with 5197 documents from '../data/db_books_embeddings'.\n",
      "Query Result: Returning to his hometown of Bramwell after years of wandering, mercenary Darrick Lang discovers that a dark and horrifying force has ensnared its citizens and now seeks to seize him, in a chilling novel of dark fantasy, based on the popular video game. Original. (A Blizzard Entertainment M-rated electronic game) (Horror)\n"
     ]
    }
   ],
   "source": [
    "# Load vector databse for semantic recommendations\n",
    "embedding_model = HuggingFaceEmbeddings(model_name=\"sentence-transformers/all-MiniLM-L6-v2\")\n",
    "\n",
    "db_books = Chroma(\n",
    "    embedding_function=embedding_model,\n",
    "    persist_directory=PERSIST_DIR\n",
    ")\n",
    "\n",
    "count = db_books._collection.count()\n",
    "print(f\"Successfully loaded database with {count} documents from '{PERSIST_DIR}'.\")\n",
    "\n",
    "# Example query to show it works\n",
    "results = db_books.similarity_search(\"Fantasy\", k=1)\n",
    "print(f\"Query Result: {results[0].page_content}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6609e85a-70b5-4b8f-b663-31946c22d014",
   "metadata": {},
   "outputs": [],
   "source": [
    "books['large_thumbnail'] = np.where(books['thumbnail'].isna(),\n",
    "                                    \"../data/cover_not_found.png\",\n",
    "                                    books['thumbnail'] + \"&wfife=w800\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e10c82fd-cb1a-405e-844b-474125f9eed7",
   "metadata": {},
   "outputs": [],
   "source": [
    "def retrieve_semantic_recommendations(\n",
    "    query: str,\n",
    "    category: str = None,\n",
    "    tone: str = None,\n",
    "    initial_top_k: int = 50,\n",
    "    final_top_k: int = 16\n",
    ") -> pd.DataFrame:\n",
    "\n",
    "    tone_emotion_dict = dict(\n",
    "        Happy = 'joy',\n",
    "        Surprising = 'surprise',\n",
    "        Angry = 'anger',\n",
    "        Suspensful = 'fear',\n",
    "        Sad = 'sadness'\n",
    "    )\n",
    "    \n",
    "    recommendations = db_books.similarity_search(query, k=initial_top_k)\n",
    "    books_ids = [doc.id for doc in recommendations]\n",
    "    book_recs = books.query(\"isbn13.isin(@books_ids)\").copy()\n",
    "    \n",
    "    if category != \"All\":\n",
    "        book_recs.query(\"simple_categories == @category\", inplace=True)    # What about missing categories??\n",
    "\n",
    "    book_recs.sort_values(tone_emotion_dict.get(tone, 'title_and_subtitle'),\n",
    "                          ascending=False, inplace=True)\n",
    "\n",
    "    return book_recs.iloc[:final_top_k, :]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "50bffa03-a4d5-4bc3-9a1a-4c0f0209e128",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend_books(\n",
    "    query: str,\n",
    "    category: str,\n",
    "    tone: str\n",
    "):\n",
    "    recommendations = retrieve_semantic_recommendations(query, category, tone)\n",
    "    \n",
    "    results = []\n",
    "    for _, row in recommendations.iterrows():\n",
    "        description = row[\"description\"]\n",
    "        truncated_description = \" \".join(description.split()[:20]) + \"...\"\n",
    "\n",
    "        if row['authors'] is np.nan:\n",
    "            authors_str = \"[Unkown author]\"\n",
    "            \n",
    "        else:\n",
    "            authors_list = row['authors'].split(\";\")\n",
    "            if len(authors_list) == 1:\n",
    "                authors_str = authors_list[0]\n",
    "            elif len(authors_list) == 2:\n",
    "                authors_str = f\"{authors_list[0]} and {authors_list[1]}\"\n",
    "            else:\n",
    "                authors_str = f\"{', '.join(authors_list[:-1])}, and {authors_list[-1]}\"\n",
    "    \n",
    "        caption = f\"{row['title']} by {authors_str}: {truncated_description}\"\n",
    "    \n",
    "        results.append((row['large_thumbnail'], caption))\n",
    "\n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "043aba89-e674-461a-940d-155e7efbe36c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "16\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[('http://books.google.com/books/content?id=ESS5HAAACAAJ&printsec=frontcover&img=1&zoom=1&source=gbs_api&wfife=w800',\n",
       "  'Maisie Dobbs by Jacqueline Winspear: A favorite mystery series of Hillary Clinton (as mentioned in What Happened, The New York Times Book Review, and New York Magazine) A New York Times Notable Book of the...'),\n",
       " ('http://books.google.com/books/content?id=V6aEAAAACAAJ&printsec=frontcover&img=1&zoom=1&source=gbs_api&wfife=w800',\n",
       "  \"The Mystery of the Disappearing Cat by Enid Blyton: A fantastic children’s crime story from the world’s best-loved children’s author, Enid Blyton. Lady Candling's best Siamese cat has gone missing. It's another mystery for the Find-Outers! The gardener, Luke,...\")]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "query = \"A book about mystery\"\n",
    "category = \"All\"\n",
    "tone = \"Surprising\"\n",
    "\n",
    "res = recommend_books(query, category, tone)\n",
    "print(len(res))\n",
    "res[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a45c1b5f-6b7c-462e-af9c-0f5d8f16a2f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "categories = [\"All\"] + sorted(books['simple_categories'].dropna().unique())\n",
    "tones = [\"All\"] + ['Happy', 'Surprising', 'Angry', 'Suspensful', 'Sad']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5be32682-2413-48be-a7ff-568b1633c595",
   "metadata": {},
   "source": [
    "# Start of the `gradio` dashboard"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "fc5a535c-6b4d-4103-9c93-2aa4b00d69b5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://0.0.0.0:7864\n",
      "\n",
      "To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://localhost:7864/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "with gr.Blocks(theme = gr.themes.Glass()) as dashboard:\n",
    "    gr.Markdown(\"# Semantic Book Recommender\")\n",
    "\n",
    "    with gr.Row():\n",
    "        user_query = gr.Textbox(label = \"What kind of book would you like to read?\",\n",
    "                                placeholder = \"e.g., A story about mystery\")\n",
    "        category_dropdown = gr.Dropdown(choices=categories, label=\"Select a category:\", value=\"All\")\n",
    "        tone_dropdown = gr.Dropdown(choices=tones, label=\"Select an emotional tone:\", value=\"All\")\n",
    "        submit_button = gr.Button(\"Find recommendations\")\n",
    "\n",
    "    gr.Markdown(\"## Recommendations\")\n",
    "    output = gr.Gallery(label=\"Recommended books\", columns=8, rows=2)\n",
    "\n",
    "    submit_button.click(fn=recommend_books,\n",
    "                        inputs=[user_query, category_dropdown, tone_dropdown],\n",
    "                        outputs=output)\n",
    "\n",
    "    # dashboard.launch(inline=True)\n",
    "    dashboard.launch(server_name=\"0.0.0.0\", server_port=7864)\n",
    "\n",
    "\n",
    "# ssh -J rhernand@labta.math.unipd.it rhernand@labsrv7.math.unipd.it -L 7864:localhost:7864 -N"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "510e9d40-6814-495b-8cea-7fe4b36706e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Closing server running on port: 7864\n"
     ]
    }
   ],
   "source": [
    "dashboard.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
