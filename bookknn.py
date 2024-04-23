import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

filter_1 = df_ratings['user'].value_counts()
filter_2 = df_ratings['isbn'].value_counts()
df_ratings = df_ratings[~df_ratings['user'].isin(filter_1[filter_1 < 200].index) & ~df_ratings['isbn'].isin(filter_2[filter_2 < 100].index)]

df_table = df_ratings.pivot_table(index='isbn', columns='user', values='rating').fillna(0)
df_table

df_table.index = df_table.join(df_books.set_index('isbn'))['title']
df_table

def get_recommends(book = ""):
  recommended_books = []
  nbrs = NearestNeighbors(n_neighbors=6, metric="cosine").fit(df_table.values)
  distances, indices = nbrs.kneighbors([df_table.loc[book].values], n_neighbors=6)
  for i in range(1,6):
    recommended_books.append([df_table.index[indices[0][-i]], distances[0][-i]])

  return [book, recommended_books]

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
