from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', all_books=list(book_pivot.index))

@app.route('/recommend', methods=['POST'])
def recommend():
    book_name = request.form['book_name']
    if book_name not in book_pivot.index:
        return render_template('index.html', recommendations=[f"Book '{book_name}' not found. Try another."])
    # Fetch index of the input book
    book_index = np.where(book_pivot.index == book_name)[0][0]
    # Find similar books
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
    recommended_books = []
    for i in range(1, len(suggestions[0])):
        recommended_books.append(book_pivot.index[suggestions[0][i]])
    return render_template('index.html', recommendations=recommended_books, all_books=list(book_pivot.index))


if __name__ == '__main__':
    app.run(debug=True)
