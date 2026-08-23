"""
Punkt wejscia aplikacji webowej MCDM Toolkit.

Uruchomienie:
    cd mcdm-toolkit
    python app.py

Aplikacja bedzie dostepna pod http://127.0.0.1:5000
"""

from webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
