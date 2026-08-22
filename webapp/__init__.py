"""
Warstwa GUI (Flask) dla MCDM Toolkit.

Implementuje interaktywnie przypadki uzycia opisane w pracy:
- PU1: Zarzadzanie problemem decyzyjnym i walidacja danych
- PU2: Definicja preferencji eksperckich (AHP -- porownania parami)
- PU4: Analiza wynikow i symulacja odwrocenia rankingu (rank reversal)

Architektura: Flask routes = warstwa Controller (deleguje do
MCDMController z rdzenia biblioteki), Jinja2 templates = warstwa
View, mcdm/ = warstwa Model. To ten sam podzial MVC, ktory jest
opisany w rozdziale o architekturze pracy inzynierskiej.
"""

from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    # UWAGA: sekret dev-owy. W realnym wdrozeniu nalezy pobierac go
    # ze zmiennej srodowiskowej (np. os.environ["FLASK_SECRET_KEY"]).
    # Sesja przechowuje wylacznie biezacy problem decyzyjny (mala
    # struktura JSON), wiec domyslny mechanizm ciasteczek Flask
    # (signed cookie, bez bazy danych) jest wystarczajacy dla
    # prototypu na potrzeby pracy inzynierskiej.
    app.secret_key = "dev-secret-key-change-in-production"

    # Udostepnia wbudowana funkcje zip() w szablonach Jinja2
    # (potrzebne do rownoleglego iterowania po kilku listach, np.
    # nazwach kryteriow i ich wagach, w problem_detail.html).
    app.jinja_env.globals.update(zip=zip)

    from .routes import bp

    app.register_blueprint(bp)

    return app
