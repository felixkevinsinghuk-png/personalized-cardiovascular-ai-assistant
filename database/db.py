"""
database/db.py
SQLAlchemy database object initialisation.

This module creates the shared `db` instance that is imported by both the
Flask application factory (app.py) and the ORM model definitions (models.py).
Keeping this in its own module avoids circular imports.
"""

from flask_sqlalchemy import SQLAlchemy

# Shared SQLAlchemy instance — bound to the Flask app in app.py via db.init_app(app)
db = SQLAlchemy()
