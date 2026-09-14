# Shared SQLAlchemy instance — bound to the app in app.py via db.init_app(app).
# Kept in its own file to avoid circular imports between models.py and app.py.

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
