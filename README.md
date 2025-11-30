# Mettre git bash dans le terminal

# Environnement uv
## Synchroniser l'environnement
uv sync

## Ajouter un package
uv add nom_package

# .gitignore
A copier dans .gitignore : 
# --------------------------
# Environnement virtuel
# --------------------------
uv/                  # Ignorer le dossier local
# uv.lock sera suivi, donc ne pas l'ignorer

# --------------------------
# Fichiers compilés Python
# --------------------------
__pycache__/
*.pyc
*.pyo
*.pyd

# --------------------------
# Fichiers système
# --------------------------
.DS_Store
Thumbs.db

# --------------------------
# IDE / éditeurs
# --------------------------
.vscode/
.idea/

# --------------------------
# Logs et fichiers temporaires
# --------------------------
*.log
*.tmp