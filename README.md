Mettre git bash dans le terminal
Aller dans le dossier où il y a uv.lock et pyproject.toml

# 1. Créer environnement virtuel local
python -m venv venv
# 2. Activation de l'environnement
source venv/Scripts/activate
# 3. Synchroniser l'environnement uv
uv sync
# Pour ajouter un package si besoin
uv add nom_package

# .gitignore si pas présent lors de l'import
git add .gitignore
git commit -m "Ajout du .gitignore"
git push

A copier dans .gitignore : 
# Environnement virtuel
venv/
uv/                  # Ignorer le dossier local
# uv.lock sera suivi, donc ne pas l'ignorer
# Fichiers compilés Python
__pycache__/
*.pyc
*.pyo
*.pyd
# Fichiers système
.DS_Store
Thumbs.db
# IDE / éditeurs
.vscode/
.idea/
# Logs et fichiers temporaires
*.log
*.tmp
.ipynb_checkpoints/