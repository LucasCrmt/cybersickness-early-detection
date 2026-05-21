# cybersickness-early-detection

## Installation des dépendances

### Environnement standard (CPU ou GPU CUDA)

Pour la plupart des utilisateurs (Linux, macOS, Windows sans GPU ou avec GPU NVIDIA compatible CUDA), utilisez l'environnement standard :

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install -r requirements.txt
```

### Environnement DirectML (GPU sous Windows, sans CUDA/WSL)

Pour utiliser le GPU sur Windows **sans** carte NVIDIA/CUDA ni WSL, utilisez DirectML :

- **Nécessite Windows**
- **Nécessite Python 3.10** (DirectML n'est pas compatible avec Python ≥3.11)

```bash
python -m venv .venv-directml
.venv-directml\Scripts\activate
pip install --upgrade pip
pip install -r requirements-directml.txt
```

#### Vérification de l'accès GPU (DirectML)

Après installation, vérifiez que TensorFlow détecte le GPU :

```bash
python cybersickness/verify_directml.py
```
Le script doit afficher une ou plusieurs lignes `GPU devices: [...]` avec votre matériel, et `Matmul ok, shape: ...`.

**Remarque :**
- DirectML fonctionne uniquement sous Windows.
- Si vous utilisez Jupyter/VS Code, sélectionnez le kernel Python associé à l'environnement `.venv-directml` pour profiter du GPU.

### Résumé des fichiers requirements

- `requirements.txt` : environnement standard (CPU ou GPU CUDA, multiplateforme)
- `requirements-directml.txt` : environnement GPU DirectML (Windows + Python 3.10 uniquement)