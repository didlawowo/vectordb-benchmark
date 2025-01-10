# Milvus-Qdrant Benchmark

Ce repository contient un ensemble d'outils et de scripts pour effectuer des benchmarks comparatifs entre les bases de données vectorielles Milvus et Qdrant en utilisant Locust comme outil de test de charge.

## Structure du Projet



### Données (`/data`)

- `dataset.csv` : Jeu de données source
- `questions.txt` : Liste des requêtes de test
- `testset.json` : Jeu de données de test formaté

### generer le dataset depuis l'archives

```shell
gzcat data/output_dataset.jsonl.gz > data/output_dataset.jsonl
```

### Configuration Milvus (`/milvus-local`)

- `docker-compose.yaml` : Configuration Docker pour déployer Milvus localement

### Scripts de Préparation

- `prepare_custom_query.py` : Génération de requêtes personnalisées transform le question txt en testset.json
- `prepare_dataset.py` : Préparation et transformation du jeu de données transforme le csv en jsonl
- `prepare_milvus_db.py` : Initialisation et configuration de la base Milvus
- `benchmark_locust_milvus.py` : Script de benchmark Locust pour Milvus

### Configuration et Dépendances

- `Pipfile` et `Pipfile.lock` : Gestion des dépendances Python avec Pipenv
- `pyproject.toml` : Configuration du projet Python
- `common.py` : Constantes et fonctions communes

## Installation

### prérequis

- python3.11
- pip
- pipenv

----

1. Installer les dépendances :

   ```bash
   pipenv install
   ```

## Configuration de Milvus

### Démarrage de Milvus avec Docker Compose



1. Démarrer les services Milvus :

   ```bash
   cd milvus-local
   docker-compose up -d
   ```

2. Vérifier que tous les services sont démarrés :

   ```bash
   docker-compose ps
   ```

3. Initialiser la base de données :

### Configuration des Variables d'Environnement

```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
```

   ```bash
   python prepare_milvus_db.py init
   ```

## Tests de Performance avec Locust

### Configuration de Locust

1. Structure du script de test (`benchmark_locust_milvus.py`) :

   ```python
   from locust import HttpUser, task, between
   
   class MilvusUser(HttpUser):
       wait_time = between(1, 2)
       
       @task
       def search_vectors(self):
           # Définition des tâches de test
   ```

2. Types de tests implémentés :
   - Recherche de vecteurs denses
   - Recherche de vecteurs sparse
   - Recherche hybride
   - Insertions en masse
   - Requêtes de métadonnées

### Exécution des Tests

1. Démarrer Locust pour Milvus :

   ```bash
   locust -f benchmark_locust_milvus.py --host http://localhost:19530
   ```

2. Accéder à l'interface Web de Locust :
   - Ouvrir `http://localhost:8089` dans un navigateur
   - Configurer le test :
     - Number of users : nombre d'utilisateurs simulés
     - Spawn rate : taux de création des utilisateurs
     - Host : URL de l'instance Milvus

3. Métriques disponibles :
   - Temps de réponse (min, max, moyenne)
   - Nombre de requêtes par seconde
   - Taux d'erreur
   - Distribution des temps de réponse

### Personnalisation des Tests

Pour modifier les scénarios de test :

1. Éditer les requêtes dans `questions.txt`
2. Ajuster les paramètres dans `common.py` :

   ```python
   DIMENSION = 768  # Dimension des vecteurs
   COLLECTION_NAME = "benchmark_collection"
   DATABASE_NAME = "benchmark_db"
   ```

3. Modifier les patterns de charge dans `benchmark_locust_milvus.py` :

   ```python
   @task(3)  # Poids relatif de la tâche
   def search_vectors(self):
       # Configuration de la recherche
   ```

### Analyse des Résultats

Locust génère des rapports détaillés au format HTML et CSV, incluant :

- Graphiques de performance en temps réel
- Statistiques par endpoint
- Distribution des temps de réponse
- Logs d'erreurs

Les rapports sont sauvegardés dans le dossier `locust-reports/` après chaque session de test.

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à proposer une pull request pour améliorer les benchmarks ou ajouter de nouveaux tests.

## Licence

[License à définir]
