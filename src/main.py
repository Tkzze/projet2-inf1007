"""Point d'entrée du projet.

Utilisation (depuis le dossier du projet) ::

    python src/main.py              # lance l'interface (entraîne d'abord si aucun modèle n'existe)
    python src/main.py --entrainer  # ré-entraîne le modèle, affiche les résultats et quitte
"""

import argparse
import json
import logging
import sys

from donnees import chemin_projet
from entrainement import (CHEMIN_METADONNEES, CHEMIN_MODELE, charger_metadonnees,
                          executer_entrainement, resumer)
from exceptions import ErreurProjet, FichierIntrouvableError, ModeleNonEntraineError
from historique import Historique
from modele import ModeleAssurance

CHEMIN_HISTORIQUE: str = chemin_projet("historique", "predictions.csv")


def lire_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Analyse les options de la ligne de commande."""
    analyseur = argparse.ArgumentParser(description="Estimation du coût médical annuel (INF1007).")
    analyseur.add_argument("--entrainer", action="store_true",
                           help="ré-entraîner le modèle puis quitter sans ouvrir l'interface")
    return analyseur.parse_args(arguments)


def obtenir_modele() -> tuple[ModeleAssurance, dict]:
    """Charge le modèle et ses métadonnées ; entraîne une seule fois s'ils sont absents ou illisibles.

    Returns:
        ``(modèle, métadonnées)``.
    """
    try:
        return ModeleAssurance.charger(CHEMIN_MODELE), charger_metadonnees(CHEMIN_METADONNEES)
    except (FichierIntrouvableError, ModeleNonEntraineError, json.JSONDecodeError) as erreur:
        print(f"{erreur}\nEntraînement du modèle (quelques secondes)…")
        print(resumer(executer_entrainement()))
        return ModeleAssurance.charger(CHEMIN_MODELE), charger_metadonnees(CHEMIN_METADONNEES)


def main(arguments: list[str] | None = None) -> int:
    """Exécute le programme et retourne un code de sortie (0 = succès)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
    options = lire_arguments(arguments)
    try:
        if options.entrainer:
            print(resumer(executer_entrainement()))
            return 0
        modele, metadonnees = obtenir_modele()
        from interface import Application  # importé ici : Tkinter n'est requis que pour l'interface
        historique = Historique(CHEMIN_HISTORIQUE, modele.colonnes)
        Application(modele, metadonnees, historique).mainloop()
        return 0
    except ErreurProjet as erreur:
        print(f"Erreur : {erreur}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
