"""Démarche complète d'entraînement : données → modèles → évaluation → sauvegarde.

Exécutée une seule fois (``python src/main.py --entrainer``) ; l'interface se
contente ensuite de charger le modèle et ses métadonnées.
"""

import json
import os
from datetime import datetime

import numpy as np
import sklearn
from sklearn.inspection import permutation_importance

import graphiques
from donnees import (COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, LIBELLES, CIBLE,
                     JeuDeDonnees, chemin_projet)
from exceptions import FichierIntrouvableError
from modele import (DUREES, ModeleAssurance, quantile_erreur_absolue, trier_importances)

GRAINE: int = 42
PROPORTION_TEST: float = 0.2
METRIQUE_PRINCIPALE: str = "mae"

CHEMIN_DONNEES: str = chemin_projet("donnees", "assurance.csv")
CHEMIN_MODELE: str = chemin_projet("modeles", "modele.joblib")
CHEMIN_METADONNEES: str = chemin_projet("modeles", "metadonnees.json")
DOSSIER_FIGURES: str = chemin_projet("figures")


def sauvegarder_metadonnees(metadonnees: dict, chemin: str) -> None:
    """Écrit les métadonnées du modèle en JSON (UTF-8, indenté).

    Raises:
        OSError: si le fichier ne peut pas être écrit.
    """
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)


def charger_metadonnees(chemin: str) -> dict:
    """Lit les métadonnées JSON du modèle.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
        json.JSONDecodeError: si le fichier n'est pas un JSON valide.
    """
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Métadonnées introuvables : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        return json.load(fichier)


def executer_entrainement(
    chemin_donnees: str = CHEMIN_DONNEES,
    chemin_modele: str = CHEMIN_MODELE,
    chemin_metadonnees: str = CHEMIN_METADONNEES,
    dossier_figures: str = DOSSIER_FIGURES,
) -> dict:
    """Charge et nettoie les données, entraîne et compare les modèles, sauvegarde le meilleur.

    Args:
        chemin_donnees: CSV du jeu de données.
        chemin_modele: fichier ``.joblib`` à écrire.
        chemin_metadonnees: fichier JSON à écrire.
        dossier_figures: dossier où enregistrer les graphiques PNG.

    Returns:
        Les métadonnées enregistrées (elles contiennent toutes les métriques).

    Raises:
        FichierIntrouvableError, DonneeInvalideError: si les données sont absentes ou invalides.
    """
    # 1. Explorer et nettoyer
    jeu = JeuDeDonnees(chemin_donnees).charger()
    rapport_nettoyage = jeu.nettoyer()
    graphiques.enregistrer_figure(graphiques.figure_exploration(jeu.donnees),
                                  os.path.join(dossier_figures, "exploration.png"))

    # 2-3. Séparer (graine fixée) : le test ne sert jamais à l'entraînement.
    x_entr, x_test, y_entr, y_test = jeu.separer(PROPORTION_TEST, GRAINE)

    # 4-5. Référence triviale puis deux vrais modèles, tous évalués sur le test.
    modeles: dict[str, ModeleAssurance] = {}
    metriques: dict[str, dict[str, float]] = {}
    durees: dict[str, float] = {}
    for type_modele in ("reference", "lineaire", "foret"):
        modele = ModeleAssurance(type_modele, COLONNES_NUMERIQUES, COLONNES_CATEGORIELLES, GRAINE)
        modele.entrainer(x_entr, y_entr)
        durees[type_modele] = DUREES["ModeleAssurance.entrainer"]
        modeles[type_modele] = modele
        metriques[type_modele] = modele.evaluer(x_test, y_test)

    candidats = [t for t in modeles if t != "reference"]
    retenu = min(candidats, key=lambda t: metriques[t][METRIQUE_PRINCIPALE])
    meilleur = modeles[retenu]
    predictions_test = meilleur.predire_lot(x_test)

    graphiques.enregistrer_figure(
        graphiques.figure_predit_contre_reel(y_test, predictions_test, meilleur.nom),
        os.path.join(dossier_figures, "predit_contre_reel.png"))

    resultat_permutation = permutation_importance(
        meilleur.pipeline, x_test[meilleur.colonnes], y_test,
        n_repeats=10, random_state=GRAINE, scoring="r2")
    importances = dict(zip(meilleur.colonnes, map(float, resultat_permutation.importances_mean)))
    importances_triees = trier_importances(importances)
    graphiques.enregistrer_figure(graphiques.figure_importance(importances_triees, LIBELLES),
                                  os.path.join(dossier_figures, "importance.png"))

    # 6. Sauvegarder le modèle et ses métadonnées (l'interface se construit à partir de ces dernières).
    meilleur.sauvegarder(chemin_modele)
    comptes, bords = np.histogram(y_entr, bins=30)
    metadonnees = {
        "date_entrainement": datetime.now().isoformat(timespec="seconds"),
        "version_scikit_learn": sklearn.__version__,
        "source": "Kaggle — Medical Cost Personal Datasets (insurance.csv)",
        "cible": CIBLE,
        "libelle_cible": LIBELLES[CIBLE],
        "colonnes": meilleur.colonnes,
        "variables": jeu.decrire_variables(),
        "nettoyage": rapport_nettoyage,
        "graine": GRAINE,
        "proportion_test": PROPORTION_TEST,
        "lignes_entrainement": len(x_entr),
        "lignes_test": len(x_test),
        "metrique_principale": METRIQUE_PRINCIPALE,
        "modele_retenu": retenu,
        "nom_modele_retenu": meilleur.nom,
        "metriques": metriques,
        "durees_entrainement_s": durees,
        "erreur_absolue_80": quantile_erreur_absolue(y_test, predictions_test, 0.8),
        "importances": dict(importances_triees),
        "distribution_cible": {"bords": bords.tolist(), "comptes": comptes.tolist()},
    }
    sauvegarder_metadonnees(metadonnees, chemin_metadonnees)
    return metadonnees


def resumer(metadonnees: dict) -> str:
    """Texte récapitulatif des résultats, affiché par ``main.py`` après l'entraînement."""
    lignes = [f"Nettoyage : {metadonnees['nettoyage']}",
              f"Entraînement : {metadonnees['lignes_entrainement']} lignes, test : {metadonnees['lignes_test']} lignes",
              "Résultats sur l'ensemble de test :"]
    for type_modele, valeurs in metadonnees["metriques"].items():
        lignes.append(f"  {type_modele:<10} MAE = {valeurs['mae']:>9.0f} $   RMSE = {valeurs['rmse']:>9.0f} $"
                      f"   R² = {valeurs['r2']:.3f}")
    lignes.append(f"Modèle retenu : {metadonnees['nom_modele_retenu']}")
    lignes.append(f"80 % des erreurs de test sont inférieures à {metadonnees['erreur_absolue_80']:.0f} $")
    return "\n".join(lignes)
