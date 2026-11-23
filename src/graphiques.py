"""Graphiques Matplotlib du projet.

Chaque fonction construit et retourne une ``Figure`` sans l'afficher : la même
figure peut être enregistrée en PNG (``enregistrer_figure``) ou intégrée dans
l'interface Tkinter. On n'utilise pas ``pyplot`` afin d'éviter les conflits de
fenêtres avec Tkinter.
"""

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

COULEUR_NON_FUMEUR = "#2a7ab0"
COULEUR_FUMEUR = "#d0603a"
COULEUR_PREDICTION = "#c0392b"


def figure_exploration(donnees: pd.DataFrame) -> Figure:
    """Graphique d'exploration : distribution des coûts et lien IMC / coût selon le tabagisme.

    Args:
        donnees: données nettoyées (colonnes ``charges``, ``bmi`` et ``smoker``).

    Returns:
        Une figure à deux panneaux.
    """
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    gauche, droite = figure.subplots(1, 2)
    groupes = [("no", "Non-fumeurs", COULEUR_NON_FUMEUR), ("yes", "Fumeurs", COULEUR_FUMEUR)]
    bords = np.linspace(0, donnees["charges"].max(), 40)

    for code, etiquette, couleur in groupes:
        sous_ensemble = donnees[donnees["smoker"] == code]
        gauche.hist(sous_ensemble["charges"], bins=bords, alpha=0.7, color=couleur,
                    label=f"{etiquette} (n = {len(sous_ensemble)})")
        droite.scatter(sous_ensemble["bmi"], sous_ensemble["charges"], s=12, alpha=0.6,
                       color=couleur, label=etiquette)

    gauche.set_title("Distribution des coûts médicaux annuels")
    gauche.set_xlabel("Coût annuel ($ US)")
    gauche.set_ylabel("Nombre de personnes")
    gauche.legend()

    droite.axvline(30, color="grey", linestyle="--", linewidth=1, label="IMC = 30 (obésité)")
    droite.set_title("Coût selon l'IMC et le tabagisme")
    droite.set_xlabel("Indice de masse corporelle (IMC)")
    droite.set_ylabel("Coût annuel ($ US)")
    droite.legend()
    return figure


def figure_predit_contre_reel(y_vrai: Sequence[float], y_predit: Sequence[float], nom_modele: str) -> Figure:
    """Nuage « prédit contre réel » sur l'ensemble de test.

    Les points proches de la diagonale sont bien prédits.
    """
    figure = Figure(figsize=(6, 5.5), layout="constrained")
    axe = figure.subplots()
    vrai = np.asarray(y_vrai, dtype=float)
    predit = np.asarray(y_predit, dtype=float)
    axe.scatter(vrai, predit, s=14, alpha=0.6, color=COULEUR_NON_FUMEUR, label="Personnes du jeu de test")
    limite = max(vrai.max(), predit.max()) * 1.05
    axe.plot([0, limite], [0, limite], color=COULEUR_PREDICTION, linestyle="--", label="Prédiction parfaite")
    axe.set_xlim(0, limite)
    axe.set_ylim(0, limite)
    axe.set_title(f"Prédit contre réel — {nom_modele}")
    axe.set_xlabel("Coût réel ($ US)")
    axe.set_ylabel("Coût prédit ($ US)")
    axe.legend()
    return figure


def figure_position_prediction(
    bords: Sequence[float],
    comptes: Sequence[int],
    prediction: float | None = None,
    fourchette: tuple[float, float] | None = None,
) -> Figure:
    """Crée la figure « position de la prédiction » (voir ``dessiner_position_prediction``)."""
    figure = Figure(figsize=(5.5, 4), layout="constrained")
    dessiner_position_prediction(figure, bords, comptes, prediction, fourchette)
    return figure


def dessiner_position_prediction(
    figure: Figure,
    bords: Sequence[float],
    comptes: Sequence[int],
    prediction: float | None = None,
    fourchette: tuple[float, float] | None = None,
) -> Figure:
    """(Re)dessine dans ``figure`` la distribution des coûts avec la prédiction en surimpression.

    La figure est vidée avant d'être redessinée : l'interface réutilise ainsi la
    même figure à chaque prédiction.

    Args:
        figure: figure Matplotlib à (re)dessiner.
        bords: bords des barres de l'histogramme (``len(comptes) + 1`` valeurs).
        comptes: nombre de personnes par barre.
        prediction: coût prédit à marquer d'une ligne verticale (optionnel).
        fourchette: bornes ``(bas, haut)`` à ombrer autour de la prédiction (optionnel).

    Returns:
        La même figure.
    """
    figure.clear()
    axe = figure.subplots()
    bords = np.asarray(bords, dtype=float)
    axe.bar(bords[:-1], comptes, width=np.diff(bords), align="edge", color=COULEUR_NON_FUMEUR,
            alpha=0.6, edgecolor="white", label="Personnes du jeu de données")
    if fourchette is not None:
        axe.axvspan(fourchette[0], fourchette[1], color=COULEUR_PREDICTION, alpha=0.15,
                    label="Fourchette probable")
    if prediction is not None:
        axe.axvline(prediction, color=COULEUR_PREDICTION, linewidth=2.5, label="Votre estimation")
    axe.set_title("Position de l'estimation parmi les coûts observés")
    axe.set_xlabel("Coût médical annuel ($ US)")
    axe.set_ylabel("Nombre de personnes")
    axe.legend()
    return figure


def figure_importance(importances: list[tuple[str, float]], libelles: dict[str, str]) -> Figure:
    """Barres horizontales de l'importance de chaque variable (par permutation).

    Args:
        importances: couples ``(variable, importance)`` déjà triés.
        libelles: libellés français des variables.
    """
    figure = Figure(figsize=(6, 4), layout="constrained")
    axe = figure.subplots()
    noms = [libelles.get(nom, nom) for nom, _ in importances][::-1]
    valeurs = [valeur for _, valeur in importances][::-1]
    axe.barh(noms, valeurs, color=COULEUR_NON_FUMEUR)
    axe.set_title("Importance des variables pour le modèle")
    axe.set_xlabel("Baisse du R² quand la variable est mélangée")
    axe.set_ylabel("Variable")
    return figure


def enregistrer_figure(figure: Figure, chemin: str) -> str:
    """Enregistre une figure en PNG (crée le dossier au besoin) et retourne son chemin."""
    os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
    figure.savefig(chemin, dpi=120)
    return chemin
