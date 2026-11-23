"""Modèle de régression : construction, entraînement, évaluation, prédiction et sauvegarde."""

import functools
import logging
import os
import pickle
import time
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError

journal = logging.getLogger(__name__)

TYPES_MODELES: dict[str, str] = {
    "reference": "Référence (moyenne des coûts)",
    "lineaire": "Régression linéaire",
    "foret": "Forêt aléatoire",
}

# Durée (en secondes) du dernier appel de chaque fonction chronométrée.
DUREES: dict[str, float] = {}


def chronometrer(fonction: Callable) -> Callable:
    """Décorateur qui mesure la durée d'exécution d'une fonction.

    La durée est enregistrée dans ``DUREES[nom de la fonction]`` et écrite dans
    le journal (``logging``) ; le résultat de la fonction est retourné tel quel.

    Args:
        fonction: fonction à chronométrer.

    Returns:
        La fonction enveloppée.
    """
    @functools.wraps(fonction)
    def enveloppe(*args: Any, **kwargs: Any) -> Any:
        debut = time.perf_counter()
        resultat = fonction(*args, **kwargs)
        DUREES[fonction.__qualname__] = time.perf_counter() - debut
        journal.info("%s exécutée en %.3f s", fonction.__qualname__, DUREES[fonction.__qualname__])
        return resultat
    return enveloppe


# --------------------------------------------------------------------------
# Métriques calculées « à la main » avec NumPy (vectorisées, sans boucle).
# --------------------------------------------------------------------------

def _verifier_vecteurs(y_vrai: Any, y_predit: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convertit deux séquences en tableaux NumPy et vérifie leur compatibilité.

    Raises:
        DonneeInvalideError: si les tableaux sont vides ou de tailles différentes.
    """
    vrai = np.asarray(y_vrai, dtype=float)
    predit = np.asarray(y_predit, dtype=float)
    if vrai.size == 0:
        raise DonneeInvalideError("Impossible de calculer une métrique sur des données vides.")
    if vrai.shape != predit.shape:
        raise DonneeInvalideError(
            f"Tailles différentes : {vrai.shape} valeurs réelles contre {predit.shape} prédictions."
        )
    return vrai, predit


def erreur_absolue_moyenne(y_vrai: Any, y_predit: Any) -> float:
    """MAE : moyenne des écarts absolus entre valeurs réelles et prédites.

    Args:
        y_vrai: valeurs réelles.
        y_predit: valeurs prédites (même taille).

    Returns:
        La MAE, dans l'unité de la cible (ici des dollars).

    Raises:
        DonneeInvalideError: si les entrées sont vides ou de tailles différentes.
    """
    vrai, predit = _verifier_vecteurs(y_vrai, y_predit)
    return float(np.mean(np.abs(vrai - predit)))


def racine_erreur_quadratique_moyenne(y_vrai: Any, y_predit: Any) -> float:
    """RMSE : racine de la moyenne des carrés des écarts (pénalise les grosses erreurs).

    Raises:
        DonneeInvalideError: si les entrées sont vides ou de tailles différentes.
    """
    vrai, predit = _verifier_vecteurs(y_vrai, y_predit)
    return float(np.sqrt(np.mean((vrai - predit) ** 2)))


def coefficient_determination(y_vrai: Any, y_predit: Any) -> float:
    """R² : part de la variance de la cible expliquée par le modèle (1 = parfait).

    Returns:
        Le R². Cas particulier : si toutes les valeurs réelles sont identiques,
        on retourne 1.0 si les prédictions sont exactes, sinon 0.0 (même
        convention que scikit-learn).

    Raises:
        DonneeInvalideError: si les entrées sont vides ou de tailles différentes.
    """
    vrai, predit = _verifier_vecteurs(y_vrai, y_predit)
    somme_residus = np.sum((vrai - predit) ** 2)
    somme_totale = np.sum((vrai - vrai.mean()) ** 2)
    if somme_totale == 0:
        return 1.0 if somme_residus == 0 else 0.0
    return float(1 - somme_residus / somme_totale)


def calculer_metriques(y_vrai: Any, y_predit: Any) -> dict[str, float]:
    """Regroupe les trois métriques de régression dans un dictionnaire."""
    return {
        "mae": erreur_absolue_moyenne(y_vrai, y_predit),
        "rmse": racine_erreur_quadratique_moyenne(y_vrai, y_predit),
        "r2": coefficient_determination(y_vrai, y_predit),
    }


def quantile_erreur_absolue(y_vrai: Any, y_predit: Any, quantile: float = 0.8) -> float:
    """Erreur absolue sous laquelle se trouvent ``quantile`` × 100 % des cas.

    Sert à donner à l'utilisateur une fourchette honnête autour de la prédiction.

    Raises:
        DonneeInvalideError: si le quantile n'est pas dans [0, 1] ou si les
            entrées sont invalides.
    """
    if not 0 <= quantile <= 1:
        raise DonneeInvalideError("Le quantile doit être compris entre 0 et 1.")
    vrai, predit = _verifier_vecteurs(y_vrai, y_predit)
    return float(np.quantile(np.abs(vrai - predit), quantile))


def calculer_fourchette(prediction: float, erreur: float) -> tuple[float, float]:
    """Fourchette ``[prediction - erreur ; prediction + erreur]`` bornée à 0 en bas.

    Un coût médical ne pouvant pas être négatif, la borne basse est ramenée à 0.

    Raises:
        DonneeInvalideError: si l'erreur est négative.
    """
    if erreur < 0:
        raise DonneeInvalideError("L'erreur utilisée pour la fourchette ne peut pas être négative.")
    return max(0.0, prediction - erreur), prediction + erreur


def formater_montant(montant: float) -> str:
    """Formate un montant à la française : ``12345.6`` → ``"12 346 $"``."""
    return f"{montant:,.0f}".replace(",", " ") + " $"


# --------------------------------------------------------------------------
# Classe du modèle
# --------------------------------------------------------------------------

class ModeleAssurance:
    """Modèle scikit-learn (prétraitement + régresseur) qui prédit le coût médical annuel.

    Attributes:
        type_modele: ``"reference"``, ``"lineaire"`` ou ``"foret"``.
    """

    def __init__(
        self,
        type_modele: str,
        colonnes_numeriques: list[str],
        colonnes_categorielles: list[str],
        graine: int = 42,
    ) -> None:
        """Construit le pipeline (non entraîné).

        Args:
            type_modele: clé de ``TYPES_MODELES``.
            colonnes_numeriques: caractéristiques numériques (passées telles quelles).
            colonnes_categorielles: caractéristiques catégorielles (encodées en one-hot).
            graine: graine aléatoire de la forêt, pour la reproductibilité.

        Raises:
            DonneeInvalideError: si le type de modèle est inconnu.
        """
        if type_modele not in TYPES_MODELES:
            raise DonneeInvalideError(f"Type de modèle inconnu : {type_modele!r}.")
        self.type_modele = type_modele
        self._colonnes = list(colonnes_numeriques) + list(colonnes_categorielles)
        self._pipeline = self._construire_pipeline(colonnes_numeriques, colonnes_categorielles, graine)
        self._est_entraine = False

    @staticmethod
    def _creer_regresseur(type_modele: str, graine: int) -> Any:
        """Retourne le régresseur scikit-learn correspondant au type demandé."""
        if type_modele == "reference":
            return DummyRegressor(strategy="mean")
        if type_modele == "lineaire":
            return LinearRegression()
        # Profondeur limitée pour éviter le surapprentissage sur ~1 000 lignes.
        return RandomForestRegressor(n_estimators=200, max_depth=5, random_state=graine)

    def _construire_pipeline(
        self, numeriques: list[str], categorielles: list[str], graine: int
    ) -> Pipeline:
        """Assemble l'encodage des catégories et le régresseur dans un seul objet."""
        preparation = ColumnTransformer([
            ("numeriques", "passthrough", list(numeriques)),
            ("categorielles", OneHotEncoder(handle_unknown="ignore"), list(categorielles)),
        ])
        return Pipeline([
            ("preparation", preparation),
            ("regresseur", self._creer_regresseur(self.type_modele, graine)),
        ])

    @property
    def nom(self) -> str:
        """Nom lisible du modèle (ex. ``"Forêt aléatoire"``)."""
        return TYPES_MODELES[self.type_modele]

    @property
    def est_entraine(self) -> bool:
        """``True`` une fois le modèle entraîné ou chargé."""
        return self._est_entraine

    @property
    def colonnes(self) -> list[str]:
        """Caractéristiques attendues, dans l'ordre."""
        return list(self._colonnes)

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline scikit-learn sous-jacent (utile pour l'importance des variables)."""
        return self._pipeline

    @chronometrer
    def entrainer(self, x: pd.DataFrame, y: pd.Series) -> "ModeleAssurance":
        """Entraîne le modèle (durée mesurée par le décorateur ``chronometrer``).

        Args:
            x: caractéristiques d'entraînement.
            y: cible d'entraînement.

        Returns:
            Le modèle lui-même.

        Raises:
            DonneeInvalideError: si ``x`` et ``y`` sont vides ou de tailles différentes.
        """
        if len(x) == 0 or len(x) != len(y):
            raise DonneeInvalideError("Données d'entraînement vides ou de tailles incohérentes.")
        self._pipeline.fit(x[self._colonnes], y)
        self._est_entraine = True
        return self

    def _verifier_entraine(self) -> None:
        """Lève ``ModeleNonEntraineError`` si le modèle n'est pas prêt."""
        if not self._est_entraine:
            raise ModeleNonEntraineError("Le modèle doit être entraîné ou chargé avant utilisation.")

    def predire_lot(self, x: pd.DataFrame) -> np.ndarray:
        """Prédit la cible pour plusieurs lignes.

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
        """
        self._verifier_entraine()
        return self._pipeline.predict(x[self._colonnes])

    def predire(self, valeurs: dict[str, float | int | str]) -> float:
        """Prédit le coût pour une seule personne.

        Args:
            valeurs: ``{caractéristique: valeur validée}``.

        Returns:
            Le coût annuel prédit (en dollars).

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
            DonneeInvalideError: s'il manque une caractéristique.
        """
        self._verifier_entraine()
        manquantes = [c for c in self._colonnes if c not in valeurs]
        if manquantes:
            raise DonneeInvalideError(f"Valeurs manquantes : {', '.join(manquantes)}.")
        ligne = pd.DataFrame([{c: valeurs[c] for c in self._colonnes}])
        return float(self.predire_lot(ligne)[0])

    def evaluer(self, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Calcule MAE, RMSE et R² (avec NumPy) sur un ensemble jamais vu à l'entraînement.

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
        """
        return calculer_metriques(y, self.predire_lot(x))

    def sauvegarder(self, chemin: str) -> None:
        """Enregistre le modèle entraîné avec joblib.

        Raises:
            ModeleNonEntraineError: si le modèle n'est pas entraîné.
        """
        self._verifier_entraine()
        os.makedirs(os.path.dirname(chemin) or ".", exist_ok=True)
        joblib.dump(self, chemin)

    @staticmethod
    def charger(chemin: str) -> "ModeleAssurance":
        """Charge un modèle sauvegardé avec ``sauvegarder``.

        Raises:
            FichierIntrouvableError: si le fichier n'existe pas.
            ModeleNonEntraineError: si le fichier est illisible (corrompu ou créé
                avec une version incompatible de scikit-learn).
        """
        if not os.path.isfile(chemin):
            raise FichierIntrouvableError(f"Modèle introuvable : {chemin}")
        try:
            modele = joblib.load(chemin)
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError, ValueError) as erreur:
            raise ModeleNonEntraineError(f"Modèle illisible ({erreur}) : il faut le ré-entraîner.") from None
        if not isinstance(modele, ModeleAssurance):
            raise ModeleNonEntraineError("Le fichier ne contient pas un ModeleAssurance.")
        return modele

    def __repr__(self) -> str:
        """Représentation technique."""
        return f"ModeleAssurance(type_modele={self.type_modele!r}, entraine={self._est_entraine})"

    def __str__(self) -> str:
        """Nom lisible et état du modèle."""
        etat = "entraîné" if self._est_entraine else "non entraîné"
        return f"{self.nom} ({etat})"


def trier_importances(importances: dict[str, float]) -> list[tuple[str, float]]:
    """Trie les importances des variables de la plus forte à la plus faible.

    Args:
        importances: ``{variable: importance}``.

    Returns:
        Liste de couples ``(variable, importance)`` triée par importance décroissante
        (clé de tri donnée par une fonction lambda).
    """
    return sorted(importances.items(), key=lambda couple: couple[1], reverse=True)
