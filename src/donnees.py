"""Lecture, nettoyage et description du jeu de données sur les coûts d'assurance médicale."""

import os
from collections.abc import Iterator

import pandas as pd
from sklearn.model_selection import train_test_split

from exceptions import DonneeInvalideError, FichierIntrouvableError

# Dossier racine du projet (le dossier parent de src/), construit avec le module os.
DOSSIER_PROJET: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CIBLE: str = "charges"
COLONNES_NUMERIQUES: list[str] = ["age", "bmi", "children"]
COLONNES_CATEGORIELLES: list[str] = ["sex", "smoker", "region"]

# Libellés en français affichés dans l'interface (le modèle, lui, travaille sur les codes d'origine).
LIBELLES: dict[str, str] = {
    "age": "Âge (années)",
    "bmi": "Indice de masse corporelle (IMC)",
    "children": "Nombre d'enfants à charge",
    "sex": "Sexe",
    "smoker": "Fumeur",
    "region": "Région de résidence (États-Unis)",
    "charges": "Coût médical annuel ($ US)",
}
LIBELLES_CATEGORIES: dict[str, dict[str, str]] = {
    "sex": {"female": "Femme", "male": "Homme"},
    "smoker": {"no": "Non", "yes": "Oui"},
    "region": {
        "northeast": "Nord-Est",
        "northwest": "Nord-Ouest",
        "southeast": "Sud-Est",
        "southwest": "Sud-Ouest",
    },
}
# Variables numériques qui n'acceptent que des entiers.
COLONNES_ENTIERES: list[str] = ["age", "children"]

# Bornes physiquement plausibles : une ligne hors de ces bornes est une erreur de saisie.
BORNES_PLAUSIBLES: dict[str, tuple[float, float]] = {
    "age": (0, 120),
    "bmi": (10, 80),
    "children": (0, 20),
    "charges": (0, 1_000_000),
}


def chemin_projet(*parties: str) -> str:
    """Construit un chemin absolu à l'intérieur du dossier du projet.

    Args:
        *parties: morceaux du chemin relatif (ex. ``"donnees", "assurance.csv"``).

    Returns:
        Le chemin absolu correspondant, construit avec ``os.path.join``.
    """
    return os.path.join(DOSSIER_PROJET, *parties)


def lire_csv_par_blocs(chemin: str, taille_bloc: int = 500) -> Iterator[pd.DataFrame]:
    """Générateur qui lit un fichier CSV par blocs de lignes.

    Lire par blocs évite de charger un très gros fichier d'un coup en mémoire ;
    chaque bloc est produit avec ``yield`` au fur et à mesure.

    Args:
        chemin: chemin du fichier CSV.
        taille_bloc: nombre maximal de lignes par bloc (strictement positif).

    Yields:
        Un ``DataFrame`` pandas par bloc.

    Raises:
        FichierIntrouvableError: si le fichier n'existe pas.
        DonneeInvalideError: si ``taille_bloc`` n'est pas strictement positif.
    """
    if taille_bloc <= 0:
        raise DonneeInvalideError("La taille d'un bloc doit être strictement positive.")
    if not os.path.isfile(chemin):
        raise FichierIntrouvableError(f"Fichier de données introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        for bloc in pd.read_csv(fichier, chunksize=taille_bloc):
            yield bloc


def nombre_valeurs_aberrantes(serie: pd.Series) -> int:
    """Compte les valeurs aberrantes d'une série selon la règle des 1,5 × écart interquartile.

    Args:
        serie: série numérique.

    Returns:
        Le nombre de valeurs situées hors de [Q1 - 1,5·IQR ; Q3 + 1,5·IQR].
    """
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    ecart = q3 - q1
    masque = (serie < q1 - 1.5 * ecart) | (serie > q3 + 1.5 * ecart)
    return int(masque.sum())


class JeuDeDonnees:
    """Jeu de données tabulaire : chargement, nettoyage, description et séparation.

    Attributes:
        cible: nom de la colonne à prédire.
        colonnes_numeriques: noms des caractéristiques numériques.
        colonnes_categorielles: noms des caractéristiques catégorielles.
    """

    def __init__(
        self,
        chemin: str,
        cible: str = CIBLE,
        colonnes_numeriques: list[str] | None = None,
        colonnes_categorielles: list[str] | None = None,
    ) -> None:
        """Prépare le jeu de données sans le lire (la lecture se fait avec ``charger``).

        Args:
            chemin: chemin du fichier CSV.
            cible: colonne cible.
            colonnes_numeriques: caractéristiques numériques (défaut : celles du projet).
            colonnes_categorielles: caractéristiques catégorielles (défaut : celles du projet).
        """
        self._chemin = chemin
        self.cible = cible
        self.colonnes_numeriques = list(colonnes_numeriques or COLONNES_NUMERIQUES)
        self.colonnes_categorielles = list(colonnes_categorielles or COLONNES_CATEGORIELLES)
        self._donnees: pd.DataFrame | None = None

    @property
    def caracteristiques(self) -> list[str]:
        """Liste de toutes les caractéristiques (numériques puis catégorielles)."""
        return self.colonnes_numeriques + self.colonnes_categorielles

    @property
    def donnees(self) -> pd.DataFrame:
        """Copie des données chargées.

        Raises:
            DonneeInvalideError: si ``charger`` n'a pas encore été appelé.
        """
        if self._donnees is None:
            raise DonneeInvalideError("Les données n'ont pas encore été chargées.")
        return self._donnees.copy()

    def charger(self, taille_bloc: int = 500) -> "JeuDeDonnees":
        """Lit le CSV (par blocs, grâce au générateur) et vérifie les colonnes attendues.

        Args:
            taille_bloc: nombre de lignes lues à la fois.

        Returns:
            L'objet lui-même, pour pouvoir enchaîner ``charger().nettoyer()``.

        Raises:
            FichierIntrouvableError: si le fichier n'existe pas.
            DonneeInvalideError: si le fichier est vide ou s'il manque des colonnes.
        """
        blocs = list(lire_csv_par_blocs(self._chemin, taille_bloc))
        if not blocs:
            raise DonneeInvalideError("Le fichier de données est vide.")
        donnees = pd.concat(blocs, ignore_index=True)
        manquantes = [c for c in self.caracteristiques + [self.cible] if c not in donnees.columns]
        if manquantes:
            raise DonneeInvalideError(f"Colonnes manquantes dans le CSV : {', '.join(manquantes)}")
        self._donnees = donnees[self.caracteristiques + [self.cible]]
        return self

    def nettoyer(self) -> dict[str, int]:
        """Nettoie les données en place et décrit ce qui a été corrigé.

        Étapes : normalisation du texte des catégories (espaces, majuscules),
        conversion des numériques, suppression des lignes incomplètes, des doublons
        exacts et des valeurs physiquement impossibles. Les valeurs aberrantes
        statistiques de la cible sont seulement comptées, pas supprimées : ce sont
        de vrais cas (surtout des fumeurs) que le modèle doit apprendre.

        Returns:
            Un rapport ``{étape: nombre}`` (lignes initiales, retirées, finales, etc.).

        Raises:
            DonneeInvalideError: si les données ne sont pas chargées ou si le
                nettoyage ne laisse aucune ligne.
        """
        donnees = self.donnees
        rapport = {"lignes_initiales": len(donnees)}

        for colonne in self.colonnes_categorielles:
            donnees[colonne] = donnees[colonne].astype("string").str.strip().str.lower()
        for colonne in self.colonnes_numeriques + [self.cible]:
            donnees[colonne] = pd.to_numeric(donnees[colonne], errors="coerce")

        avant = len(donnees)
        donnees = donnees.dropna()
        rapport["lignes_incompletes_retirees"] = avant - len(donnees)

        avant = len(donnees)
        donnees = donnees.drop_duplicates()
        rapport["doublons_retires"] = avant - len(donnees)

        avant = len(donnees)
        for colonne, (minimum, maximum) in BORNES_PLAUSIBLES.items():
            if colonne in donnees.columns:
                donnees = donnees[donnees[colonne].between(minimum, maximum)]
        rapport["valeurs_impossibles_retirees"] = avant - len(donnees)

        if donnees.empty:
            raise DonneeInvalideError("Aucune ligne valide après le nettoyage.")

        for colonne in self.colonnes_categorielles:
            donnees[colonne] = donnees[colonne].astype(str)
        rapport["valeurs_aberrantes_cible_conservees"] = nombre_valeurs_aberrantes(donnees[self.cible])
        rapport["lignes_finales"] = len(donnees)
        self._donnees = donnees.reset_index(drop=True)
        return rapport

    def decrire_variables(self) -> list[dict]:
        """Décrit chaque caractéristique pour construire l'interface.

        Returns:
            Une liste de dictionnaires : pour une variable numérique, son intervalle
            observé ; pour une variable catégorielle, ses valeurs possibles et
            leurs libellés en français.
        """
        donnees = self.donnees
        descriptions: list[dict] = []
        for colonne in self.colonnes_numeriques:
            descriptions.append({
                "nom": colonne,
                "type": "numerique",
                "libelle": LIBELLES.get(colonne, colonne),
                "minimum": float(donnees[colonne].min()),
                "maximum": float(donnees[colonne].max()),
                "entier": colonne in COLONNES_ENTIERES,
            })
        for colonne in self.colonnes_categorielles:
            valeurs = sorted(donnees[colonne].unique())
            traductions = LIBELLES_CATEGORIES.get(colonne, {})
            descriptions.append({
                "nom": colonne,
                "type": "categorielle",
                "libelle": LIBELLES.get(colonne, colonne),
                "categories": {v: traductions.get(v, v) for v in valeurs},
            })
        return descriptions

    def separer(
        self, proportion_test: float = 0.2, graine: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Sépare les données en ensembles d'entraînement et de test.

        Args:
            proportion_test: part des lignes réservée au test (entre 0 et 1, exclus).
            graine: graine aléatoire fixée pour des résultats reproductibles.

        Returns:
            ``(X_entrainement, X_test, y_entrainement, y_test)``.

        Raises:
            DonneeInvalideError: si la proportion n'est pas strictement entre 0 et 1.
        """
        if not 0 < proportion_test < 1:
            raise DonneeInvalideError("La proportion de test doit être strictement entre 0 et 1.")
        donnees = self.donnees
        return train_test_split(
            donnees[self.caracteristiques],
            donnees[self.cible],
            test_size=proportion_test,
            random_state=graine,
        )

    def __len__(self) -> int:
        """Nombre de lignes chargées (0 si rien n'est chargé)."""
        return 0 if self._donnees is None else len(self._donnees)

    def __repr__(self) -> str:
        """Représentation technique, utile au débogage."""
        return f"JeuDeDonnees(chemin={self._chemin!r}, cible={self.cible!r}, lignes={len(self)})"

    def __str__(self) -> str:
        """Résumé lisible du jeu de données."""
        return (
            f"Jeu de données « {os.path.basename(self._chemin)} » : {len(self)} lignes, "
            f"{len(self.caracteristiques)} caractéristiques, cible « {self.cible} »"
        )
