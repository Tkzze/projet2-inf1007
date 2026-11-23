"""Historique des prédictions, conservé dans un fichier CSV."""

import csv
import os
from datetime import datetime

from exceptions import DonneeInvalideError


class Historique:
    """Journal des prédictions faites dans l'interface.

    Chaque prédiction devient une ligne du CSV : date, valeurs saisies et résultat.
    """

    def __init__(self, chemin: str, colonnes: list[str]) -> None:
        """Prépare l'historique (le fichier est créé à la première prédiction).

        Args:
            chemin: chemin du fichier CSV.
            colonnes: noms des caractéristiques à enregistrer.
        """
        self._chemin = chemin
        self._entetes = ["date"] + list(colonnes) + ["prediction"]

    @property
    def chemin(self) -> str:
        """Chemin du fichier CSV."""
        return self._chemin

    @property
    def entetes(self) -> list[str]:
        """Colonnes du fichier CSV."""
        return list(self._entetes)

    @property
    def caracteristiques(self) -> list[str]:
        """Colonnes des valeurs saisies (toutes sauf la date et la prédiction)."""
        return self._entetes[1:-1]

    def derniere_entree(self) -> dict[str, str] | None:
        """Retourne la dernière ligne écrite dans le fichier, ou ``None`` s'il est vide ou absent."""
        lignes = self.lire(plus_recent_d_abord=False)
        return lignes[-1] if lignes else None

    def est_identique_a_la_derniere(self, valeurs: dict) -> bool:
        """Indique si ``valeurs`` sont exactement celles de la dernière prédiction enregistrée.

        Le CSV ne contient que du texte : on compare donc ``str(valeur)`` au texte
        relu (``35`` et ``"35"``, ``26.5`` et ``"26.5"`` sont considérés égaux).

        Args:
            valeurs: valeurs validées ``{caractéristique: valeur}``.

        Returns:
            ``True`` si toutes les caractéristiques sont identiques, ``False`` sinon
            (ou si l'historique est vide).
        """
        derniere = self.derniere_entree()
        if derniere is None:
            return False
        return all(str(valeurs.get(c)) == derniere.get(c) for c in self.caracteristiques)

    def ajouter(self, valeurs: dict, prediction: float, date: datetime | None = None) -> dict | None:
        """Ajoute une prédiction à la fin du fichier (crée le fichier et son en-tête au besoin).

        Si les valeurs sont identiques à celles de la dernière prédiction enregistrée,
        rien n'est écrit : refaire la même prédiction ne remplit pas l'historique.

        Args:
            valeurs: valeurs saisies ``{caractéristique: valeur}``.
            prediction: coût prédit.
            date: date de la prédiction (maintenant par défaut).

        Returns:
            La ligne enregistrée, ou ``None`` si elle n'a pas été ajoutée
            parce qu'elle répète la dernière prédiction.

        Raises:
            DonneeInvalideError: s'il manque une caractéristique dans ``valeurs``.
            OSError: si le fichier ne peut pas être écrit (disque plein, droits…).
        """
        manquantes = [c for c in self.caracteristiques if c not in valeurs]
        if manquantes:
            raise DonneeInvalideError(f"Valeurs manquantes pour l'historique : {', '.join(manquantes)}.")
        if self.est_identique_a_la_derniere(valeurs):
            return None
        ligne = {"date": (date or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")}
        ligne.update({c: valeurs[c] for c in self._entetes[1:-1]})
        ligne["prediction"] = round(float(prediction), 2)

        dossier = os.path.dirname(self._chemin)
        if dossier:
            os.makedirs(dossier, exist_ok=True)
        nouveau = not os.path.isfile(self._chemin)
        with open(self._chemin, "a", newline="", encoding="utf-8") as fichier:
            redacteur = csv.DictWriter(fichier, fieldnames=self._entetes)
            if nouveau:
                redacteur.writeheader()
            redacteur.writerow(ligne)
        return ligne

    def lire(self, plus_recent_d_abord: bool = True) -> list[dict[str, str]]:
        """Lit toutes les prédictions enregistrées.

        Args:
            plus_recent_d_abord: trie par date décroissante si ``True``.

        Returns:
            Liste de lignes (dictionnaires de chaînes) ; liste vide si le fichier
            n'existe pas encore.
        """
        if not os.path.isfile(self._chemin):
            return []
        with open(self._chemin, newline="", encoding="utf-8") as fichier:
            lignes = list(csv.DictReader(fichier))
        return sorted(lignes, key=lambda ligne: ligne["date"], reverse=plus_recent_d_abord)

    def __len__(self) -> int:
        """Nombre de prédictions enregistrées."""
        return len(self.lire())

    def __repr__(self) -> str:
        """Représentation technique."""
        return f"Historique(chemin={self._chemin!r})"
