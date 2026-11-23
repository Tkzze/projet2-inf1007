"""Classes de variables d'entrée : chacune sait valider la saisie de l'utilisateur.

Hiérarchie (héritage et polymorphisme) ::

    Variable (abstraite)
    ├── VariableNumerique     -> valide un nombre dans un intervalle
    └── VariableCategorielle  -> valide un choix dans une liste
"""

import math
from abc import ABC, abstractmethod

from exceptions import DonneeInvalideError


class Variable(ABC):
    """Variable d'entrée du modèle (classe abstraite).

    Attributes:
        nom: nom de la colonne dans le jeu de données (ex. ``"age"``).
        libelle: texte affiché à l'utilisateur (ex. ``"Âge (années)"``).
    """

    def __init__(self, nom: str, libelle: str) -> None:
        """Initialise les attributs communs à toutes les variables.

        Args:
            nom: nom de la colonne.
            libelle: libellé affiché dans l'interface.
        """
        self._nom = nom
        self._libelle = libelle

    @property
    def nom(self) -> str:
        """Nom de la colonne (lecture seule)."""
        return self._nom

    @property
    def libelle(self) -> str:
        """Libellé affiché (lecture seule)."""
        return self._libelle

    @abstractmethod
    def valider(self, saisie: str) -> float | int | str:
        """Convertit et valide une saisie brute de l'utilisateur.

        Args:
            saisie: texte tapé ou choisi dans l'interface.

        Returns:
            La valeur prête à être donnée au modèle.

        Raises:
            DonneeInvalideError: si la saisie est vide ou invalide.
        """

    @abstractmethod
    def decrire_contrainte(self) -> str:
        """Texte d'aide décrivant les valeurs acceptées (affiché sous le champ)."""

    def _verifier_non_vide(self, saisie: str | None) -> str:
        """Retire les espaces autour de la saisie et refuse une saisie vide.

        Args:
            saisie: texte brut.

        Returns:
            Le texte sans espaces autour.

        Raises:
            DonneeInvalideError: si la saisie est vide.
        """
        texte = "" if saisie is None else str(saisie).strip()
        if not texte:
            raise DonneeInvalideError(f"« {self._libelle} » : ce champ est obligatoire.")
        return texte

    def __eq__(self, autre: object) -> bool:
        """Deux variables sont égales si elles ont le même type et les mêmes attributs."""
        return type(self) is type(autre) and vars(self) == vars(autre)

    def __hash__(self) -> int:
        """Hash cohérent avec ``__eq__`` (basé sur le type et le nom)."""
        return hash((type(self).__name__, self._nom))

    def __str__(self) -> str:
        """Libellé suivi de la contrainte, ex. ``Âge (années) : entier entre 18 et 64``."""
        return f"{self._libelle} : {self.decrire_contrainte()}"


class VariableNumerique(Variable):
    """Variable numérique bornée (entière ou réelle)."""

    def __init__(
        self, nom: str, libelle: str, minimum: float, maximum: float, entier: bool = False
    ) -> None:
        """Crée une variable numérique.

        Args:
            nom: nom de la colonne.
            libelle: libellé affiché.
            minimum: plus petite valeur acceptée (incluse).
            maximum: plus grande valeur acceptée (incluse).
            entier: ``True`` si seules les valeurs entières sont acceptées.

        Raises:
            DonneeInvalideError: si ``minimum > maximum``.
        """
        super().__init__(nom, libelle)
        if minimum > maximum:
            raise DonneeInvalideError(f"Intervalle invalide pour « {libelle} » : {minimum} > {maximum}.")
        self._minimum = minimum
        self._maximum = maximum
        self._entier = entier

    @property
    def minimum(self) -> float:
        """Borne inférieure acceptée."""
        return self._minimum

    @property
    def maximum(self) -> float:
        """Borne supérieure acceptée."""
        return self._maximum

    def _formater(self, valeur: float) -> str:
        """Formate une borne pour l'affichage (sans décimales inutiles, virgule française)."""
        if float(valeur).is_integer():
            return str(int(valeur))
        return f"{valeur:.2f}".replace(".", ",")

    def decrire_contrainte(self) -> str:
        """Ex. ``nombre entier entre 18 et 64``."""
        genre = "nombre entier" if self._entier else "nombre"
        return f"{genre} entre {self._formater(self._minimum)} et {self._formater(self._maximum)}"

    def valider(self, saisie: str) -> float | int:
        """Convertit la saisie en nombre et vérifie qu'il est dans l'intervalle.

        La virgule décimale française (``"27,5"``) est acceptée.

        Args:
            saisie: texte tapé par l'utilisateur.

        Returns:
            Un ``int`` si la variable est entière, sinon un ``float``.

        Raises:
            DonneeInvalideError: si la saisie est vide, non numérique, non entière
                alors qu'un entier est attendu, ou hors intervalle.
        """
        texte = self._verifier_non_vide(saisie).replace(",", ".")
        try:
            valeur = float(texte)
        except ValueError:
            raise DonneeInvalideError(
                f"« {self._libelle} » : « {saisie} » n'est pas un nombre valide."
            ) from None
        if not math.isfinite(valeur):
            raise DonneeInvalideError(f"« {self._libelle} » : la valeur doit être un nombre fini.")
        if self._entier and not valeur.is_integer():
            raise DonneeInvalideError(f"« {self._libelle} » : un nombre entier est attendu.")
        if not self._minimum <= valeur <= self._maximum:
            raise DonneeInvalideError(
                f"« {self._libelle} » : {self._formater(valeur)} est hors de l'intervalle accepté "
                f"({self.decrire_contrainte()})."
            )
        return int(valeur) if self._entier else valeur

    def __repr__(self) -> str:
        """Représentation technique."""
        return (
            f"VariableNumerique(nom={self._nom!r}, minimum={self._minimum}, "
            f"maximum={self._maximum}, entier={self._entier})"
        )


class VariableCategorielle(Variable):
    """Variable catégorielle : la valeur doit faire partie d'une liste fermée."""

    def __init__(self, nom: str, libelle: str, categories: dict[str, str]) -> None:
        """Crée une variable catégorielle.

        Args:
            nom: nom de la colonne.
            libelle: libellé affiché.
            categories: dictionnaire ``{code du jeu de données: libellé en français}``,
                ex. ``{"yes": "Oui", "no": "Non"}``.

        Raises:
            DonneeInvalideError: si aucune catégorie n'est fournie.
        """
        super().__init__(nom, libelle)
        if not categories:
            raise DonneeInvalideError(f"Aucune catégorie définie pour « {libelle} ».")
        self._categories = dict(categories)

    @property
    def libelles(self) -> list[str]:
        """Libellés proposés dans la liste déroulante, dans l'ordre des codes."""
        return list(self._categories.values())

    def decrire_contrainte(self) -> str:
        """Ex. ``choisir parmi : Non, Oui``."""
        return "choisir parmi : " + ", ".join(self.libelles)

    def libelle_de(self, code: str) -> str:
        """Retourne le libellé français d'un code (ou le code s'il est inconnu)."""
        return self._categories.get(code, code)

    def valider(self, saisie: str) -> str:
        """Vérifie que la saisie est une catégorie connue et retourne son code.

        Le libellé français (``"Oui"``) comme le code d'origine (``"yes"``) sont
        acceptés, sans tenir compte des majuscules.

        Args:
            saisie: libellé choisi dans la liste déroulante.

        Returns:
            Le code de la catégorie, tel qu'attendu par le modèle.

        Raises:
            DonneeInvalideError: si la saisie est vide ou ne correspond à aucune catégorie.
        """
        texte = self._verifier_non_vide(saisie).lower()
        for code, libelle in self._categories.items():
            if texte in (code.lower(), libelle.lower()):
                return code
        raise DonneeInvalideError(
            f"« {self._libelle} » : « {saisie} » n'est pas un choix valide ({self.decrire_contrainte()})."
        )

    def __repr__(self) -> str:
        """Représentation technique."""
        return f"VariableCategorielle(nom={self._nom!r}, categories={self._categories!r})"


def creer_variable(description: dict) -> Variable:
    """Fabrique la bonne sous-classe de ``Variable`` à partir de sa description JSON.

    Args:
        description: dictionnaire issu des métadonnées du modèle (clé ``"type"``
            valant ``"numerique"`` ou ``"categorielle"``).

    Returns:
        Une ``VariableNumerique`` ou une ``VariableCategorielle``.

    Raises:
        DonneeInvalideError: si le type est inconnu ou si une clé obligatoire manque.
    """
    try:
        type_variable = description["type"]
        if type_variable == "numerique":
            return VariableNumerique(
                description["nom"],
                description["libelle"],
                description["minimum"],
                description["maximum"],
                description.get("entier", False),
            )
        if type_variable == "categorielle":
            return VariableCategorielle(description["nom"], description["libelle"], description["categories"])
    except KeyError as erreur:
        raise DonneeInvalideError(f"Métadonnées incomplètes : clé {erreur} manquante.") from None
    raise DonneeInvalideError(f"Type de variable inconnu : {type_variable!r}.")


def creer_variables_depuis(metadonnees: dict) -> list[Variable]:
    """Crée la liste des variables d'entrée décrites dans les métadonnées du modèle.

    Args:
        metadonnees: dictionnaire lu dans ``metadonnees.json`` (clé ``"variables"``).

    Returns:
        Les variables, dans l'ordre des métadonnées.

    Raises:
        DonneeInvalideError: si la clé ``"variables"`` manque ou si une description est invalide.
    """
    if "variables" not in metadonnees:
        raise DonneeInvalideError("Métadonnées incomplètes : clé 'variables' manquante.")
    return [creer_variable(description) for description in metadonnees["variables"]]


def valider_saisies(variables: list[Variable], saisies: dict[str, str]) -> dict[str, float | int | str]:
    """Valide toutes les saisies d'un formulaire et regroupe les messages d'erreur.

    Grâce au polymorphisme, chaque variable valide sa saisie à sa façon, via le
    même appel ``variable.valider(...)``.

    Args:
        variables: variables du modèle.
        saisies: ``{nom de variable: texte saisi}``.

    Returns:
        ``{nom de variable: valeur validée}`` prêt pour la prédiction.

    Raises:
        DonneeInvalideError: si au moins une saisie est invalide ; le message
            contient une ligne par problème.
    """
    valeurs: dict[str, float | int | str] = {}
    erreurs: list[str] = []
    for variable in variables:
        try:
            valeurs[variable.nom] = variable.valider(saisies.get(variable.nom, ""))
        except DonneeInvalideError as erreur:
            erreurs.append(str(erreur))
    if erreurs:
        raise DonneeInvalideError("\n".join(erreurs))
    return valeurs
