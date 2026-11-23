"""Exceptions personnalisées du projet.

Toutes héritent de ``ErreurProjet`` afin que le point d'entrée puisse les
intercepter d'un seul bloc ``except ErreurProjet`` sans jamais utiliser un
``except`` nu.
"""


class ErreurProjet(Exception):
    """Classe parente de toutes les erreurs propres au projet."""


class DonneeInvalideError(ErreurProjet):
    """Levée quand une donnée (saisie de l'utilisateur ou contenu du CSV) est invalide.

    Le message est rédigé pour être affiché tel quel à un utilisateur
    non informaticien.
    """


class FichierIntrouvableError(ErreurProjet):
    """Levée quand un fichier attendu (jeu de données, modèle, métadonnées) n'existe pas."""


class ModeleNonEntraineError(ErreurProjet):
    """Levée quand on tente d'utiliser un modèle qui n'a pas été entraîné ou chargé."""
