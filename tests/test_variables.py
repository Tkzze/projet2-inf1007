"""Tests des classes de variables (validation des saisies)."""

import pytest

from exceptions import DonneeInvalideError
from variables import (VariableCategorielle, VariableNumerique, creer_variable,
                       creer_variables_depuis, valider_saisies)


@pytest.fixture
def age() -> VariableNumerique:
    return VariableNumerique("age", "Âge", 18, 64, entier=True)


@pytest.fixture
def imc() -> VariableNumerique:
    return VariableNumerique("bmi", "IMC", 15.96, 53.13)


@pytest.fixture
def fumeur() -> VariableCategorielle:
    return VariableCategorielle("smoker", "Fumeur", {"no": "Non", "yes": "Oui"})


# --- cas usuels -------------------------------------------------------------

def test_numerique_valide_retourne_un_entier(age):
    assert age.valider("45") == 45
    assert isinstance(age.valider(" 45 "), int)


def test_numerique_accepte_la_virgule_decimale(imc):
    assert imc.valider("27,5") == pytest.approx(27.5)


def test_categorielle_accepte_libelle_ou_code(fumeur):
    assert fumeur.valider("Oui") == "yes"
    assert fumeur.valider("non") == "no"
    assert fumeur.valider("YES") == "yes"


# --- cas limites ------------------------------------------------------------

@pytest.mark.parametrize("saisie, attendu", [("18", 18), ("64", 64)])
def test_numerique_bornes_incluses(age, saisie, attendu):
    assert age.valider(saisie) == attendu


def test_numerique_entier_accepte_45_point_0(age):
    assert age.valider("45.0") == 45


# --- cas d'erreur -----------------------------------------------------------

@pytest.mark.parametrize("saisie", ["", "   ", None])
def test_saisie_vide_refusee(age, fumeur, saisie):
    with pytest.raises(DonneeInvalideError, match="obligatoire"):
        age.valider(saisie)
    with pytest.raises(DonneeInvalideError, match="obligatoire"):
        fumeur.valider(saisie)


@pytest.mark.parametrize("saisie", ["abc", "12a", "nan", "inf"])
def test_numerique_non_numerique_refuse(imc, saisie):
    with pytest.raises(DonneeInvalideError):
        imc.valider(saisie)


@pytest.mark.parametrize("saisie", ["17", "65", "-3"])
def test_numerique_hors_intervalle_refuse(age, saisie):
    with pytest.raises(DonneeInvalideError, match="hors de l'intervalle"):
        age.valider(saisie)


def test_numerique_entier_refuse_une_decimale(age):
    with pytest.raises(DonneeInvalideError, match="entier"):
        age.valider("30.5")


def test_categorielle_choix_inconnu_refuse(fumeur):
    with pytest.raises(DonneeInvalideError, match="pas un choix valide"):
        fumeur.valider("Peut-être")


def test_intervalle_inverse_refuse():
    with pytest.raises(DonneeInvalideError):
        VariableNumerique("x", "X", 10, 1)


# --- fabrique, polymorphisme, méthodes spéciales ----------------------------

def test_fabrique_cree_la_bonne_sous_classe(age, fumeur):
    description_age = {"nom": "age", "type": "numerique", "libelle": "Âge",
                       "minimum": 18, "maximum": 64, "entier": True}
    description_fumeur = {"nom": "smoker", "type": "categorielle", "libelle": "Fumeur",
                          "categories": {"no": "Non", "yes": "Oui"}}
    assert creer_variable(description_age) == age
    assert creer_variable(description_fumeur) == fumeur


def test_fabrique_type_inconnu_ou_cle_manquante():
    with pytest.raises(DonneeInvalideError, match="inconnu"):
        creer_variable({"nom": "x", "type": "date", "libelle": "X"})
    with pytest.raises(DonneeInvalideError, match="manquante"):
        creer_variable({"nom": "x", "type": "numerique", "libelle": "X"})
    with pytest.raises(DonneeInvalideError):
        creer_variables_depuis({})


def test_valider_saisies_regroupe_toutes_les_erreurs(age, imc, fumeur):
    with pytest.raises(DonneeInvalideError) as erreur:
        valider_saisies([age, imc, fumeur], {"age": "abc", "bmi": "", "smoker": "Oui"})
    message = str(erreur.value)
    assert "Âge" in message and "IMC" in message and "Fumeur" not in message


def test_valider_saisies_polymorphisme(age, imc, fumeur):
    valeurs = valider_saisies([age, imc, fumeur], {"age": "30", "bmi": "22,4", "smoker": "Non"})
    assert valeurs == {"age": 30, "bmi": pytest.approx(22.4), "smoker": "no"}


def test_str_et_eq(age, fumeur):
    assert str(age) == "Âge : nombre entier entre 18 et 64"
    assert age != fumeur
    assert age == VariableNumerique("age", "Âge", 18, 64, entier=True)
    assert "VariableNumerique" in repr(age)
