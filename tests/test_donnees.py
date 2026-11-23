"""Tests du chargement et du nettoyage des données."""

import os

import pandas as pd
import pytest

from donnees import JeuDeDonnees, chemin_projet, lire_csv_par_blocs, nombre_valeurs_aberrantes
from exceptions import DonneeInvalideError, FichierIntrouvableError

ENTETE = "age,sex,bmi,children,smoker,region,charges\n"


@pytest.fixture
def petit_csv(tmp_path) -> str:
    """CSV de 6 lignes : 1 doublon, 1 valeur manquante, 1 IMC impossible, 1 catégorie mal écrite."""
    contenu = ENTETE + (
        "19,female,27.9,0,yes,southwest,16884.92\n"
        "18,male,33.77,1,no,southeast,1725.55\n"
        "18,male,33.77,1,no,southeast,1725.55\n"
        "28,male,,3,no,southeast,4449.46\n"
        "33,male,500,0,no,northwest,21984.47\n"
        "32, Male ,28.88,0,NO,northwest,3866.86\n"
    )
    chemin = tmp_path / "mini.csv"
    chemin.write_text(contenu, encoding="utf-8")
    return str(chemin)


def test_fichier_absent_leve_une_erreur(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        JeuDeDonnees(str(tmp_path / "absent.csv")).charger()


def test_colonne_manquante_leve_une_erreur(tmp_path):
    chemin = tmp_path / "incomplet.csv"
    chemin.write_text("age,sex\n20,male\n", encoding="utf-8")
    with pytest.raises(DonneeInvalideError, match="Colonnes manquantes"):
        JeuDeDonnees(str(chemin)).charger()


def test_nettoyage_retire_doublons_manquants_et_impossibles(petit_csv):
    jeu = JeuDeDonnees(petit_csv).charger(taille_bloc=2)
    rapport = jeu.nettoyer()
    assert rapport["lignes_initiales"] == 6
    assert rapport["lignes_incompletes_retirees"] == 1
    assert rapport["doublons_retires"] == 1
    assert rapport["valeurs_impossibles_retirees"] == 1
    assert rapport["lignes_finales"] == len(jeu) == 3
    # « Male » / « NO » ont été normalisés.
    assert set(jeu.donnees["sex"]) == {"female", "male"}
    assert set(jeu.donnees["smoker"]) == {"yes", "no"}


def test_generateur_lit_par_blocs(petit_csv):
    blocs = list(lire_csv_par_blocs(petit_csv, taille_bloc=4))
    assert [len(bloc) for bloc in blocs] == [4, 2]


def test_generateur_taille_bloc_invalide(petit_csv):
    with pytest.raises(DonneeInvalideError):
        next(lire_csv_par_blocs(petit_csv, taille_bloc=0))


def test_donnees_non_chargees():
    jeu = JeuDeDonnees("inutile.csv")
    assert len(jeu) == 0
    with pytest.raises(DonneeInvalideError):
        _ = jeu.donnees


def test_vrai_jeu_respecte_les_criteres_de_l_enonce():
    jeu = JeuDeDonnees(chemin_projet("donnees", "assurance.csv")).charger()
    jeu.nettoyer()
    assert 500 <= len(jeu) <= 50_000
    assert len(jeu.caracteristiques) >= 5
    types = {d["type"] for d in jeu.decrire_variables()}
    assert types == {"numerique", "categorielle"}


def test_separation_reproductible_et_sans_chevauchement():
    jeu = JeuDeDonnees(chemin_projet("donnees", "assurance.csv")).charger()
    jeu.nettoyer()
    x_a, x_test_a, _, _ = jeu.separer(0.2, graine=42)
    x_b, x_test_b, _, _ = jeu.separer(0.2, graine=42)
    assert list(x_test_a.index) == list(x_test_b.index)
    assert set(x_a.index).isdisjoint(x_test_a.index)
    assert len(x_test_a) == pytest.approx(0.2 * len(jeu), abs=1)


@pytest.mark.parametrize("proportion", [0, 1, 1.5])
def test_separation_proportion_invalide(petit_csv, proportion):
    jeu = JeuDeDonnees(petit_csv).charger()
    with pytest.raises(DonneeInvalideError):
        jeu.separer(proportion)


def test_decrire_variables_intervalles(petit_csv):
    jeu = JeuDeDonnees(petit_csv).charger()
    jeu.nettoyer()
    age = next(d for d in jeu.decrire_variables() if d["nom"] == "age")
    assert (age["minimum"], age["maximum"], age["entier"]) == (18, 32, True)
    region = next(d for d in jeu.decrire_variables() if d["nom"] == "region")
    assert region["categories"]["southwest"] == "Sud-Ouest"


def test_valeurs_aberrantes():
    assert nombre_valeurs_aberrantes(pd.Series([1, 2, 3, 4, 5, 100])) == 1
    assert nombre_valeurs_aberrantes(pd.Series([5, 5, 5])) == 0


def test_chemin_projet_et_representations(petit_csv):
    assert os.path.isabs(chemin_projet("donnees"))
    jeu = JeuDeDonnees(petit_csv).charger()
    assert "6 lignes" in str(jeu)
    assert "JeuDeDonnees(" in repr(jeu)
