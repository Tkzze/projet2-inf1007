"""Tests des métriques NumPy, du modèle, de sa sauvegarde et de l'historique."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import entrainement
from donnees import COLONNES_CATEGORIELLES, COLONNES_NUMERIQUES, chemin_projet
from exceptions import DonneeInvalideError, FichierIntrouvableError, ModeleNonEntraineError
from historique import Historique
from modele import (DUREES, ModeleAssurance, calculer_fourchette, chronometrer,
                    coefficient_determination, erreur_absolue_moyenne, formater_montant,
                    quantile_erreur_absolue, racine_erreur_quadratique_moyenne, trier_importances)

Y_VRAI = [3000.0, 12000.5, 25000.0, 8000.0, 41000.0]
Y_PREDIT = [3500.0, 11000.0, 27000.0, 7600.0, 38000.0]


# --- métriques NumPy comparées à scikit-learn --------------------------------

def test_mae_coincide_avec_scikit_learn():
    assert erreur_absolue_moyenne(Y_VRAI, Y_PREDIT) == pytest.approx(mean_absolute_error(Y_VRAI, Y_PREDIT))


def test_rmse_coincide_avec_scikit_learn():
    attendu = np.sqrt(mean_squared_error(Y_VRAI, Y_PREDIT))
    assert racine_erreur_quadratique_moyenne(Y_VRAI, Y_PREDIT) == pytest.approx(attendu)


def test_r2_coincide_avec_scikit_learn():
    assert coefficient_determination(Y_VRAI, Y_PREDIT) == pytest.approx(r2_score(Y_VRAI, Y_PREDIT))


def test_metriques_prediction_parfaite_et_cible_constante():
    assert erreur_absolue_moyenne(Y_VRAI, Y_VRAI) == 0
    assert coefficient_determination(Y_VRAI, Y_VRAI) == pytest.approx(1.0)
    assert coefficient_determination([5, 5, 5], [5, 5, 5]) == 1.0
    assert coefficient_determination([5, 5, 5], [4, 5, 6]) == 0.0


@pytest.mark.parametrize("vrai, predit", [([], []), ([1, 2, 3], [1, 2])])
def test_metriques_entrees_invalides(vrai, predit):
    with pytest.raises(DonneeInvalideError):
        erreur_absolue_moyenne(vrai, predit)


def test_quantile_erreur():
    assert quantile_erreur_absolue([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], 0.5) == pytest.approx(3)
    with pytest.raises(DonneeInvalideError):
        quantile_erreur_absolue([1], [1], 1.5)


# --- petites fonctions utilitaires -------------------------------------------

def test_fourchette_bornee_a_zero():
    assert calculer_fourchette(1000, 2500) == (0.0, 3500)
    assert calculer_fourchette(10000, 2500) == (7500, 12500)
    with pytest.raises(DonneeInvalideError):
        calculer_fourchette(1000, -1)


def test_formater_montant():
    assert formater_montant(12345.6) == "12 346 $"


def test_trier_importances_avec_lambda():
    assert trier_importances({"a": 0.1, "b": 0.9, "c": 0.5}) == [("b", 0.9), ("c", 0.5), ("a", 0.1)]


def test_decorateur_chronometrer():
    @chronometrer
    def addition(a: int, b: int) -> int:
        return a + b
    assert addition(2, 3) == 5
    assert DUREES["test_decorateur_chronometrer.<locals>.addition"] >= 0


# --- modèle -------------------------------------------------------------------

@pytest.fixture(scope="module")
def donnees_reelles():
    return pd.read_csv(chemin_projet("donnees", "assurance.csv")).drop_duplicates()


def nouveau_modele(type_modele: str = "foret") -> ModeleAssurance:
    return ModeleAssurance(type_modele, COLONNES_NUMERIQUES, COLONNES_CATEGORIELLES, graine=42)


def test_type_de_modele_inconnu():
    with pytest.raises(DonneeInvalideError):
        nouveau_modele("reseau_de_neurones")


def test_predire_avant_entrainement_leve_une_erreur():
    with pytest.raises(ModeleNonEntraineError):
        nouveau_modele().predire({"age": 30})


def test_entrainement_sur_donnees_vides():
    with pytest.raises(DonneeInvalideError):
        nouveau_modele().entrainer(pd.DataFrame(), pd.Series(dtype=float))


def test_foret_bat_la_reference(donnees_reelles):
    """Le modèle retenu doit faire mieux que « prédire toujours la moyenne »."""
    x = donnees_reelles.drop(columns="charges")
    y = donnees_reelles["charges"]
    x_entr, x_test = x.iloc[:1000], x.iloc[1000:]
    y_entr, y_test = y.iloc[:1000], y.iloc[1000:]
    reference = nouveau_modele("reference").entrainer(x_entr, y_entr).evaluer(x_test, y_test)
    foret = nouveau_modele("foret").entrainer(x_entr, y_entr).evaluer(x_test, y_test)
    assert foret["mae"] < reference["mae"]
    assert foret["r2"] > 0.5


def test_fumeur_coute_plus_cher(donnees_reelles):
    modele = nouveau_modele().entrainer(donnees_reelles.drop(columns="charges"), donnees_reelles["charges"])
    profil = {"age": 40, "bmi": 30.0, "children": 1, "sex": "male", "region": "northeast"}
    assert modele.predire({**profil, "smoker": "yes"}) > modele.predire({**profil, "smoker": "no"})
    with pytest.raises(DonneeInvalideError, match="manquantes"):
        modele.predire({"age": 40})


def test_sauvegarde_puis_chargement_donne_la_meme_prediction(donnees_reelles, tmp_path):
    modele = nouveau_modele("lineaire").entrainer(donnees_reelles.drop(columns="charges"),
                                                  donnees_reelles["charges"])
    chemin = str(tmp_path / "modele.joblib")
    modele.sauvegarder(chemin)
    recharge = ModeleAssurance.charger(chemin)
    profil = {"age": 25, "bmi": 22.0, "children": 0, "sex": "female", "smoker": "no", "region": "southwest"}
    assert recharge.predire(profil) == pytest.approx(modele.predire(profil))
    assert recharge.est_entraine and "Régression linéaire" in str(recharge)


def test_chargement_fichier_absent_ou_corrompu(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        ModeleAssurance.charger(str(tmp_path / "absent.joblib"))
    corrompu = tmp_path / "corrompu.joblib"
    corrompu.write_bytes(b"pas un modele")
    with pytest.raises(ModeleNonEntraineError):
        ModeleAssurance.charger(str(corrompu))


def test_sauvegarder_modele_non_entraine(tmp_path):
    with pytest.raises(ModeleNonEntraineError):
        nouveau_modele().sauvegarder(str(tmp_path / "m.joblib"))


# --- démarche complète d'entraînement -----------------------------------------

def test_entrainement_complet_produit_modele_et_metadonnees(tmp_path):
    metadonnees = entrainement.executer_entrainement(
        chemin_modele=str(tmp_path / "m.joblib"),
        chemin_metadonnees=str(tmp_path / "meta.json"),
        dossier_figures=str(tmp_path / "figures"))
    relues = entrainement.charger_metadonnees(str(tmp_path / "meta.json"))
    assert relues["modele_retenu"] == metadonnees["modele_retenu"]
    assert metadonnees["metriques"][metadonnees["modele_retenu"]]["mae"] < metadonnees["metriques"]["reference"]["mae"]
    assert {v["nom"] for v in relues["variables"]} == set(COLONNES_NUMERIQUES + COLONNES_CATEGORIELLES)
    assert (tmp_path / "figures" / "exploration.png").exists()
    assert "Modèle retenu" in entrainement.resumer(metadonnees)
    json.dumps(relues)  # sérialisable


def test_metadonnees_absentes(tmp_path):
    with pytest.raises(FichierIntrouvableError):
        entrainement.charger_metadonnees(str(tmp_path / "absent.json"))


# --- historique ---------------------------------------------------------------

def test_historique_vide_puis_ajout_et_tri(tmp_path):
    historique = Historique(str(tmp_path / "sous_dossier" / "hist.csv"), ["age", "smoker"])
    assert historique.lire() == []
    historique.ajouter({"age": 30, "smoker": "no"}, 4000.123, date=datetime(2026, 1, 1, 9, 0))
    historique.ajouter({"age": 50, "smoker": "yes"}, 40000, date=datetime(2026, 1, 2, 9, 0))
    lignes = historique.lire()
    assert len(historique) == 2
    assert lignes[0]["age"] == "50"                      # la plus récente d'abord
    assert float(lignes[1]["prediction"]) == pytest.approx(4000.12)


def test_historique_ignore_une_prediction_identique_a_la_derniere(tmp_path):
    historique = Historique(str(tmp_path / "hist.csv"), ["age", "bmi", "smoker"])
    valeurs = {"age": 35, "bmi": 26.5, "smoker": "no"}
    assert historique.ajouter(valeurs, 6152.0) is not None
    assert historique.ajouter(dict(valeurs), 6152.0) is None     # mêmes données : rien n'est ajouté
    assert len(historique) == 1


def test_historique_ajoute_si_une_valeur_change(tmp_path):
    historique = Historique(str(tmp_path / "hist.csv"), ["age", "bmi", "smoker"])
    historique.ajouter({"age": 35, "bmi": 26.5, "smoker": "no"}, 6152.0)
    historique.ajouter({"age": 35, "bmi": 26.5, "smoker": "yes"}, 19509.0)   # une valeur change
    historique.ajouter({"age": 35, "bmi": 26.5, "smoker": "no"}, 6152.0)     # différente de la dernière
    assert len(historique) == 3


def test_est_identique_a_la_derniere_historique_vide(tmp_path):
    historique = Historique(str(tmp_path / "hist.csv"), ["age"])
    assert historique.derniere_entree() is None
    assert historique.est_identique_a_la_derniere({"age": 35}) is False


def test_historique_valeur_manquante(tmp_path):
    historique = Historique(str(tmp_path / "hist.csv"), ["age", "smoker"])
    with pytest.raises(DonneeInvalideError):
        historique.ajouter({"age": 30}, 1000)
