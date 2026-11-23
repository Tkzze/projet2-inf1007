"""Interface graphique Tkinter.

L'interface ne contient aucune logique de calcul : elle lit les saisies, appelle
``variables.valider_saisies`` puis ``ModeleAssurance.predire``, et affiche les
résultats (séparation du calcul et des entrées-sorties).
"""

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import graphiques
from donnees import LIBELLES
from exceptions import DonneeInvalideError
from historique import Historique
from modele import ModeleAssurance, calculer_fourchette, formater_montant, trier_importances
from variables import (Variable, VariableCategorielle, VariableNumerique, creer_variables_depuis,
                       valider_saisies)

AVERTISSEMENT = (
    "⚠  Exercice pédagogique (INF1007) — cet outil n'est PAS une aide à la décision médicale, "
    "financière ou d'assurance. Les données sont américaines et ne s'appliquent pas au Québec."
)
COULEUR_ERREUR = "#b00020"
COULEUR_RESULTAT = "#1b5e20"
TEXTE_ACCUEIL = "Remplissez le profil puis cliquez sur « Prédire »."
# En-têtes courts pour le tableau de l'historique.
ENTETES_COURTS = {"date": "Date", "age": "Âge", "bmi": "IMC", "children": "Enfants", "sex": "Sexe",
                  "smoker": "Fumeur", "region": "Région", "prediction": "Coût estimé"}


class Application(tk.Tk):
    """Fenêtre principale.

    Composition : l'application contient le modèle, la liste des variables et
    l'historique des prédictions.
    """

    def __init__(self, modele: ModeleAssurance, metadonnees: dict, historique: Historique) -> None:
        """Construit la fenêtre à partir du modèle chargé et de ses métadonnées.

        Args:
            modele: modèle déjà entraîné.
            metadonnees: dictionnaire lu dans ``metadonnees.json``.
            historique: historique des prédictions.
        """
        super().__init__()
        self._modele = modele
        self._metadonnees = metadonnees
        self._historique = historique
        self._variables: list[Variable] = creer_variables_depuis(metadonnees)
        self._champs: dict[str, tk.Widget] = {}

        self.title("Estimation du coût médical annuel — Projet 2 INF1007")
        self.geometry("1200x720")
        self.minsize(950, 620)

        self._construire_bandeau()
        onglets = ttk.Notebook(self)
        onglets.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._construire_onglet_prediction(onglets)
        self._construire_onglet_historique(onglets)
        self._construire_onglet_modele(onglets)
        self._rafraichir_historique()

    # ------------------------------------------------------------------ construction

    def _construire_bandeau(self) -> None:
        """Bandeau d'avertissement, toujours visible en haut de la fenêtre."""
        bandeau = tk.Label(self, text=AVERTISSEMENT, bg="#fff3cd", fg="#664d03", wraplength=1150,
                           justify="left", padx=10, pady=8, font=("Segoe UI", 10, "bold"))
        bandeau.pack(fill="x", padx=10, pady=10)

    def _construire_onglet_prediction(self, onglets: ttk.Notebook) -> None:
        """Onglet principal : formulaire à gauche, résultat et graphique à droite."""
        onglet = ttk.Frame(onglets, padding=10)
        onglets.add(onglet, text="Prédiction")

        formulaire = ttk.LabelFrame(onglet, text="Profil de la personne", padding=10)
        formulaire.pack(side="left", fill="y")

        for rang, variable in enumerate(self._variables):
            ttk.Label(formulaire, text=variable.libelle, font=("Segoe UI", 10, "bold")).grid(
                row=2 * rang, column=0, sticky="w", pady=(6, 0))
            champ = self._creer_champ(formulaire, variable)
            champ.grid(row=2 * rang, column=1, sticky="ew", padx=(10, 0), pady=(6, 0))
            ttk.Label(formulaire, text=variable.decrire_contrainte(), foreground="grey", wraplength=200).grid(
                row=2 * rang + 1, column=1, sticky="w", padx=(10, 0))
            self._champs[variable.nom] = champ

        boutons = ttk.Frame(formulaire)
        boutons.grid(row=2 * len(self._variables), column=0, columnspan=2, pady=(15, 5), sticky="ew")
        ttk.Button(boutons, text="Prédire", command=self._predire).pack(side="left", expand=True, fill="x")
        ttk.Button(boutons, text="Effacer", command=self._effacer).pack(side="left", expand=True, fill="x",
                                                                        padx=(8, 0))

        self._message = tk.Label(formulaire, text="", fg=COULEUR_ERREUR, justify="left", wraplength=330,
                                 anchor="w")
        self._message.grid(row=2 * len(self._variables) + 1, column=0, columnspan=2, sticky="ew")

        droite = ttk.Frame(onglet)
        droite.pack(side="left", fill="both", expand=True, padx=(15, 0))
        self._resultat = tk.Label(droite, text=TEXTE_ACCUEIL,
                                  font=("Segoe UI", 13), justify="left", anchor="w", wraplength=650)
        self._resultat.pack(fill="x")
        self._detail = tk.Label(droite, text="", justify="left", anchor="w", wraplength=650, fg="grey25")
        self._detail.pack(fill="x", pady=(4, 8))
        self._canevas_prediction = self._integrer_figure(droite, self._figure_distribution())

        self.bind("<Return>", lambda _evenement: self._predire())

    def _creer_champ(self, parent: ttk.Frame, variable: Variable) -> tk.Widget:
        """Crée le widget adapté au type de variable (polymorphisme côté affichage)."""
        if isinstance(variable, VariableCategorielle):
            champ = ttk.Combobox(parent, values=variable.libelles, state="readonly", width=16)
            champ.current(0)
            return champ
        champ = ttk.Entry(parent, width=18)
        if isinstance(variable, VariableNumerique):
            exemple = round((variable.minimum + variable.maximum) / 2)
            champ.insert(0, str(exemple))
        return champ

    def _construire_onglet_historique(self, onglets: ttk.Notebook) -> None:
        """Onglet listant les prédictions enregistrées dans le CSV."""
        onglet = ttk.Frame(onglets, padding=10)
        onglets.add(onglet, text="Historique")
        colonnes = self._historique.entetes
        self._tableau = ttk.Treeview(onglet, columns=colonnes, show="headings", height=18)
        for colonne in colonnes:
            self._tableau.heading(colonne, text=ENTETES_COURTS.get(colonne, LIBELLES.get(colonne, colonne)))
            self._tableau.column(colonne, width=150 if colonne == "date" else 115, anchor="center")
        defilement = ttk.Scrollbar(onglet, orient="vertical", command=self._tableau.yview)
        self._tableau.configure(yscrollcommand=defilement.set)
        self._tableau.pack(side="left", fill="both", expand=True)
        defilement.pack(side="left", fill="y")
        ttk.Label(onglet, text=f"Fichier :\n{self._historique.chemin}", wraplength=180,
                  foreground="grey").pack(side="top", padx=10, anchor="n")
        ttk.Button(onglet, text="Actualiser", command=self._rafraichir_historique).pack(
            side="top", padx=10, pady=10, anchor="n")

    def _construire_onglet_modele(self, onglets: ttk.Notebook) -> None:
        """Onglet expliquant la qualité du modèle et l'importance des variables."""
        onglet = ttk.Frame(onglets, padding=10)
        onglets.add(onglet, text="À propos du modèle")
        metriques = self._metadonnees["metriques"]
        retenu = self._metadonnees["modele_retenu"]
        texte = (
            f"Modèle utilisé : {self._metadonnees['nom_modele_retenu']}, entraîné sur "
            f"{self._metadonnees['lignes_entrainement']} personnes et testé sur "
            f"{self._metadonnees['lignes_test']} autres personnes jamais vues.\n\n"
            f"Erreur moyenne sur le test : {formater_montant(metriques[retenu]['mae'])} "
            f"(R² = {metriques[retenu]['r2']:.2f}).\n"
            f"À titre de comparaison, prédire toujours la moyenne donnerait une erreur moyenne de "
            f"{formater_montant(metriques['reference']['mae'])}, et une régression linéaire "
            f"{formater_montant(metriques['lineaire']['mae'])}.\n\n"
            "Limites : 1 337 personnes seulement, données américaines simulées à partir de statistiques "
            "de recensement, aucune information sur l'état de santé réel. Le modèle ne fait que "
            "reproduire des tendances moyennes."
        )
        tk.Label(onglet, text=texte, justify="left", anchor="w", wraplength=1000).pack(fill="x")
        importances = trier_importances(self._metadonnees["importances"])
        self._integrer_figure(onglet, graphiques.figure_importance(importances, LIBELLES))

    def _integrer_figure(self, parent: tk.Widget, figure: Figure) -> FigureCanvasTkAgg:
        """Intègre une figure Matplotlib dans un conteneur Tkinter."""
        canevas = FigureCanvasTkAgg(figure, master=parent)
        canevas.draw()
        canevas.get_tk_widget().pack(fill="both", expand=True)
        return canevas

    def _figure_distribution(self) -> Figure:
        """Construit le graphique initial (distribution des coûts, sans prédiction)."""
        distribution = self._metadonnees["distribution_cible"]
        return graphiques.figure_position_prediction(distribution["bords"], distribution["comptes"])

    def _redessiner_distribution(self, prediction: float | None = None,
                                 fourchette: tuple[float, float] | None = None) -> None:
        """Redessine le graphique intégré, avec ou sans prédiction."""
        distribution = self._metadonnees["distribution_cible"]
        graphiques.dessiner_position_prediction(self._canevas_prediction.figure, distribution["bords"],
                                                distribution["comptes"], prediction, fourchette)
        self._canevas_prediction.draw()

    # ------------------------------------------------------------------ actions

    def _lire_saisies(self) -> dict[str, str]:
        """Récupère le texte brut de chaque champ."""
        return {nom: champ.get() for nom, champ in self._champs.items()}

    def _predire(self) -> None:
        """Valide les saisies, prédit, affiche le résultat et l'ajoute à l'historique."""
        try:
            valeurs = valider_saisies(self._variables, self._lire_saisies())
            prediction = self._modele.predire(valeurs)
        except DonneeInvalideError as erreur:
            self._message.config(text=f"Veuillez corriger :\n{erreur}", fg=COULEUR_ERREUR)
            self._reinitialiser_resultat()
            return

        fourchette = calculer_fourchette(prediction, self._metadonnees["erreur_absolue_80"])
        self._resultat.config(text=f"Coût médical annuel estimé : {formater_montant(prediction)}",
                              fg=COULEUR_RESULTAT, font=("Segoe UI", 15, "bold"))
        self._detail.config(text=(
            f"Fourchette probable : de {formater_montant(fourchette[0])} à {formater_montant(fourchette[1])}. "
            f"Lors des tests, l'erreur du modèle était inférieure à "
            f"{formater_montant(self._metadonnees['erreur_absolue_80'])} dans 8 cas sur 10."))
        self._redessiner_distribution(prediction, fourchette)

        try:
            self._historique.ajouter(valeurs, prediction)
            self._message.config(text="Prédiction ajoutée à l'historique.", fg="grey25")
        except OSError as erreur:
            self._message.config(text=f"Prédiction affichée, mais l'historique n'a pas pu être "
                                      f"enregistré : {erreur.strerror}", fg=COULEUR_ERREUR)
        self._rafraichir_historique()

    def _reinitialiser_resultat(self) -> None:
        """Retire l'ancien résultat pour ne jamais laisser croire qu'il correspond aux saisies actuelles."""
        self._resultat.config(text=TEXTE_ACCUEIL, fg="black", font=("Segoe UI", 13))
        self._detail.config(text="")
        self._redessiner_distribution()

    def _effacer(self) -> None:
        """Vide les champs numériques et remet les listes à leur premier choix."""
        for champ in self._champs.values():
            if isinstance(champ, ttk.Combobox):
                champ.current(0)
            else:
                champ.delete(0, tk.END)
        self._message.config(text="")
        self._reinitialiser_resultat()

    def _rafraichir_historique(self) -> None:
        """Recharge le tableau de l'onglet Historique depuis le fichier CSV."""
        self._tableau.delete(*self._tableau.get_children())
        libelles_categories = {v.nom: v for v in self._variables if isinstance(v, VariableCategorielle)}
        for ligne in self._historique.lire():
            affichage = []
            for colonne in self._historique.entetes:
                valeur = ligne.get(colonne, "")
                if colonne in libelles_categories:
                    valeur = libelles_categories[colonne].libelle_de(valeur)
                elif colonne == "prediction":
                    try:
                        valeur = formater_montant(float(valeur))
                    except ValueError:
                        pass  # valeur illisible : on l'affiche telle quelle
                affichage.append(valeur)
            self._tableau.insert("", tk.END, values=affichage)

