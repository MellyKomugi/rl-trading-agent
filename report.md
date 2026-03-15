Une chose à noter : le train commence en 2019-01-03 au lieu de 2018-01-01, c'est normal et attendu. Les 252 premiers jours (1 an) sont consommés par le calcul de la covariance rolling. À mentionner dans le rapport












CHOIX DES INDICATEURS 




Justification 1 (peu fiable) — Reproductibilité? 

Ce que le papier dit exactement pour portfolio allocation
Pour leur expérience portfolio allocation sur le DJIA 30 stocks, ils utilisent :

Features de prix/volume :

Balance (cash disponible)
Shares owned (positions)
Closing price
OHLC prices (Open, High, Low, Close)
Trading volume
Indicateurs techniques :

MACD
RSI
"etc." ← ils ne précisent pas les autres
Conclusion
Le papier ne liste pas explicitement CCI et DX pour portfolio allocation. Ils utilisent MACD + RSI et laissent le reste vague avec "etc."

D'où viennent CCI et DX alors ? De la documentation et des exemples de code FinRL — c'est ce que leur implémentation utilise par défaut dans les notebooks d'exemple, mais pas formalisé dans le papier scientifique.


C'est honnête et scientifiquement solide.





Justification 2 — Couverture des dimensions

"Nous choisissons un indicateur par famille : momentum (MACD), force relative (RSI), déviation prix (CCI), force de tendance (DX) — pour couvrir les aspects principaux du marché sans redondance."

Mieux encore — en faire une contribution
Au lieu de subir ce choix, transformez-le en expérience :

Expérience 3 (optionnelle) : Ablation sur le state space

Configuration A : 4 indicateurs (paper FinRL)
Configuration B : 2 indicateurs (MACD + RSI seulement)
Configuration C : 6 indicateurs (+ close_30_sma + boll)

→ Question : est-ce que plus d'indicateurs = meilleur agent ?








Les indicateurs choisis — explication 
Ce que FinRL propose
FinRL utilise la librairie stockstats qui fournit des dizaines d'indicateurs. Voici les principaux utilisables :

Indicateur	Famille
macd	Momentum
rsi_30	Force relative
cci_30	Déviation prix
dx_30	Force de tendance
close_30_sma	Moyenne mobile
close_60_sma	Moyenne mobile longue
boll_ub, boll_lb	Bollinger Bands
atr	Volatilité
Pourquoi ces 4 en particulier
Ce sont exactement les indicateurs utilisés dans le papier FinRL original — ce qui vous donne une référence directe pour comparer vos résultats aux leurs. C'est un argument solide pour le jury.

Ils couvrent aussi 4 dimensions différentes du marché :

Ce que signifie chaque indicateur
MACD (Moving Average Convergence Divergence)

Compare deux moyennes mobiles exponentielles (12j vs 26j)
MACD > 0 → tendance haussière, MACD < 0 → tendance baissière
Signal de momentum : est-ce que le prix accélère ou ralentit ?
Prix monte vite  →  MACD positif et croissant  →  signal d'achat
Prix ralentit    →  MACD positif mais décroissant  →  signal de prudence

RSI_30 (Relative Strength Index sur 30 jours)

Mesure la vitesse des variations de prix, entre 0 et 100
RSI > 70 → surachat (le prix a trop monté, potentiel retournement)
RSI < 30 → survente (le prix a trop baissé, potentiel rebond)
Signal de retour à la moyenne
CCI_30 (Commodity Channel Index sur 30 jours)

Mesure l'écart entre le prix actuel et sa moyenne sur 30 jours
CCI > +100 → prix anormalement haut
CCI < -100 → prix anormalement bas
Complémentaire au RSI — même idée mais calcul différent
DX_30 (Directional Movement Index sur 30 jours)

Mesure la force d'une tendance (pas sa direction)
DX > 25 → tendance forte (haussière ou baissière)
DX < 20 → marché sans tendance claire (range)
Utile pour savoir si MACD et RSI sont fiables dans le contexte actuel
Résumé visuel
MACD    →  "Dans quelle direction va le prix ?"
RSI_30  →  "Le prix est-il trop haut ou trop bas ?"
CCI_30  →  "Le prix s'écarte-t-il de sa moyenne ?"
DX_30   →  "Y a-t-il une vraie tendance en ce moment ?"

Ces 4 indicateurs ensemble donnent à l'agent une vue complète du marché sans le surcharger d'information — c'est le bon compromis pour un projet de cours.








Différence fondamentale avec approche RL
Optimisation Mathématique approche RL
Méthode	Formule analytique	Apprentissage par essai-erreur
Adaptatif ?	Non — poids figés	Oui — l'agent s'adapte
Besoin de données futures ?	Suppose que le passé = futur	Apprend des patterns
Exemple	Markowitz, Min-Variance	DQN, REINFORCE
Pourquoi c'est utile pour vous
PortfolioOptimizationEnv implémente Min-Variance (minimiser le risque) — c'est exactement la baseline que le papier FinRL utilise dans leurs comparaisons.

Vous pouvez l'utiliser comme troisième baseline aux côtés de Buy & Hold et PPO :

Vos agents    : DQN, REINFORCE, Exp3
Baselines     : Buy & Hold, PPO (SB3), Min-Variance (Markowitz)
                                        ↑
                              ajoute une baseline "intelligente"
                              mais sans RL — renforce analyse






def add_features(df):

Prend le DataFrame brut (prix OHLCV) et ajoute des colonnes d'indicateurs techniques. Retourne un DataFrame enrichi.

FeatureEngineer(...) — les paramètres
use_technical_indicator=True

Active le calcul des indicateurs techniques. Si False, FinRL ne calcule rien — vous auriez juste les prix bruts dans l'état, ce qui est insuffisant pour un agent RL.

tech_indicator_list=INDICATORS
# INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]

Spécifie exactement quels indicateurs calculer. FinRL utilise la librairie stockstats en arrière-plan pour les calculer sur chaque stock.

use_turbulence=False

La turbulence est un indicateur de risque de marché développé par Kritzman (2010) — il mesure si le marché se comporte de façon anormale par rapport à son historique. Utile pour éviter les crises.

Vous le désactivez pour simplifier — sinon FinRL ajouterait une colonne turbulence à l'état et pourrait bloquer l'agent quand le marché est "trop risqué".

user_defined_feature=False

Désactive la possibilité d'ajouter vos propres features custom. Vous n'en avez pas besoin ici — vos indicateurs sont déjà dans INDICATORS.

## Turbulence — faut-il l'activer ?

### Votre période de test (2022-2023) est effectivement risquée

```
2022 :
- S&P500 : -20%
- AAPL   : -27%
- JPM    : -15%
- XOM    : +58% (exception — choc pétrolier Ukraine)
- Cause  : hausse des taux Fed, inflation, guerre Ukraine
```

Donc oui — votre période de test contient exactement le type de crise que la turbulence détecte.

### Mais voici le problème clé — comment FinRL utilise la turbulence

Ce n'est **pas** juste un indicateur ajouté à l'état. C'est une **contrainte forcée** :

```
Si turbulence > seuil → l'agent est FORCÉ de tout vendre
                         peu importe ce que son réseau veut faire
```

C'est un mécanisme de stop-loss automatique qui **court-circuite la décision de l'agent**.

### Pourquoi ça pose problème pour votre projet

```
Avec turbulence ON :
DQN    → forcé de vendre en 2022  ┐
REINFORCE → forcé de vendre        ├── tous identiques en crise
Exp3   → forcé de vendre          ┘

→ Vous ne comparez plus les agents, vous comparez
  le même mécanisme de stop-loss appliqué à tous
```

La différence entre vos agents disparaît exactement pendant la période la plus intéressante — la crise 2022.

### Recommandation

```python
# Option 1 — OFF (votre setup actuel)
use_turbulence=False
# → Les agents gèrent le risque eux-mêmes via le reward
# → Différences entre agents visibles même en crise 

# Option 2 — Contribution originale
# Ajouter turbulence comme feature dans l'état SANS la contrainte forcée
INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30", "turbulence"]
# → L'agent VOIT le risque mais décide lui-même quoi faire 
```

L'option 2 est plus intéressante scientifiquement — et c'est une ligne à changer dans `config.py`.

---

## `fillna(0)` ?

Deux sources de NaN :

**Source 1 — Début de série (garanti)**
```
Jour 1  à 29 → RSI_30  = NaN  (pas encore 30 jours d'historique)
Jour 1  à 25 → MACD    = NaN  (besoin de 26 jours)
Jour 1  à 29 → CCI_30  = NaN
Jour 1  à 29 → DX_30   = NaN
```

**Source 2 — Données manquantes Yahoo Finance (possible)**
```
Jours fériés, suspensions de cotation, données corrompues → NaN
```

### Est-ce que `fillna(0)` est la meilleure approche ?

Honnêtement — c'est discutable :

| Méthode | Problème |
|---|---|
| `fillna(0)` | RSI=0 signifie "extrêmement survendu" — c'est un faux signal |
| `fillna(method='ffill')` | Propage la dernière valeur connue — plus réaliste |
| Supprimer les 30 premières lignes | Propre mais perd des données |

```python
# Plus propre que fillna(0)
df = df.fillna(method='ffill').fillna(0)
# ffill d'abord → propage valeurs précédentes
# fillna(0) ensuite → gère les NaN restants au tout début
```






L'état réel est : matrice de covariance + indicateurs. Pas de prix, pas de shares, pas de balance.

state_space = stock_dim  # = 3
# L'observation sera de shape (stock_dim + len(INDICATORS), stock_dim)
# soit (3 + 4, 3) = (7, 3)

La covariance mesure comment les stocks évoluent ensemble. Pour 3 stocks :

          AAPL    JPM    XOM
AAPL  [  0.02   0.01  -0.003 ]   → ligne 1
JPM   [  0.01   0.015  0.002 ]   → ligne 2
XOM   [ -0.003  0.002  0.018 ]   → ligne 3

L'état complet est donc une matrice 2D :

              AAPL    JPM    XOM
cov_AAPL  [  0.02   0.01  -0.003 ]   ← corrélations
cov_JPM   [  0.01   0.015  0.002 ]   ← corrélations
cov_XOM   [ -0.003  0.002  0.018 ]   ← corrélations
macd      [  0.23  -0.11   0.05  ]   ← indicateur par stock
rsi_30    [ 45.2   52.1   38.4  ]   ← indicateur par stock
cci_30    [ 82.1  -12.3   45.6  ]   ← indicateur par stock
dx_30     [ 31.4   28.7   19.2  ]   ← indicateur par stock

shape = (3 + 4, 3) = (7, 3)
              ↑       ↑
         nb_lignes   nb_colonnes = stock_dim

state_space = stock_dim = 3 définit le nombre de colonnes de cette matrice.






add_covariance — d'où vient-elle ?
FinRL ne fournit aucune fonction pour calculer cov_list. Seul env_portfolio.py l'attend sans jamais la calculer.

# Preuve — cov_list n'existe que dans l'env, jamais dans les utilitaires
# → /finrl/meta/env_portfolio_allocation/env_portfolio.py  
# → Aucun autre fichier FinRL ✗

Notre implémentation vient des notebooks tutoriels officiels FinRL sur GitHub, c'est le code standard qu'ils utilisent dans leurs démos de portfolio allocation mais qui n'est pas packagé dans la librairie.

La logique est simple :

Pour chaque date t :
  → prendre les 252 jours précédents (= 1 an de trading)
  → calculer les rendements journaliers de chaque stock
  → calculer la matrice de covariance 3×3
  → stocker dans cov_list[t]

Pourquoi 252 jours ? C'est le nombre standard de jours de trading par an, utilisé partout en finance pour les calculs de covariance rolling.

Ce que ça implique : les 252 premiers jours de vos données (début 2018) seront perdus, l'env ne commencera vraiment qu'à partir de début 2019. C'est normal et attendu



dans config
REWARD_SCALING = 1e-4 #Censé diviser le reward par 10 000 pour stabiliser l'apprentissage. Mais dans le code source :
# Ligne 197-199 dans StockPortfolioEnv.step() :
# self.reward = new_portfolio_value
# self.reward = self.reward * self.reward_scaling  ← commenté = désactivé dans finrl
#Reward brut FinRL = valeur portfolio = 100 000, 101 543, 99 821...
#Réseau de neurones voit :
#  Q(s,a) ≈ 100 000  → gradients énormes → apprentissage instable
# Du coup
# Dans chaque agent — une seule ligne à ajouter :
# obs, reward, done, _, _ = env.step(action)
# reward = reward * REWARD_SCALING  # ← chacun applique ça dans sa boucle