# Notes techniques : évaluation de réponses en français et en arabe

## Périmètre

Ce dépôt contient un prototype d'inférence ASAG (Automated Student Answer Grading), une interface Streamlit et des jeux de données préparés. Il ne contient ni les poids entraînés, ni les scripts de préparation, d'entraînement et d'évaluation permettant de reproduire les performances annoncées auparavant.

## Implémentation

Le routeur recherche des caractères de la plage Unicode `U+0600–U+06FF` dans la question et la réponse de l'étudiant. Il sélectionne l'expert arabe si cette plage est présente et l'expert français dans les autres cas. Cette règle traite aussi les textes anglais comme du français et privilégie l'arabe pour les textes mixtes.

Le nom « Mixture of Experts » désigne ici une sélection déterministe entre deux classifieurs. Il ne s'agit pas d'un réseau de routage appris. Les deux modèles sont chargés en mémoire lorsqu'ils sont disponibles.

L'entrée neuronale associe la question et la réponse de référence, puis la réponse de l'étudiant, avec une limite de 256 tokens. Le code suppose trois classes, dans l'ordre `correct`, `partiel`, `incorrect`. Cette convention doit être vérifiée avec la configuration d'entraînement de chaque modèle.

Un classifieur TF-IDF/SVM peut servir de secours. Il utilise la question et la réponse de l'étudiant sans exploiter la référence. Le score softmax des experts n'est pas calibré ; aucun score de confiance n'est fourni pour le SVM.

## Chiffres historiques

Les notebooks reprennent des valeurs saisies manuellement : 72 % pour TF-IDF/SVM, 77 % pour SBERT/SVM, 82,93 % pour un SVM optimisé, 85,8 % pour un Transformer ajusté, 93,2 % pour l'expert français et environ 90 % pour l'expert arabe. Les spécifications MiniLM qui y figurent sont également déclaratives, faute de checkpoints vérifiables.

Ces chiffres ne constituent pas une évaluation reproductible. Sans journaux, prédictions, protocole de séparation et configurations, ils ne démontrent ni la supériorité du routage, ni une fiabilité en production, ni une comparaison avec des correcteurs humains. Le passage de 72 à 93,2 représente une différence de 21,2 points de pourcentage, sous réserve de comparabilité des expériences.

## Audit des données présentes

Le jeu bilingue comprend 5 510 questions d'entraînement, 1 181 de validation et 1 181 de test. Les nombres de réponses associées sont respectivement 13 284, 2 802 et 2 815.

Des clés identiques `(langue, question, référence)` apparaissent dans plusieurs partitions : 3 entre entraînement et validation, 4 entre entraînement et test, 2 entre validation et test. Des passages de contexte sont également partagés. Dans le test, 992 des 1 181 réponses étiquetées `correct` sont identiques à la référence. Cette construction peut faciliter la tâche par rapport à des réponses naturelles reformulées ; son impact doit être mesuré.

Les scripts de génération, la provenance des annotations et les conditions de redistribution des données restent à documenter. Il n'est pas possible de conclure que les réponses ont toutes été corrigées par des humains.

## Validation à compléter

Pour rendre les résultats vérifiables, il faut fournir les checkpoints et leur configuration, le code de préparation et d'entraînement, les versions des dépendances et les graines aléatoires. L'évaluation devrait isoler les questions et contextes selon l'objectif de généralisation, publier les métriques par langue et par classe, et tester des réponses naturelles annotées indépendamment.

Les tests du dépôt vérifient le routage et le comportement lorsque les modèles manquent. Ils ne mesurent pas la qualité des notes. Aucun déploiement public ni benchmark de latence n'est établi par les fichiers présents.
