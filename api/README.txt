#############################

Attention !!!!
le docker file est configurer pour utiliser le fichier test_api.py.
Toute modification de ce fichier necessite de supprimer limage existante et 
de reconstruire une nouvelle image de l'api en utilisant les commandes ci-dessous:
(Note si vous lancer l'api pour la premiere fois les commandes ci-desous sont requises)

Installer Docker Desktop

Utiliser la feature Beta "terminal Docker"

Executer les commandes suivantes:

    1. Se placer dans le repertoire contenant le fichier Dockerfile:
    cd PATH/api

    2. build image of api:
    docker build . -t myfastapi:latest

    3. Run docker:
    docker run -d -p 8080:80 myfastapi:latest

l'api est maintenant accessible a l'url: localhost:8080


Pour generer L'image a utiliser:

Generer l'image: (Necessite droits admin sur compte docker hub joan77)
1. Alle dans le repertoire du DockerFile: cd PATH
2. login au compte joan77: docker login
3. create the image: docker build -t api_test . 
4. Tag l'image: docker tag api_test joan77/api_test:latest
5. Push: docker push joan77/api_test:latest

