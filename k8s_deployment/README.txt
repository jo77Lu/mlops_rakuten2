# Pour lancer l'API:

1. CMD:  cd ./api/
2. CMD: kubectl create -f api-service.yml
3. CMD: kubectl create -f api-deployment.yml
4. Attendre le demarrage de l'api peut prendre un certain temps (jusqu'a 5-10 min)

# Pour lancer Streamlit (UI):

1. CMD:  cd ./streamlit/
2. CMD: kubectl create -f streamlit-service.yml
3. CMD: kubectl create -f streamlit-deployment.yml
4. Attendre le demarrage de l'api peut prendre un certain temps (jusqu'a 5-10 min)