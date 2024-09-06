Generer l'image: (Necessite droits admin sur compte docker hub joan77)
1. Alle dans le repertoire du DockerFile: cd PATH
2. login au compte joan77: docker login
3. create the image: docker build -t test_streamlit . 
4. Tag l'image: docker tag test_streamlit joan77/test_streamlit:latest
5. Push: docker push joan77/test_streamlit:latest