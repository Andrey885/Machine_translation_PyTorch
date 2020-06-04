wget https://drive.google.com/uc?id=1d-Q-tzGo1RN_YbmvMP5gArQZZdSnclJ0 -O checkpoints/en_de_final.pt
python3 -m pip install -r requirements.txt
sudo python3 -m spacy download en
sudo python3 -m spacy download de_core_news_sm
git submodule init
git submodule update

