pip install kaggle
mkdir -p data && cd data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
unzip chest-xray-pneumonia.zip
unzip covid19-radiography-database.zip -d covid
