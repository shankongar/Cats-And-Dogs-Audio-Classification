### Directed Download Dataset
DownLoad URL:  https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs/data

### Bash Download Dataset
```bash
mkdir -p data && \
kaggle datasets download -d mmoreaux/audio-cats-and-dogs -p data && \
unzip -q data/audio-cats-and-dogs.zip -d data && \
rm data/audio-cats-and-dogs.zip
