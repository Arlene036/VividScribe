pip3 install gdown
mkdir data/volar3k
cd data/volar3k
gdown 1vWZWWHliHuejX28K5KDRUVKoAPdPxkR5
unzip valor3k.zip

cd ..
mkdir vast3k
cd vast3k
gdown 1OVFyN8JnKqez50U30717UtNyJ7Ax3YqK
unzip vast3000.zip

cd ../..
cd VALOR
#delete apex folder first
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key...
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./