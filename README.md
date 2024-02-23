create env

```bash
conda create -n mlops-main python=3.8 -y
```

activate env

```bash
conda activate mlops-main 
```
created a req file

install the req

```bash
pip install -r requirements.txt
```

download the data from

https://drive.google.com/drive/u/0/folders/1VzWT7ndsvNTVVKdLjo1ccAvwvKK8SUeV

```bash
git init
```
```bash
dvc init 
```
```bash
dvc add data_given/**.csv
```
```bash
git add .
```
```bash
git commit -m "first commit"
```


oneliner updates for readme
```bash
git add . && git commit -m "update Readme.md"
```
```bash
git remote add origin https://github.com/*
```
```bash
git branch -M main
```
```bash
git push origin main
```