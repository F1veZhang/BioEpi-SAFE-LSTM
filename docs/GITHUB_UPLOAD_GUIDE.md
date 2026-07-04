# GitHub upload guide

## Create the repository

Create a new GitHub repository named `BioEpi-SAFE-LSTM`.

## Upload from the command line

```bash
cd BioEpi-SAFE-LSTM
git init
git add .
git commit -m "Initial BioEpi-SAFE-LSTM reproducibility release"
git branch -M main
git remote add origin https://github.com/<USERNAME>/BioEpi-SAFE-LSTM.git
git push -u origin main
```

## Use Git LFS only if needed

The largest included file is below ordinary GitHub single-file limits. If larger files are added later, track them with Git LFS:

```bash
git lfs install
git lfs track "*.gz"
git lfs track "*.zip"
git add .gitattributes
git commit -m "Track large reproducibility files with Git LFS"
git push
```

## Create a release

```bash
git tag -a v0.1.0 -m "Peer-review reproducibility release"
git push origin v0.1.0
```

Then create a GitHub Release from tag `v0.1.0`. Link the release to Zenodo if a citable DOI is required.
