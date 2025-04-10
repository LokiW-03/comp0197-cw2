# README

TO run grid search

```
git checkout -b feature/ablation
mkdir -p cam/saved_models
# download classifier from: https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabc46_ucl_ac_uk/EU3B9wuc8LpBsEfLVpKVvUYBBo43vqkc528X-25_iiH09A?e=cdB93n
# save classifer to cam/saved_models

python -m ablation.grid_search ----result_path=./result.csv
```