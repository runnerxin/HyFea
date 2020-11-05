# HyFea
HyFea: Winning Solution to Social Media Popularity Prediction for Multimedia Grand Challenge 2020


The directory tree should look like this:

```bash
${ROOT}
├── data
│   ├── train_all_json
│   ├── train
│   ├── test_all_json
│   ├── test
│   ├── none_picture.jpg
│   ├── user_additional.csv
│   ├── alltags_feature.csv
│   ├── feature_data_530.csv
│   └── title_feature.csv
├── save_model
│   ├── KFold_catboost_0.pkl
│   ├── KFold_catboost_1.pkl
│   ├── KFold_catboost_2.pkl
│   ├── KFold_catboost_3.pkl
│   └── KFold_catboost_4.pkl
├── readme.txt
├── run_step2.sh
├── KFold_catboost.json
├── download_img_and_user.py
├── get_data_feature.py
├── test_k_fold_model.py
└── train_k_fold_model.py

```



#### 1 Dependencies

1. We have been implemented and tested our code on Ubuntu 16.04.5 with python == 3.6. 

2. Python packages: 

   ```python
   pip install requests beautifulsoup4 scipy Pillow gensim sklearn pandas catboost lightgbm
   ```



#### 2 Quick Start


1. You can test the model saved in folder ```./save_mode```  to get the result ```KFold_catboost.json``` base on our processed data (```feature_data_530.csv, alltags_feature.csv, title_feature.csv```) by the following command:

   ```python
   python test_k_fold_model.py
   ```

2. If you want to reproduce the process to get the processed data, you have to take the following steps: 

   2.1 Download the [GloVe word embedding](http://nlp.stanford.edu/data/glove.42B.300d.zip), then unzip ```glove.42B.300d.zip```, put the file ```glove.42B.300d.txt``` to folder ```./data```.

   2.2 Download Social Media Prediction Dataset from http://smp-challenge.com/ , unzip the ```train_all_json.zip``` to ```./data```, and put all test data files except test images to ```./data```, download ```SMP_test_images.zip``` and unzip it to folder ```./data/test```.

   2.3 Then you can run command ```python download_img_and_user.py``` to download the train images to saved in ```./data/train ``` and crawl user additional information to saved in ```./data/user_additional.csv```. 

   **Note:** You can also directly use the data ```user_additional.csv``` provided by us.

   The new added structure of the data folder should be:

   ```bash
   .
   ├── data
   │   ├── test
   │   │   ├── 1@N18
   │   │   │   ├── 56783.jpg
   │   │   │   └── ...
   │   │   └── ...
   │   ├── test_all_json
   │   │   ├── test_additional.json
   │   │   ├── test_category.json
   │   │   ├── test_imgfile.txt
   │   │   ├── test_tags.json
   │   │   ├── test_temporalspatial.json
   │   ├── train
   │   │   ├── 7107177@N05	
   │   │   │   ├── 23511051534.jpg
   │   │   │   └── ...
   │   │   └── ...
   │   ├── train_all_json
   │   │   ├── train_additional.json
   │   │   ├── train_category.json
   │   │   ├── train_img.txt
   │   │   ├── train_label.txt
   │   │   ├── train_tags.json
   │   │   ├── train_temporalspatial.json
   │   │   └── train_userdata.json
   │   ├── ...
   │   ├── user_additional.csv
   │   └── glove.42B.300d.txt
   └── ...
   ```

   2.4 Next you should run the command ```python get_data_feature.py``` to get the feature data file ```feature_data_530.csv, alltags_feature.csv, title_feature.csv```, and put all of them in folder ```./data```.

   2.5 Run command ```python train_k_fold_model.py``` to train the model, and model will be saved in the folder ```./save_model```.

   2.6 Finally you can test the model like step1. run ```python test_k_fold_model.py```  to get the submission file.

3. If the **Social Media Prediction Dataset** and **GloVe word embedding** have been downloaded, You can do all steps in 2 by running the script ```sh run_step2.sh``` .
