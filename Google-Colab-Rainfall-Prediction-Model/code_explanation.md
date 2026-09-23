# 🌧️ Rainfall Prediction ML Pipeline – Code Explanation

## 1. Project Overview

* এই project-এর উদ্দেশ্য হলো historical weather data ব্যবহার করে **daily rainfall prediction** করা।
* Dataset-এ বিভিন্ন weather এবং station-related features ব্যবহার করা হয়েছে।
* একাধিক Machine Learning model এবং LSTM model train করে performance compare করা হয়েছে।
* Final model হিসেবে validation performance অনুযায়ী best model নির্বাচন করা হয়েছে।

---

## 2. Required Libraries

* `numpy` → numerical calculation-এর জন্য।
* `pandas` → dataset read, clean এবং manipulate করার জন্য।
* `matplotlib` → graph এবং visualization-এর জন্য।
* `scikit-learn` → preprocessing এবং evaluation metrics-এর জন্য।
* `CatBoost` → gradient boosting regression model।
* `LightGBM` → fast gradient boosting regression model।
* `XGBoost` → powerful boosting-based regression model।
* `TensorFlow/Keras` → LSTM deep learning model তৈরির জন্য।

---

## 3. Random Seed

* `numpy` এবং `tensorflow`-এর random seed `42` সেট করা হয়েছে।
* এর ফলে model training বারবার চালালেও result তুলনামূলকভাবে reproducible থাকে।

---

# 4. Dataset Upload

* Google Colab-এর `files.upload()` ব্যবহার করে CSV dataset upload করা হয়েছে।
* Code নিশ্চিত করে যে একটি CSV file পাওয়া গেছে।
* Dataset `pandas DataFrame` হিসেবে load করা হয়েছে।

---

# 5. Required Column Validation

* Model চালানোর আগে প্রয়োজনীয় columns dataset-এ আছে কিনা check করা হয়েছে।
* কোনো required column missing থাকলে error দেখানো হয়।
* এতে পরবর্তী processing-এর সময় unexpected error কমে।

---

# 6. Date Processing

* `Date` column-কে pandas datetime format-এ convert করা হয়েছে।
* Invalid date থাকলে সেগুলো `NaT` হিসেবে পাওয়া যায়।
* Date অনুযায়ী dataset sort করা হয়েছে।
* প্রতিটি station-এর date sequence ঠিক রাখা হয়েছে।

---

# 7. Duplicate Check

* `Station_ID + Date` combination ব্যবহার করে duplicate data check করা হয়েছে।
* একই station-এর একই দিনের duplicate record থাকলে তা identify করা হয়।
* এতে একই day's data একাধিকবার model training-এ যাওয়ার সমস্যা কমে।

---

# 8. Target Variable

* Main target variable:

  * `rain_sum`
* Dataset-এ `precipitation_sum`-ও ছিল।
* দুইটি rainfall column-এর মধ্যে correlation এবং difference check করা হয়েছে।
* পাওয়া গেছে:

  * Correlation = `1.0`
  * Maximum absolute difference = `0.0`
* অর্থাৎ দুইটি column একই rainfall information represent করছে।
* Model-এর target হিসেবে `rain_sum` ব্যবহার করা হয়েছে।

---

# 9. Negative Rainfall Handling

* Rainfall কখনো negative হওয়া উচিত নয়।
* তাই `rain_sum < 0` হলে সেই value invalid হিসেবে ধরা হয়েছে।
* Invalid target values `NaN` করা হয়েছে।
* এরপর missing target rows বাদ দেওয়া হয়েছে।

---

# 10. Calendar Feature Engineering

Date থেকে নতুন features তৈরি করা হয়েছে:

* `year`
* `month`
* `day`
* `dayofyear`
* `dayofweek`
* `weekofyear`

এগুলো model-কে seasonal এবং yearly rainfall pattern বুঝতে সাহায্য করে।

---

# 11. Cyclic Feature Encoding

Calendar features-এর জন্য `sin()` এবং `cos()` encoding ব্যবহার করা হয়েছে।

তৈরি করা হয়েছে:

* `month_sin`
* `month_cos`
* `dayofyear_sin`
* `dayofyear_cos`
* `dayofweek_sin`
* `dayofweek_cos`

### কেন?

* December এবং January calendar-wise কাছাকাছি হলেও numerical value হিসেবে 12 এবং 1 অনেক দূরে।
* Sin/Cos encoding এই cyclic relationship model-কে বুঝতে সাহায্য করে।

---

# 12. Rainfall Lag Features

Previous days-এর rainfall থেকে lag features তৈরি করা হয়েছে।

Used lags:

* 1 day
* 2 days
* 3 days
* 5 days
* 7 days
* 14 days
* 21 days
* 30 days

উদাহরণ:

* `rain_lag_1` → previous day's rainfall
* `rain_lag_7` → 7 days আগের rainfall
* `rain_lag_30` → 30 days আগের rainfall

### উদ্দেশ্য

Previous rainfall থেকে current rainfall-এর temporal pattern বোঝানো।

---

# 13. Rain Occurrence Features

Rainfall আছে কিনা সেটা binary feature হিসেবে তৈরি করা হয়েছে।

* Rainfall > 0 → `1`
* Rainfall = 0 → `0`

এরপর lag তৈরি করা হয়েছে:

* `rain_occurrence_lag_1`
* `rain_occurrence_lag_2`
* `rain_occurrence_lag_3`
* `rain_occurrence_lag_7`

এগুলো model-কে previous days-এ বৃষ্টি হয়েছিল কিনা বুঝতে সাহায্য করে।

---

# 14. Rolling Rainfall Features

Previous rainfall ব্যবহার করে rolling statistics তৈরি করা হয়েছে।

Windows:

* 3 days
* 7 days
* 14 days
* 30 days

প্রতিটি window-এর জন্য তৈরি করা হয়েছে:

* Rolling Mean
* Rolling Sum
* Rolling Standard Deviation

### Important

Rolling calculation-এর আগে `shift(1)` ব্যবহার করা হয়েছে।

এর ফলে current day's rainfall accidentally feature-এর মধ্যে চলে আসে না।

---

# 15. Previous Weather Features

Weather variables-এর historical values ব্যবহার করার জন্য lag features তৈরি করা হয়েছে।

Lag:

* 1 day
* 2 days
* 3 days
* 7 days

এর মাধ্যমে model previous weather conditions থেকে rainfall pattern শেখে।

---

# 16. Station Encoding

Dataset-এ একাধিক weather station ছিল।

* `Station_ID` categorical variable হিসেবে handle করা হয়েছে।
* `pd.get_dummies()` ব্যবহার করে One-Hot Encoding করা হয়েছে।
* প্রতিটি station-এর জন্য আলাদা binary feature তৈরি হয়েছে।

---

# 17. Data Leakage Prevention

কিছু columns model feature হিসেবে রাখা হয়নি।

Removed leakage columns:

* `rain_sum`
* `precipitation_sum`
* `precipitation_hours`

কারণ এগুলো target/current rainfall-এর directly related information।

এছাড়া identifier columns model input থেকে বাদ দেওয়া হয়েছে:

* `Date`
* `Station_ID`
* `Division`
* `Station`
* `District`

---

# 18. Feature Matrix এবং Target

দুইটি প্রধান variable তৈরি করা হয়েছে:

### X

* সব selected input features।

### y

* Target rainfall:

  * `rain_sum`

অর্থাৎ:

**X → Weather + Time + Historical Rainfall Features**

**y → Current Rainfall**

---

# 19. Chronological Train/Validation/Test Split

Random split ব্যবহার করা হয়নি।

Date অনুযায়ী split করা হয়েছে:

### Training

* 2016–2022

### Validation

* 2023–2024

### Testing

* 2025

Dataset size:

* Total rows: `124,202`
* Training: `86,938`
* Validation: `24,854`
* Testing: `12,410`

### কেন chronological split?

কারণ rainfall forecasting একটি time-dependent problem।

Future data randomly training-এর মধ্যে চলে গেলে data leakage হতে পারে।

---

# 20. Numeric Data Cleaning

Features-এর সব values numeric format-এ convert করা হয়েছে।

* Invalid numeric values → `NaN`
* Infinite values → `NaN`
* Training data-এর median ব্যবহার করে missing values fill করা হয়েছে।
* Remaining missing values `0` দিয়ে fill করা হয়েছে।
* শেষে finite value check করা হয়েছে।

### Important

Median শুধুমাত্র training data থেকে calculate করা হয়েছে।

এতে validation/test data থেকে information training-এ leak হওয়ার সম্ভাবনা কমে।

---

# 21. Evaluation Metrics

Model performance measure করার জন্য ব্যবহার করা হয়েছে:

### R² Score

Model কতটা variance explain করতে পারছে তা বোঝায়।

* `1.0` → perfect prediction
* `0` → baseline-এর কাছাকাছি
* Negative → poor prediction হতে পারে

### RMSE

Prediction error-এর magnitude measure করে।

### MAE

Average absolute prediction error measure করে।

---

# 22. Prediction Clipping

Rainfall negative হতে পারে না।

তাই prediction-এর পরে:

`prediction = max(prediction, 0)`

ব্যবহার করা হয়েছে।

এর ফলে negative rainfall prediction বাদ দেওয়া হয়।

---

# 23. CatBoost Regression

প্রথম tree-based model:

**CatBoostRegressor**

ব্যবহার করা হয়েছে।

Main configuration:

* Iterations = `1500`
* Learning Rate = `0.03`
* Depth = `8`
* Loss = `RMSE`
* L2 regularization = `5`
* Random seed = `42`
* Early stopping = `80`

---

# 24. CatBoost Normal Target

প্রথমে original rainfall value দিয়েই CatBoost train করা হয়েছে।

Model:

`CatBoost_Normal`

Validation result:

* R² ≈ `0.8273`
* RMSE ≈ `5.836`
* MAE ≈ `1.804`

---

# 25. CatBoost Log Target

Rainfall distribution skewed হওয়ায় log transformation ব্যবহার করে দ্বিতীয় CatBoost model train করা হয়েছে।

Transformation:

`log1p(y)`

Prediction-এর পরে original scale-এ ফেরত আনা হয়েছে:

`expm1(prediction)`

Model:

`CatBoost_Log`

Validation R² ≈ `0.8051`

---

# 26. LightGBM Regression

দ্বিতীয় boosting model:

**LightGBM**

ব্যবহার করা হয়েছে।

দুই version train করা হয়েছে:

* LightGBM Normal
* LightGBM Log

Normal validation R²:

≈ `0.8189`

Log validation R²:

≈ `0.8096`

---

# 27. XGBoost Regression

তৃতীয় boosting model:

**XGBoost**

ব্যবহার করা হয়েছে।

দুই version train করা হয়েছে:

* XGBoost Normal
* XGBoost Log

Normal validation R²:

≈ `0.8185`

Log validation R²:

≈ `0.7999`

---

# 28. Tree Model Comparison

Validation performance compare করা হয়েছে।

Models:

* CatBoost Normal
* CatBoost Log
* LightGBM Normal
* LightGBM Log
* XGBoost Normal
* XGBoost Log

Best validation model:

**CatBoost Normal**

Validation R²:

≈ `0.8273`

---

# 29. LSTM Preparation

Rainfall একটি time-series problem হওয়ায় LSTM model-ও তৈরি করা হয়েছে।

LSTM-এর জন্য:

* Sequence length = `30`
* Previous 30 days-এর information ব্যবহার করা হয়েছে।
* Current day-ও sequence-এর মধ্যে রাখা হয়েছে।

তাই final sequence:

**31 time steps × 132 features**

---

# 30. LSTM Data Scaling

LSTM training-এর আগে `StandardScaler` ব্যবহার করা হয়েছে।

Scaling:

* Mean = 0-এর কাছাকাছি
* Standard deviation = 1-এর কাছাকাছি

Scaler শুধুমাত্র training data দিয়ে fit করা হয়েছে।

---

# 31. LSTM Sequence Creation

প্রতিটি station-এর জন্য chronological sequence তৈরি করা হয়েছে।

Shape:

### Training

`(85918, 31, 132)`

### Validation

`(24854, 31, 132)`

### Testing

`(12410, 31, 132)`

অর্থাৎ:

**31 time steps**

এবং প্রতিটি timestep-এ:

**132 features**

---

# 32. LSTM Architecture

Model architecture:

1. Input Layer
2. LSTM(64)
3. Dropout(0.20)
4. LSTM(32)
5. Dropout(0.20)
6. Dense(16, ReLU)
7. Dense(1)

### Optimizer

* Adam

### Learning rate

* `0.0005`

### Loss

* MSE

### Gradient clipping

* `clipnorm = 1.0`

---

# 33. LSTM Callbacks

### EarlyStopping

* Validation performance improve না করলে training stop করে।
* Best weights restore করে।

### ReduceLROnPlateau

* Validation improvement কমে গেলে learning rate কমিয়ে দেয়।

এতে unnecessary training কমে এবং model optimization improve হতে পারে।

---

# 34. LSTM Normal Model

Original rainfall target দিয়ে LSTM train করা হয়েছে।

Validation:

* R² ≈ `0.7253`
* RMSE ≈ `7.360`
* MAE ≈ `2.272`

---

# 35. LSTM Log Model

Log-transformed rainfall দিয়ে দ্বিতীয় LSTM train করা হয়েছে।

Validation:

* R² ≈ `0.6597`
* RMSE ≈ `8.192`
* MAE ≈ `2.281`

---

# 36. Best Model Selection

সব model-এর validation R² compare করা হয়েছে।

Highest validation R² model নির্বাচন করা হয়েছে।

Selected model:

**CatBoost Normal**

---

# 37. 2025 Test Prediction

Best model ব্যবহার করে unseen 2025 data-তে prediction করা হয়েছে।

Final test performance:

* R² = `0.879588`
* RMSE = `4.254902`
* MAE = `1.720131`

এই test set model training-এর সময় ব্যবহার করা হয়নি।

---

# 38. Actual vs Predicted Visualization

Actual rainfall এবং predicted rainfall compare করার জন্য scatter plot তৈরি করা হয়েছে।

* X-axis → Actual Rainfall
* Y-axis → Predicted Rainfall
* Reference line → `y = x`

Reference line-এর কাছাকাছি prediction হলে actual এবং predicted values-এর difference কম।

---

# 39. Residual Analysis

Residual calculate করা হয়েছে:

**Residual = Actual − Predicted**

Residual analysis-এর মাধ্যমে বোঝা যায় model কোথায় বেশি বা কম prediction করছে।

---

# 40. Feature Importance

Tree-based best model থেকে feature importance বের করা হয়েছে।

Top important features-এর মধ্যে ছিল:

* `weather_code`
* `shortwave_radiation_sum`
* `et0_fao_evapotranspiration`
* `sunshine_duration`
* `Longitude`
* `rain_lag_1`
* `wind_gusts_10m_max`
* `wind_speed_10m_max`
* `Latitude`
* `wind_direction_10m_dominant`

এগুলো model-এর prediction-এ তুলনামূলকভাবে বেশি contribution করেছে।

---

# 41. Heavy Rainfall Evaluation

শুধু heavy rainfall cases আলাদাভাবে evaluate করা হয়েছে।

Condition:

`rainfall >= 50 mm`

Results:

* Number of cases = `199`
* MAE ≈ `16.65 mm`
* RMSE ≈ `21.99 mm`
* R² ≈ `0.00127`

এটি দেখায় যে overall performance ভালো হলেও heavy rainfall event prediction তুলনামূলকভাবে দুর্বল।

---

# 42. Heavy Rainfall Examples

কিছু large rainfall event-এর actual এবং predicted value compare করা হয়েছে।

Example:

* Actual = `168.3 mm`
* Predicted ≈ `109.81 mm`

আরেকটি:

* Actual = `146.1 mm`
* Predicted ≈ `88.54 mm`

অর্থাৎ model কিছু extreme rainfall event significantly underestimate করেছে।

---

# 43. Monthly Analysis

Prediction এবং actual rainfall month অনুযায়ী group করা হয়েছে।

প্রতিটি month-এর জন্য:

* Mean Actual Rainfall
* Mean Predicted Rainfall

calculate করা হয়েছে।

এর মাধ্যমে model seasonal rainfall pattern কতটা capture করছে তা দেখা যায়।

---

# 44. Final Pipeline

পুরো pipeline সংক্ষেপে:

**Raw CSV**

↓

**Data Validation**

↓

**Date Processing**

↓

**Data Cleaning**

↓

**Feature Engineering**

↓

**Lag Features**

↓

**Rolling Features**

↓

**Weather Features**

↓

**Station Encoding**

↓

**Leakage Removal**

↓

**Chronological Split**

↓

**Train / Validation / Test**

↓

**CatBoost / LightGBM / XGBoost**

↓

**LSTM**

↓

**Model Comparison**

↓

**Best Model Selection**

↓

**2025 Test Prediction**

↓

**Performance Evaluation**

↓

**Feature Importance**

↓

**Heavy Rainfall Analysis**

↓

**Monthly Analysis**

---

# 45. Important Methodological Note

এই code-এ previous rainfall এবং previous weather features-এর পাশাপাশি কিছু **same-day weather observations** ব্যবহার করা হয়েছে।

তাই এটিকে strictly “future next-day rainfall forecasting” না বলে:

**daily rainfall estimation/prediction using available weather observations and historical rainfall information**

হিসেবে describe করা বেশি accurate।

---

# 46. Final Model Performance

### Best Validation Model

**CatBoost Normal**

Validation:

* R² ≈ `0.8273`
* RMSE ≈ `5.836`
* MAE ≈ `1.804`

### Final 2025 Test

* R² = `0.8796`
* RMSE = `4.2549`
* MAE = `1.7201`

### Heavy Rainfall

* R² ≈ `0.0013`
* MAE ≈ `16.65 mm`
* RMSE ≈ `21.99 mm`

---

# 47. Main Conclusion

* Multiple ML models এবং LSTM ব্যবহার করে rainfall prediction করা হয়েছে।
* Feature engineering-এর মাধ্যমে temporal এবং seasonal information যুক্ত করা হয়েছে।
* Chronological train-validation-test split ব্যবহার করা হয়েছে।
* CatBoost Normal validation-এ best performance দিয়েছে।
* 2025 unseen test data-তে R² ≈ `0.8796` পাওয়া গেছে।
* তবে extreme/heavy rainfall events-এর prediction accuracy তুলনামূলকভাবে কম।
* তাই overall rainfall prediction এবং extreme rainfall prediction-কে আলাদাভাবে evaluate করা গুরুত্বপূর্ণ।
