# Деревья решений и ансамблевые методы классификации

Исследование алгоритмов на основе деревьев решений: реализация с нуля, сравнительный анализ ансамблевых методов на реальных данных.

## Реализовано

### Decision Tree с нуля (`decision_tree.py`)
- Критерий разбиения: Gini impurity (векторизованное вычисление через кумулятивные суммы)
- Поддержка вещественных и категориальных признаков
- Настраиваемые критерии остановки: `max_depth`, `min_samples_split`, `min_samples_leaf`
- Вычисление feature importances

### Random Forest с нуля
- Бутстрап-выборки + случайное подпространство признаков
- Совместимость с интерфейсом sklearn (`fit`, `predict`, `predict_proba`)

### Сравнительный анализ
- Decision Tree vs Bagging vs Random Forest (кросс-валидация)
- Анализ зависимости качества от числа деревьев
- XGBoost vs CatBoost vs LightGBM (подбор гиперпараметров через RandomizedSearchCV)
- Анализ важности признаков (сводная heatmap по четырём моделям)
- Влияние кодирования категориальных признаков (OHE, Label, Target Encoding)

## Данные

- **Mushroom Dataset** (`agaricus-lepiota.csv`)
- **BNP Paribas Cardif Claims Management** - [Kaggle](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/data) (`train.csv` - скачать и положить в директорию)

## Стек

Python, NumPy, pandas, scikit-learn, XGBoost, CatBoost, LightGBM, Matplotlib