# nlp_project
[Тетрадка](nlp_evaluation.ipynb) с запуском на тестовых данных.

## ABTE
Для выделения аспектов мы пользовались натренированной на русском `RuBert`, и дообучали её двумя способами. Первое - полноценный файнтьюн на наших данных. Второе - дообучение последнего слоя через библиотеку [`adapters`](https://docs.adapterhub.ml/model_overview.html). Скелет модели и идею обучения мы взяли [отсюда](https://github.com/nicolezattarin/BERT-Aspect-Based-Sentiment-Analysis/tree/main), и переделали саму модель, чтобы она могла работать с токенизатором, который умеет в спаны, и работала с нашим количеством классов.
Внутренности модели лежат в `abte.py`.

Для Берта не подходят спаны, которыми отмечены фрагменты текста в нашем датасете. Берт требует последовательность токенов + такой же длины последовательность индексов классов.  Функция `clean_data` оформляет данные в датафрейм, `spans_to_ids` - переводит спаны и тексты в последовательности токенов, `predictor` - делает предсказание по тексту и сразу переводит его обратно в спаны и записывает в файл в нужном формате. Все вспомогательные функции ~~и прочие ухищрения~~ лежат в [utils_for_abte.py].

--> [Тетрадка](nlp_project_abte.py), в которой всё собиралось, училось и тестилось. Мы прогоняли обучение со скедулером на 10 эпохах, т.к. на меньшем количестве эпох качество не успевало подняться.

[Вот](https://drive.google.com/file/d/10dWiPoGRqGP2bjYbA5FWCqPOp9ek65Jk/view?usp=sharing) сравнительный график обучения двух моделей.

NB: лосс на трейне считался через `CrossEntropyLoss` над последовательностями токенов, без перевода в спаны. То есть Берт решал задачу классификации для токенов по одному, а не их последовательностей. Склеивались токены в последовательности уже в пост-обработке на `predictor`, где проверялось, есть ли стыкующиеся токены с одинаковым классом, и все такие объединялись в последовательности.

[Здесь](https://drive.google.com/drive/folders/1qlgDgESbVsTUKmwEXEDl62LmSrhOpZf5?usp=sharing) лежат веса, лоссы моделей и предсказания для ABTE.

На тетрадке с тестовыми данными видно, что обе модели просто жутко глупенькие, но адаптер чуть-чуть менее глупый, чем простой файнтьюн.

## ABSA
Мы использовали два способа:
1. out-of-the-box решение с помощью [PyABSA](https://pyabsa.readthedocs.io/en/latest/#): см. `absa_pyabsa.py` или [тетрадку](https://www.kaggle.com/code/smthhappens/nlp-project-absa).
2. при помощи BERT и маскирования аспектных слов: см. `absa_bert.py` или [тетрадку](https://colab.research.google.com/drive/1mtp-J_70KhlKtId0quk6fyw2onOROrUI?usp=sharing).

**Проблемы PyABSA**:
1. Есть только три класса тональности (нет *both*).
2. ~~Нельзя~~ Очень сложно дообучить (свой формат датасетов...).

**Проблемы BERT**:
1. Долго
2. Нельзя большие батчи, так как иначе ни одно GPU не выдерживает.
3. В какой-то момент начинаются беды с лоссом, потому что он стагнируется.

[Здесь](https://drive.google.com/drive/folders/14UNHmf1qIIm9TXMEogwSKYSNhsU36YB5?usp=sharing) лежат модели и нужные данные.


### Задачи
- Тональность по категориям - Поля Карпова
- ABTE - Даша Сидоркина
- ABSA - Катя Козлова
