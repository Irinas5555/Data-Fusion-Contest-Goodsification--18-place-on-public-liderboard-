# Data Fusion Contest Goodsification (18 place on public liderboard)
Командное решение задачи на соревновании от ВТБ: Data Fusion Contest: https://boosters.pro/championship/data_fusion/data .
Решение в команде с @livington.

Необходимо разработать алгоритм определения категории товара по данным из чеков, смоделированных в соответствии с реальным профилем данных ОФД. Все зависимости между различными товарами в чеках сохранены. Тренировочный датасет содержит более 8 000 000 уникальных чеков, из них около 800 000 чеков размечены и около 7 000 000 чеков без разметки. Товары без определенной категории отмечены "-1". Тестовый датасетсостоит из ~400 000 чеков, метрика считается только на уникальных item name, которые отсутствуют в тренировочном датасете.

Признаки:

receipt_id — id чека; receipt_dayofweek — день недели; receipt_time — время создания чека; item_name — наименование товара; item_quantity — количество товара; item_price — цена товара; item_nds_rate — ставка НДС; category_id — категория товара.

Идея нашего решения заключается в следующем:

1. Строим предсказания классов на основе только наименований товаров (word2vec+tfidf+LinearSVC).
2. С помощью данной модели прогнозируем классы для всех строк исходного датасета (в том числе дубли и неразчеченные).
3. Имеющиеся категории товаров делим на укрупненные группы (все, что связано с автомобилями, канцелярия и печатная продукция, кафе/рестораны, продуктовые магазины и т.д.)
4. Для каждого чека находим наиболее популярную группу товаров и делаем из нее отдельную фичу (так мы можем разделить, например, рестораны и продуктовые магазины)
5. Добавляем ранее сгенерированные агрегированные признаки по категориям чеков
6. И все это подаем в catboost.

"Part1_preprocessing+ LinearSVC.ipynb" - предварительная обработка данных и построение базовой модели (word2vec+tfidf+LinearSVC) на основе наименований.
"Part2_receipt_type.ipynb" - построение прогноза для всего первоначального массива данных, объединение категорий в группы, формирование признака "тип чека",
"Part3_Catboost.ipynb" - итоговая модель catboost
"submission" - вариант сабмита.
