import datetime
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_dataset(start_date, end_date, path='aero/datasets/'):
    """
    this is a preparation for the future dataset loading function
    :param path: path to files
    :param start_date: star date
    :param end_date: end date
    :return: Pandas dataframe objet concatenated from several files
    """

    start_year = start_date.year
    end_year = end_date.year

    start_month = start_date.month
    end_month = end_date.month

    ds_files = []
    for year in range(start_year, end_year + 1):
        if year == end_year:
            close_month = end_month
        else:
            close_month = 12
        for month in range(start_month, close_month + 1):
            ds_files.append(os.path.join(path, 'CLASS_{:02d}{:04d}.pickle'.format(month, year)))
        start_month = 1

    df = pd.concat((
        pd.read_pickle(f) for f in ds_files),
        ignore_index=False)
    df['SDAT_S'] = pd.to_datetime(df['SDAT_S'], format='%d.%m.%Y')
    df['DD'] = pd.to_datetime(df['DD'], format='%d.%m.%Y')

    df.sort_values(by=['SDAT_S', 'DD'], inplace=True)
    return df


def get_properties(dd, n_data_end, names_prop):
    """

    :param dd:
    :param n_data_end:
    :param names_prop:
    :return:
    """

    if isinstance(dd, int):
        curr_dd = dd
    else:
        curr_dd = np.datetime64(dd)

    d_end = n_data_end[
        np.where(n_data_end >= curr_dd)
    ].min()

    try:
        name_p = names_prop[
            np.where(n_data_end == d_end)
        ][0]
    except:
        name_p = ''

    return name_p


def show_graph_task1(dep_air,
                     arr_air,
                     flights,
                     begin_date,
                     end_date, ):
    """
    Определение динамики бронирований рейса в разрезе классов бронирования по вылетевшим рейсам.
    :param dep_air: from airport
    :param arr_air: to airport
    :param flights: list of number of flights
    :param begin_date: date of begin
    :param end_date: date of end
    :return: pandas DataFrame object
    """

    df = get_dataset(begin_date, end_date)
    df = df[(df['FLT_NUM'].isin(flights)) &
            (df['DD'].isin(pd.date_range(begin_date, end_date))) &
            (df['SDAT_S'].isin(pd.date_range(begin_date, end_date))) &
            (df['DTD'] != -1) & (df['SORG'] == dep_air) & (df['SDST'] == arr_air)].sort_values(
        ['DTD', 'SEG_CLASS_CODE'])

    new_df = df.groupby(['SEG_CLASS_CODE', 'SDAT_S']).agg({'PASS_BK': ['sum']}).reset_index()
    return new_df


def show_graph_task2(dep_air,
                     arr_air,
                     flights,
                     classes,
                     begin_date,
                     end_date,
                     seas_path='aero/datasets/seasonality.csv'):
    """
    Определение сезонности спроса по классам бронирования, по вылетевшим рейсам.
    :param seas_path: File of seasonality
    :param dep_air: from airport
    :param arr_air: to airport
    :param flights: list of numbers of flights
    :param classes: list of passengers` classes
    :param begin_date: begin date
    :param end_date: end date
    :return: Three pandas dataframe objects, Numpy arrays of starting date, ending date and list of class names
    """

    df = get_dataset(begin_date, end_date)
    df = df[(df['FLT_NUM'].isin(flights)) &
            (df['DD'].isin(pd.date_range(begin_date, end_date))) &
            (df['SEG_CLASS_CODE'].isin(classes)) & (df['DTD'] != -1) & (df['SORG'] == dep_air) &
            (df['SDST'] == arr_air)].sort_values(['DTD', 'SEG_CLASS_CODE'])

    df_ses = pd.read_csv(seas_path)
    df_ses['DAT_BEGIN'] = pd.to_datetime(df_ses['DAT_BEGIN'], format='%d.%m.%Y')
    df_ses['DAT_END'] = pd.to_datetime(df_ses['DAT_END'], format='%d.%m.%Y')
    df_ses = df_ses[(df_ses['AEROPORT'].isin([dep_air, arr_air]))]

    np_str = df_ses['DAT_BEGIN'].to_numpy()
    np_end = df_ses['DAT_END'].to_numpy()
    n_names = df_ses['NAME_S'].to_numpy()

    df['seas'] = df['DD'].apply(func=get_properties, args=(np_end, n_names))

    df_seas = df.groupby(['DD', 'seas']).agg({'PASS_BK': ['sum']}).reset_index()
    df_class = df.groupby(['DD', 'SEG_CLASS_CODE']).agg({'PASS_BK': ['sum']}).reset_index()

    return df, df_seas, df_class, np_str, np_end, n_names


def show_graph_task3(dep_air,
                     arr_air,
                     classes,
                     begin_date,
                     end_date,
                     profiles_path='aero/datasets/profiles.csv'):
    """
    Определение профилей спроса в разрезе классов бронирования, по вылетевшим рейсам.
    :param profiles_path: file of profiles
    :param dep_air: from airport
    :param arr_air: to airport
    :param classes: list of classes
    :param begin_date: begin date
    :param end_date: end data
    :return: pandas DataFrame object, list of names of profiles
    """

    df = get_dataset(begin_date, end_date)
    df = df[(df['SDAT_S'].isin(pd.date_range(begin_date, end_date))) &
            (df['DTD'] != -1) &
            (df['SEG_CLASS_CODE'].isin(classes)) &
            (df['SORG'] == dep_air) &
            (df['SDST'] == arr_air)
            ].sort_values(['DTD', 'SEG_CLASS_CODE'])

    df_pfl = pd.read_csv(profiles_path, sep=',')
    n_max_interval = df_pfl['MAX_P'].to_numpy()
    n_names = df_pfl['NAME_P'].to_numpy()

    df['profile'] = df['DTD'].apply(func=get_properties,
                                    args=(n_max_interval, n_names))

    df_pf = df.groupby(
        ['SDAT_S', 'profile']
    ).agg({'PASS_BK': ['sum']}).reset_index()

    return df_pf, n_names


def get_schedule_df(path_df='aero/datasets/'):
    """
    функция загружает датасет по расписанию.
    :return: датафрейм с расписанием
    """
    df_rasp = pd.read_csv(path_df, index_col=0)

    df_rasp['DD'] = pd.to_datetime(df_rasp['DD'], format='%Y%m%d').dt.strftime('%d.%m.%Y')
    df_rasp['DD'] = pd.to_datetime(df_rasp['DD'], format='%d.%m.%Y')
    # df_rasp['EFFV_DATE'] = pd.to_datetime(df_rasp['EFFV_DATE'], format='%Y%m%d').dt.strftime('%d.%m.%Y')
    # df_rasp['DISC_DATE'] = pd.to_datetime(df_rasp['DISC_DATE'], format='%Y%m%d').dt.strftime('%d.%m.%Y')

    df_rasp['DAY_NBR_OF_YEAR'] = df_rasp['DD'].dt.dayofyear

    return df_rasp


def get_history_df(year):
    """
        функция получения датасета для модели. На вход подается год с историческими данными по которым будем предсказывать
        данные.
        может используется для проверки эффективности модели - можно будет сравнить данные по факту и по прогнозу.


        :param year: год для определения файлов. По-умолчанию 2019
        :return: возвращает готовый датафрейм для подачи в модель
        """

    # коментарий для экспертов:
    # есть уже подготовленный файл, который можно получить по ссылки
    # для этой схемы достаточно раскомментировать код ниже. файл запакован и не имеет расширение csv (!)
    # load_zip('https://drive.google.com/uc?export=download&confirm=no_antivirus&id=1huGfTjF1EhvESdxx10u9eHWQCkWEnDqG')
    # dataset_to_model = pd.read_csv('./datasets/dataset_19_to_model', index_col=0)
    # return dataset_to_model

    # Период вылетов для отбора файлов
    BEGIN_DD = datetime.datetime(year, 1, 1)
    END_DD = datetime.datetime(year, 12, 31)

    # df = getDataSet_Zad(BEGIN_DD, END_DD)  # получение из csv файлов
    df = get_pickle_df(BEGIN_DD, END_DD)  # получение из PICKLE файлов

    dataset_to_model = add_to_df(df)

    return dataset_to_model


def get_pickle_df(start_data, end_data, path='aero/datasets'):
    """
    Данные считывает из piсkle файлов: быстрее чем CSV на 30-50%

    :param start_data:
    :param end_data:
    :return: возвращает датафрейм в формате Pandas
    """

    path_pickle = path + '/PICKLE' # путь к папке c файлами CLASS в формате PICKLE

    year_s = start_data.year
    year_e = end_data.year

    all_files = []
    month_s = start_data.month
    month_e = end_data.month
    for year in range(year_s, year_e + 1):
        if year == year_e:
            month_end = month_e
        else:
            month_end = 12
        for month in range(month_s, month_end + 1):
            all_files.append(os.path.join(path_pickle, "CLASS_{:02d}{:04d}.pickle".format(month, year)))

        month_s = 1

    pd_list = []
    for file_p in all_files:
        with open(file_p, 'rb') as f:
            pd_list.append(pickle.load(f))

    df = pd.concat(pd_list, ignore_index=True)
    df.sort_values(by=['SDAT_S', 'DD'], inplace=True)
    return df


def add_to_df(df):
    """
    функция добавляет/трафсморфирует данные для модели
    :param df: входящий датафрейм
    :return: насыщенный и изменный датафрейм для модели
    """
    # Сгенерируем признак дней недели (0-понедельник, 1-вторник, 2-среда и т.д)
    df['WEEK_DAY'] = df['DD'].dt.weekday

    # Сгенерируем признак недели года (1-ая, 2-ая, 3-я и т.д.)
    df['WEEK'] = df['DD'].dt.week

    # Сгенерируем признак порядкового номера дня в году от 1 до 365(366)
    df['DAY_NBR_YEAR'] = df['DD'].dt.dayofyear

    # Создание признака маршрут
    df['ROUTE'] = df['SORG'] + '-' + df['SDST']

    # Признак дня недели выходной или рабочий
    df['WKND'] = df.WEEK_DAY.apply(lambda x: x < 5).astype(float)

    # Коэффициент забронированных к доступным местам
    df['DIV_PASS_BK_AU'] = round(df['PASS_BK'] / (df['AU'] + 1e-3))

    # Показатель доступности класса
    df['MULT_PASS_BK_FCLCLD'] = df['PASS_BK'] * df['FCLCLD']

    # Показатель относительной неявки на рейс
    df['DIV_NS_PASS_DEP'] = round(df['NS'] / (df['PASS_DEP'] + 1e-3))

    # Относительный показатель забронированных к вылетевшим
    df['DIV_PASS_BK_DEP'] = round(df['PASS_BK'] / (df['PASS_DEP'] + 1e-3))

    # Посмотрим на уникальные значения
    df['PASS_BK'].unique(), df['PASS_BK'].nunique()

    df['SEG_CLASS_CODE'].unique(), df['SEG_CLASS_CODE'].nunique()

    df['FLT_NUM'].unique(), df['FLT_NUM'].nunique()

    # Словари для классификации/категоризации признаков
    # Для классов бронирования
    rbd_map = {rbd: i for i, rbd in enumerate(df['SEG_CLASS_CODE'].unique())}

    # Отразим значения в столбце 'class_station'
    df['RBD_CAT'] = df['SEG_CLASS_CODE'].map(rbd_map)

    # Отразим значения в столбце 'class_station'
    df['RBD_CAT'] = df['SEG_CLASS_CODE'].map(rbd_map)

    # Отразим значения в столбце 'class_station'
    df['RBD_CAT'] = df['SEG_CLASS_CODE'].map(rbd_map)

    # Перевод в категории столбец датафрейма 'ROUTE'
    le = LabelEncoder()
    df['ROUTE_CAT'] = le.fit_transform(df['ROUTE'])

    # Перевод в категории столбец датафрейма 'SSCL1',	'SEG_CLASS_CODE'
    df['SSCL1_CAT'] = le.fit_transform(df['SSCL1'])
    df['SEG_CLASS_CODE_CAT'] = le.fit_transform(df['SEG_CLASS_CODE'])

    # Удаляем лишние колонки в датафрейме
    df.drop(['SDAT_S', 'DD', 'SAK', 'NBCL', 'SORG', 'SDST', 'SSCL1', 'SEG_CLASS_CODE', 'ROUTE'], axis=1, inplace=True)
    df.head(3)

    return df


def get_data_set_for_predict(df_history, rasp):
    """
    функция подготавливает датасет для модели для предсказания. Ключевые показатели заполняются из исторических данных.
    за какой период будут исторические данные - на такой период будет сделано предсказания.
    например, если нужно предсказать на 2020 год, то в качестве исторических данным нужно подать
    данные за 2019 год или 2018 год
    :param df_history: исторические данные для формирования прогноза.
    :param rasp: раписание вылетов
    :return: датафрейм для подачи в модель для предсказания
    """

    # print(rasp.FLT_NUMSH.nunique(), df_history.FLT_NUM.nunique())  # к-во уникальных рейсов в расписании и в историческом датасете

    # нужно проверить года на високосность: 2020 год - високосный = 366 дней (!)
    days_year_pred = rasp.DAY_NBR_YEAR.nunique() # к-во дней по годам в данных
    days_year_hist = df_history.DAY_NBR_YEAR.nunique()  # к-во дней по годам в данных

    need_add_day = True if days_year_pred>days_year_hist else False

    # Создание колонки день года в соответствии с форматом данных для ML (от 0 до 365(366))
    # Сгенерируем признак порядкового номера дня в году
    rasp['DAY_NBR_OF_YEAR'] = rasp['DD'].dt.dayofyear

    # Создание новой таблицы с пустым датафреймом
    dataset_predict_to_model = pd.DataFrame(columns=df_history.columns)

    # Получение уникальных значений FLT_NUMSH и DAY_NBR_YEAR из таблицы rasp
    flt_numsh_values = rasp['FLT_NUMSH'].unique()
    effv_date_yday_values = rasp['DAY_NBR_YEAR'].unique()

    # Фильтрация таблицы df_history по условию совпадения значений FLT_NUM и DAY_NBR_YEAR
    filtered_dataset = df_history[
        (df_history['FLT_NUM'].isin(flt_numsh_values)) &
        (df_history['DAY_NBR_YEAR'].isin(effv_date_yday_values))
        ]

    # Словарь для изменения типа данных в dataset_predict_to_model
    data_types = {
        'FLT_NUM': int,
        'SEG_NUM': int,
        'FCLCLD': int,
        'PASS_BK': int,
        'SA': int,
        'AU': int,
        'PASS_DEP': int,
        'NS': int,
        'DTD': int,
        'WEEK_DAY': int,
        'WEEK': int,
        'DAY_NBR_YEAR': int,
        'WKND': int,
        'DIV_PASS_BK_AU': float,
        'MULT_PASS_BK_FCLCLD': float,
        'DIV_NS_PASS_DEP': float,
        'DIV_PASS_BK_DEP': float,
        'RBD_CAT': int,
        # 'PASS_BK_CAT': int,
        # 'FLT_NUM_CAT': int,
        'ROUTE_CAT': int,
        'SSCL1_CAT': int,
        'SEG_CLASS_CODE_CAT': int
    }

    # Применение словаря с типами данных к dataset_20_to_model
    dataset_predict_to_model = dataset_predict_to_model.astype(data_types)

    # Добавление отфильтрованных строк в таблицу dataset_20_to_model
    dataset_predict_to_model = dataset_predict_to_model.append(filtered_dataset, ignore_index=True)

    # Проверка наличия данных на поледний день года, данных нет
    if need_add_day:
        # данный блок не продуман до конца. Пока определяли визуально. Автоматизировать этот процесс не успели
        # суть: нужно подобрать максимльно близки по праметрам рейс.
        # dataset_predict_to_model[dataset_predict_to_model['DAY_NBR_YEAR'] == 366]
        # # В расписании а  366-й день 2020 года, это 21 рейс
        # rasp[rasp['DAY_NBR_OF_YEAR'] == 366].count()
        # Смотрим какой рейс ближе по параметрам времени вылета, это рейс 1140
        # rasp[rasp['DAY_NBR_OF_YEAR'] == 366]
        # # У нас в исторических данных за 2019 нет информации о 1772 рейсе с вылетом в последний день года.
        # print(dataset_predict_to_model[dataset_predict_to_model['FLT_NUM'] == 1772])
        # # Однако в расписании 2020 года рейс стоит
        # rasp[rasp['FLT_NUMSH'] == 1772]

        # пока решили просто скопировать вылет за 365 день в 366 день. Это не лучшее решение, но пока только его смогли реализовать
        # Соберем 20 рейсов за 365-й день и скопируем их на 366-й
        rasp_last_day = rasp[rasp['DAY_NBR_OF_YEAR'] == 366]
        dataset_last_day = df_history[(df_history['FLT_NUM'].isin(rasp_last_day['FLT_NUMSH'])) & (
                    df_history['DAY_NBR_YEAR'] == 365)]
        # Присвоим значение 366 колонке dataset_20_last_day
        dataset_last_day['DAY_NBR_YEAR'] = int(366)

        # Соединяем данные за 366-й день с остальными данными
        merged_dataset = pd.concat([dataset_predict_to_model, dataset_last_day], ignore_index=True)

        # Найти строки с рейсом 1140 и 366 номером дня
        flight_1140_day_366 = merged_dataset[
            (merged_dataset['FLT_NUM'] == 1140) & (merged_dataset['DAY_NBR_YEAR'] == 366)].copy()

        # Заменить номер рейса на 1772 в скопированных строках
        flight_1140_day_366['FLT_NUM'] = 1772

        # Присоединить скопированные строки к merged_dataset
        dataset_predict_to_model = pd.concat([merged_dataset, flight_1140_day_366], ignore_index=True)

    return dataset_predict_to_model


def get_predicted_dataset():
    """

    :return:
    """
    df_sched = get_schedule_df()
    df_hist = get_history_df(year=2019)

    df = get_data_set_for_predict(df_hist, df_sched)

    return df




