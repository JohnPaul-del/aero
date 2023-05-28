import base64
import datetime
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from scipy.signal import savgol_filter
from sklearn import preprocessing as pre
import numpy as np

from .forms import TaskForm
from .functionst import show_graph_task1, show_graph_task2, show_graph_task3, show_graph_task4


def form_view(request):
    """
    Первая отрисовка страницы
    :param request: GET
    :return: render html
    """
    form = TaskForm()
    context = {'form': form}
    return render(request, 'base.html', context=context)


@ensure_csrf_cookie
def generate_plot(request):
    """
    Отрисовка графиков в зависимости от выбранной задачи
    :param request: POST from form parameters
    :return: img object with graph
    """
    start_date = request.POST.get('start_event')
    end_date = request.POST.get('end_event')
    dep_airports = request.POST.getlist('dep_airport')
    arr_airports = request.POST.getlist('arr_airport')
    classes = request.POST.getlist('classes')
    tasks = request.POST.get('tasks')

    by, bm, bd = make_date(start_date)
    ey, em, ed = make_date(end_date)

    print(tasks)

    if tasks == 'task_1':
        df = show_graph_task1(
            dep_airports[0].upper(),
            arr_airports[0].upper(),
            [1124, 1126, 1128, 1130, 1132, 1134],
            datetime.datetime(by, bm, bd),
            datetime.datetime(ey, em, ed)
        )

        plt.figure(figsize=(12, 6))
        for cl in classes:
            x = df[(df['SEG_CLASS_CODE'] == cl)]['SDAT_S']
            y = df[(df['SEG_CLASS_CODE'] == cl)]['PASS_BK']
            plt.plot(x, y, label=f'Class: {cl}')

        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    elif tasks == 'task_2':
        df, df_seas, df_class, np_str, np_end, n_names = show_graph_task2(
            dep_airports[0].upper(),
            arr_airports[0].upper(),
            [1124, 1126, 1128, 1130, 1132, 1134],
            classes,
            datetime.datetime(by, bm, bd),
            datetime.datetime(ey, em, ed)
        )

        plt.figure(figsize=(12, 6))

        for idx in range(len(n_names)):
            x = df_seas[
                (df_seas['seas'] == n_names[idx]) &
                (df_seas['DD'].isin(pd.date_range(np_str[idx], np_end[idx])))]['DD']
            plt.bar(
                x,
                df_seas[
                    (df_seas['seas'] == n_names[idx]) &
                    (df_seas['DD'].isin(pd.date_range(np_str[idx],
                                                      np_end[idx])))]['PASS_BK'].sum(),
            )
        for cl in classes:
            x = df_class[(df_class['SEG_CLASS_CODE'] == cl)]['DD']
            y = df_class[(df_class['SEG_CLASS_CODE'] == cl)]['PASS_BK']
            plt.plot(x, y, label=f'Class: {cl}')

        plt.legend(loc='right')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    elif tasks == 'task_3':

        df, n_names = show_graph_task3(
            dep_airports[0].upper(),
            arr_airports[0].upper(),
            classes,
            datetime.datetime(by, bm, bd),
            datetime.datetime(ey, em, ed)
        )

        plt.figure(figsize=(12, 6))
        for idx in range(len(n_names)):
            x = df[(df['profile'] == n_names[idx])]['SDAT_S']
            y = df[(df['profile'] == n_names[idx])]['PASS_BK']
            plt.plot(x, y, label=f'Class: {n_names[idx]}')

        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    elif tasks == 'task_4':

        df_seas, df_class, np_str, n_names, np_end = show_graph_task4(
            dep_airports[0].upper(),
            arr_airports[0].upper(),
        )

        for idx in range(len(n_names)):
            x = df_seas[(df_seas['seas'] == n_names[idx]) &
                        (df_seas['DD'].isin(pd.date_range(np_str[idx], np_end[idx])))]['DD']
            plt.bar(x, df_seas[(df_seas['seas'] == n_names[idx]) &
                               (df_seas['DD'].isin(pd.date_range(np_str[idx], np_end[idx])))]['Y_PRED_MODELMLG'].sum(),
                    label=f'Predict {n_names[idx]}'
                    )

        for cl in classes:
            x = df_class[(df_class['SEG_CLASS_CODE'] == cl)]['DD']
            y = df_class[(df_class['SEG_CLASS_CODE'] == cl)]['Y_PRED_MODELMLG']
            y = savgol_filter(y, 60, 1, mode='wrap')
            plt.plot(x. y, label=f'Class (predict) {cl}')

        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        graph_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return JsonResponse(
        {'data': graph_data}
    )


def make_date(date_as_str):
    """
    Make date object from string
    :param date_as_str: Date as string
    :return: date object
    """

    date_as_str = date_as_str.split('-')
    return int(date_as_str[0]), int(date_as_str[1]), int(date_as_str[2])