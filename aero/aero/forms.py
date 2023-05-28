from django import forms

AIRPORTS = (
    ('aer', 'AER'),
    ('asf', 'ASF'),
    ('svo', 'SVO')
)

CLASSES = (
    ('B', 'B'), ('C', 'C'), ('D', 'D'), ('E', 'E'), ('G', 'G'),
    ('H', 'H'), ('I', 'I'), ('J', 'J'), ('K', 'K'), ('L', 'L'),
    ('M', 'M'), ('N', 'N'), ('O', 'O'), ('P', 'P'), ('Q', 'Q'),
    ('R', 'R'), ('T', 'T'), ('U', 'U'), ('V', 'V'), ('X', 'X'),
    ('Y', 'Y'), ('Z', 'Z')
)

TASKS = (
    ('task_1', 'Task 1'),
    ('task_2', 'Task 2'),
    ('task_3', 'Task 3'),
    ('task_4', 'Task 4'),
)


class TaskForm(forms.Form):

    """
    Основная форма, которая используется для создания запросов
    """

    tasks = forms.ChoiceField(choices=TASKS,
                              required=False,
                              label='Choice Variant of graph',
                              widget=forms.Select(
                                  attrs={'class': 'form-select'}
                              ))

    dep_airport = forms.ChoiceField(choices=AIRPORTS,
                                    required=False,
                                    label='Departure Airport',
                                    widget=forms.SelectMultiple(
                                        attrs={'class': 'form-select'}
                                    ))
    arr_airport = forms.ChoiceField(choices=AIRPORTS,
                                    label='Arrival Airport',
                                    widget=forms.SelectMultiple(
                                        attrs={'class': 'form-select'}
                                    ))
    start_event = forms.DateField(label='Start Date',
                                  widget=forms.DateInput(
                                      attrs={'class': 'form-control',
                                             'type': 'date'}
                                  ))
    end_event = forms.DateField(label='End Data',
                                widget=forms.DateInput(
                                    attrs={'class': 'form-control',
                                           'type': 'date'}
                                ))
    classes = forms.ChoiceField(choices=CLASSES,
                                required=False,
                                label='Classes',
                                widget=forms.SelectMultiple(
                                    attrs={'class': 'form-select'}
                                ))
