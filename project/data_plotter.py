from typing import Dict, Any, DefaultDict, List, Tuple

from bokeh.layouts import grid
from bokeh.models import DatetimeTickFormatter, ColumnDataSource
from bokeh.palettes import Colorblind8
from bokeh.plotting import figure, curdoc, Figure, save
from bokeh.transform import cumsum


class Plotter:
    def __init__(self, title: str, chart_titles: dict):
        self.title = title
        self.chart_titles = chart_titles
        self.figures_time_series = []
        self.figures_counter_plot = []

    def lines_plot(self, data: DefaultDict[str, Tuple[List, List]], title: str, name: str) -> Figure:
        """
        Построить график временной последовательности
        :param data: данные для построения графика
        :param title: заголовок графика
        :param name: имя источника данных
        :return: график готовый для отображения
        """
        # create a new plot with a title and axis labels
        p = figure(title=title, plot_width=800, plot_height=250, y_axis_label='quantity')
        for i, (source_name, fig_source) in enumerate(data.items()):
            # add a line renderer with legend and line thickness to the plot
            p.line(fig_source[0], fig_source[1], legend_label=source_name, alpha=0.7,
                   line_color=(Colorblind8[i]), line_width=2)
        self.style(p, name, 'top_left')
        return p

    def time_series_plot(self, data: Dict[str, ColumnDataSource], title: str, name: str) -> Figure:
        """
        Построить график временной последовательности
        :param data: данные для построения графика
        :param title: заголовок графика
        :param name: имя источника данных
        :return: график готовый для отображения
        """
        # create a new plot with a title and axis labels
        p = figure(title=title, plot_width=800, plot_height=250, y_axis_label='quantity')
        p.xaxis.formatter = DatetimeTickFormatter(days=['%d-%b-%y'])
        for i, (source_name, fig_source) in enumerate(data.items()):
            # add a line renderer with legend and line thickness to the plot
            p.line('x', 'y', source=fig_source, legend_label=source_name, alpha=0.7,
                   line_color=(Colorblind8[i]), line_width=2)
        self.style(p, name, 'top_left')
        return p

    def counter_plot(self, data: Dict[str, Any], title: str, name: str) -> Figure:
        """
        Построить счетчик в виде бублика (donut)
        :param data: данные для построения графика
        :param title: заголовок графика
        :param name: имя источника данных
        :return: график готовый для отображения
        """
        p = figure(plot_height=350, title=title, toolbar_location=None,
                   tools='hover', tooltips='@name: @value', x_range=(-0.27, 0.7))

        p.annular_wedge(x=0, y=1, inner_radius=0.1, outer_radius=0.25, start_angle=cumsum('angle', include_zero=True),
                        end_angle=cumsum('angle'), line_color='white', fill_color='color', legend_field='legend',
                        source=data)

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None
        self.style(p, name)
        return p

    @staticmethod
    def style(p: Figure, name: str, leg_loc: str = 'top_right'):
        """
        Задание стиля для графика
        :param p: график для отображения
        :param name: имя источника данных
        :param leg_loc: положение легенды
        """
        p.legend.location = leg_loc

        if name.find('reasons') == -1:
            p.legend.title = 'Источник данных'
        else:
            p.legend.title = 'Причина блокировки'
        p.legend.title_text_font_style = 'bold'
        p.legend.title_text_font_size = '12px'

    def plot(self, lines: DefaultDict[str, DefaultDict[str, Tuple[List, List]]]):
             # time_series: DefaultDict[str, Dict[str, ColumnDataSource]],
             # counters: DefaultDict[str, Dict[str, List[Any]]]):
        """
        Преобразовать заголовок в удобочитаемый вид
        Построить график
        Скомпоновать графики для последующего отображения по их типу
        Определить положение графиков на выходном листе
        """

        for name, series_data in lines.items():
            title = self.get_chart_title(name)
            p = self.lines_plot(series_data, title=title, name=name)
            self.figures_time_series.append(p)
        if len(self.figures_time_series) % 2 != 0:
            self.figures_time_series.append(None)

        # for name, series_data in time_series.items():
        #     title = self.get_chart_title(name)
        #     p = self.time_series_plot(series_data, title=title, name=name)
        #     self.figures_time_series.append(p)
        # if len(self.figures_time_series) % 2 != 0:
        #     self.figures_time_series.append(None)
        #
        # for name, static_data in counters.items():
        #     title = self.get_chart_title(name)
        #     p = self.counter_plot(static_data, title=title, name=name)
        #     self.figures_counter_plot.append(p)
        # if len(self.figures_counter_plot) % 3 != 0:
        #     for _ in range(3 - len(counters) % 3):
        #         self.figures_counter_plot.append(None)

    def get_chart_title(self, name: str) -> str:
        source = ''
        if '+' in name:
            splitter_pos = name.find('+')
            source = f', по источнику: {name[splitter_pos+1:]}'
            name = name[:splitter_pos]

        return self.chart_titles[name] + source \
            if name in self.chart_titles else str(name).capitalize().replace('_', ' ')

    def get_grids(self) -> List[grid]:
        return [grid(self.figures_time_series, ncols=2, sizing_mode='scale_width'),
                grid(self.figures_counter_plot, ncols=3, sizing_mode='scale_width')]

    def save_layout(self, filename: str):
        """
        Сохранить итоговые графики в виде html страницы
        :param filename: имя файла *.html
        """
        save(self.get_grids(), filename, title=self.title)

    def curdoc_layout(self):
        """
        Отобразить итоговые графики в разворачиваемом веб-приложении на базе tornado
        """
        curdoc().title = self.title
        for sub_layout in self.get_grids():
            curdoc().add_root(sub_layout)
