<p align="center">
  <img src="https://user-images.githubusercontent.com/61945327/201784025-893efbea-a00d-4353-964f-f8e28a61e21f.png" height="100">
</p>
<p align="center">
  <a href="https://github.com/maksimio/csidata">CSI data</a> |
  <a href="https://github.com/maksimio/csidealer">CSI dealer</a> |
  <a href="https://github.com/maksimio/csiopenwrt">CSI openwrt</a>
</p>

# О проекте
Проект посвящен классификации / позиционированию при помощи маршрутизаторов Wi-Fi. Благодаря многолучевому распространению сигнала, а также дифракции и интерференции мы можем извлечь из него информацию об окружающей обстановке. Ее можно получить через простейшую метрику - RSSI (мощность сигнала). Я использую Channel State Information ([CSI](https://en.wikipedia.org/wiki/Channel_state_information), информация о состоянии канала). Это матрица комплексных чисел, описывающая амплитуды и фазы поднесущих OFDM-модулированного сигнала (поддерживается MIMO). CSI появилась в Wi-Fi стандарте 802.11n от 2009 года. Она рассчитывается на приемнике после принятия пакета.

## Структура проекта
Этот проект разделен на несколько репозиториев:
1. 📚Этот репозиторий - центральный репозиторий, в котором ведется работа с данными и строятся ML модели, а также приводится полезная информация и ссылки на остальные репозитории. Ссылка на него размещается в научных работах и презентациях
2. 🌑[csi_classification](https://github.com/maksimio/csi_classification) - старые исследования по теме, на которые я ссылаюсь в статьях
3. 📂[csidata](https://github.com/maksimio/csidata) - данные экспериментов с детальным описанием, а также общая информация об организации экспериментов и настройке маршрутизаторов
4. 📈[csidealer](https://github.com/maksimio/csidealer) - клиент и сервер для удобной работы с CSI в реальном времени
5. 📑[csiopenwrt](https://github.com/maksimio/csiopenwrt) - все, что связано с прошивкой OpenWRT для сбора CSI с акцентом на модель маршрутизатора TL-WR842NDv2. Сброс прошивки + ПО для сброса, сборка из исходников OpenWRT с добавлением функционала, готовые прошивки

## Научные работы
По данной теме [можно найти](https://scholar.google.com/scholar?hl=ru&as_sdt=0%2C5&q=channel+state+information+wi-fi&oq=channel+state+information) большое количество научных работ.

## Полезные ссылки
1. [csiread](https://github.com/citysu/csiread) - библиотека, которая очень быстро декодирует dat-файлы с CSI внутри. Сильно превосходит по скорости мое решение в [csi_classification](https://github.com/maksimio/csi_classification)
2. [Проект Atheros CSI Tool](https://github.com/xieyaxiongfly/Atheros-CSI-Tool-UserSpace-APP) - чтение csi при помощи MATLAB. [Это](https://wands.sg/research/wifi/AtherosCSI/) один из самых ранних проектов по CSI, из которого вырос мой проект. В списке репозиториев автора присутствуют собранные прошивки OpenWRT и другие полезные вещи по проекту. К сожалению, готовые прошивки устарели и содержат баги
3. [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) - работа с CSI через ESP32 Wi-Fi
4. [CSIKit](https://github.com/Gi-z/CSIKit) - библиотека для чтения, обработки и визуализации CSI с относительно большой докуменацией. Также содержит ссылки на другие репозитории и сайты по тематике CSI
