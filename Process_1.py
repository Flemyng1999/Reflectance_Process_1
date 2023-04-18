import VIs
import VegeDivision
import ROI_Cutting
import Clean_data
import Analyze_ROI
import Tiff2ColorMap
import datetime
import os


def main(path_):
    start_time = datetime.datetime.now()
    VIs.main(path_)
    end_time = datetime.datetime.now()
    print("VIs耗时: {}秒".format(end_time - start_time))

    start_time = datetime.datetime.now()
    VegeDivision.main(path_)
    end_time = datetime.datetime.now()
    print('\n'+"VegeDivision耗时: {}秒".format(end_time - start_time))

    start_time = datetime.datetime.now()
    ROI_Cutting.main(path_)
    end_time = datetime.datetime.now()
    print('\n'+"ROI_Cutting耗时: {}秒".format(end_time - start_time))

    start_time = datetime.datetime.now()
    Clean_data.main(path_)
    end_time = datetime.datetime.now()
    print('\n'+"Clean_data耗时: {}秒".format(end_time - start_time))

    start_time = datetime.datetime.now()
    Analyze_ROI.main(path_)
    end_time = datetime.datetime.now()
    print('\n'+"Analyze_ROI耗时: {}秒".format(end_time - start_time))

    start_time = datetime.datetime.now()
    Tiff2ColorMap.main(path_)
    end_time = datetime.datetime.now()
    print('\n'+"Tiff2Colormap耗时: {}秒".format(end_time - start_time))


if __name__ == '__main__':
    disk1 = 'D:'
    disk2 = 'E:'
    path = ["2022_7_16_sunny"]
    # path = ["2022_7_5_sunny", "2022_7_9_cloudy", "2022_7_12_sunny",
    #         "2022_7_13_cloudy", "2022_7_16_sunny", "2022_7_20_sunny",
    #         "2022_7_23_sunny", "2022_7_27_sunny", "2022_8_2_sunny",
    #         "2022_8_9_cloudy", "2022_8_13_cloudy", "2022_8_14_sunny",
    #         "2022_8_16_sunny", "2022_8_20_sunny", "2022_8_24_cloudy"]

    for i in range(len(path)):
        if i < 9:
            main(os.path.join(disk1, path[i]))
        else:
            main(os.path.join(disk2, path[i]))