#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os
import re
from concurrent.futures.thread import ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import arima
from pmdarima.model_selection import train_test_split
from sklearn.metrics import r2_score

province_name = {
    "Anhui": "安徽",
    "Beijing": "北京",
    "Chongqing": "重庆",
    "Fujian": "福建",
    "Gansu": "甘肃",
    "Guangdong": "广东",
    "Guangxi": "广西",
    "Guizhou": "贵州",
    "Hainan": "海南",
    "Hebei": "河北",
    "Heilongjiang": "黑龙江",
    "Henan": "河南",
    "Hong Kong": "香港",
    "Hubei": "湖北",
    "Hunan": "湖南",
    "Inner Mongolia": "内蒙古",
    "Jiangsu": "江苏",
    "Jiangxi": "江西",
    "Jilin": "吉林",
    "Liaoning": "辽宁",
    "Macau": "澳门",
    "Ningxia": "宁夏",
    "Qinghai": "青海",
    "Shaanxi": "陕西",
    "Shandong": "山东",
    "Shanghai": "上海",
    "Shanxi": "山西",
    "Sichuan": "四川",
    "Tianjin": "天津",
    "Tibet": "西藏",
    "Xinjiang": "新疆",
    "Yunnan": "云南",
    "Zhejiang": "浙江",
}


def adjust_date(s):
    t = s.split("/")
    return f"20{t[2]}-{int(t[0]):02d}-{int(t[1]):02d}"


def adjust_name(s):
    return re.sub(r"\*|\,|\(|\)|\*|\ |\'", "_", s)


def draw(province):
    draw_(province, True)
    draw_(province, False)


def draw_(province, isDaily):
    # 模型训练
    model = arima.AutoARIMA(start_p=0, max_p=4, d=None, start_q=0, max_q=1, start_P=0, max_P=1, D=None, start_Q=0, max_Q=1, m=7, seasonal=True, test="kpss", trace=True, error_action="ignore", suppress_warnings=True, stepwise=True)
    if isDaily:
        data = df[province].diff().dropna()
        model.fit(data)
    else:
        data = df[province]
        model.fit(data)

    # 模型验证
    train, test = train_test_split(data, train_size=0.8)
    pred_test = model.predict_in_sample(start=train.shape[0], dynamic=False)
    validating = pd.Series(pred_test, index=test.index)
    r2 = r2_score(test, pred_test)

    # 开始预测
    pred, pred_ci = model.predict(n_periods=14, return_conf_int=True)
    idx = pd.date_range(data.index.max() + pd.Timedelta("1D"), periods=14, freq="D")
    forecasting = pd.Series(pred, index=idx)

    # 绘图呈现
    plt.figure(figsize=(15, 6))

    plt.plot(data.index, data, label="实际值", color="blue")
    plt.plot(validating.index, validating, label="校验值", color="orange")
    plt.plot(forecasting.index, forecasting, label="预测值", color="red")
    # plt.fill_between(forecasting.index, pred_ci[:, 0], pred_ci[:, 1], color="black", alpha=.25)

    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    if isDaily:
        plt.title(f"每日新增预测 - {province_name[province]}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})")
        plt.savefig(os.path.join("figures", f"{adjust_name(province)}-daily.svg"), bbox_inches="tight")
    else:
        plt.title(f"累计确诊预测 - {province_name[province]}\nARIMA {model.model_.order}x{model.model_.seasonal_order} (R2 = {r2:.6f})")
        plt.savefig(os.path.join("figures", f"{adjust_name(province)}.svg"), bbox_inches="tight")


if __name__ == "__main__":
    # 准备数据
    df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", index_col="Province/State").drop(columns=["Lat", "Long"])
    df = df[df["Country/Region"] == "China"].transpose().drop("Country/Region")
    df.index = pd.DatetimeIndex(df.index.map(adjust_date))

    provinces = df.columns.to_list()

    # 线程池
    with ThreadPoolExecutor(max_workers=16) as pool:
        pool.map(draw, provinces)

    # 编制索引
    with codecs.open("README.md", "w", 'utf-8') as f:
        f.write("[![check status](https://github.com/winsphinx/covid-cn/actions/workflows/check.yml/badge.svg)](https://github.com/winsphinx/covid-cn/actions/workflows/check.yml)\n")
        f.write("[![build status](https://github.com/winsphinx/covid-cn/actions/workflows/build.yml/badge.svg)](https://github.com/winsphinx/covid-cn/actions/workflows/build.yml)\n")
        f.write("# COVID-19 Forecasting\n\n")
        for province in provinces:
            f.write(f"## {province_name[province]}\n\n")
            f.write(f"![img](figures/{adjust_name(province)}.svg)\n\n")
            f.write(f"![img](figures/{adjust_name(province)}-daily.svg)\n\n")
