import pandas as pd
from datetime import datetime, timedelta


def fetch_vix_10y(output_file="vix_10y.csv"):
    # FRED 的 VIX 日收盘数据
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
    # 计算近十年区间
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=365 * 10)
    # 读取数据
    df = pd.read_csv(url)
    # 标准化列名
    df.columns = ["DATE", "VIXCLS"]
    # 转换类型
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["VIXCLS"] = pd.to_numeric(df["VIXCLS"], errors="coerce")
    # 过滤近十年
    df = df[(df["DATE"].dt.date >= start_date) & (df["DATE"].dt.date <= end_date)]
    # 删除缺失值
    df = df.dropna(subset=["VIXCLS"]).reset_index(drop=True)
    # 保存为 CSV
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Saved {len(df)} rows to {output_file}")
    print(df.tail())


if __name__ == "__main__":
    fetch_vix_10y()
