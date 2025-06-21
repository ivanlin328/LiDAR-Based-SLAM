# **Robot Mapping and Localization Project**

本專案使用輪式機器人的編碼器、IMU、LiDAR 和 Kinect 感測器數據，進行機器人定位 (SLAM)、佔據柵格地圖建置、ICP 及 GTSAM 進行姿態圖優化。

## **專案架構**

```
📂 project_root
 ├── load_data.py             # 加載感測器數據
 ├── part1.py                 # 利用編碼器與 IMU 計算機器人軌跡
 ├── scan_matching.py         # 透過 ICP 進行雷射點雲匹配
 ├── scan_matching2.py        # 改進版 ICP 演算法
 ├── Occupancy.py             # 佔據柵格地圖建構
 ├── pose_graph_optimization.py # 姿態圖優化 (使用 GTSAM)
 ├── pr2_utils.py             # 輔助函數：Bresenham's 演算法、Lidar 顯示等
 ├── test_gtsam.py            # 測試 GTSAM 安裝與基本功能
 ├── texture.py               # 生成地圖的紋理貼圖
 ├── test.py                  # 讀取 RGB-D 圖像並轉換為點雲
 ├── test_icp.py              # 測試 ICP 演算法
 ├── README.md                # 本文件
```

---

## **安裝依賴套件**

請確保你已安裝以下 Python 套件：

```bash
pip install numpy matplotlib open3d gtsam tqdm opencv-python scipy
```

---

## **使用方式**

### 1. 加載數據

數據讀取由 `load_data.py` 處理，使用 `load_dataset()` 來讀取不同的感測器數據。

```python
from load_data import load_dataset
data = load_dataset(dataset=20)
```

---

### 2. 計算機器人軌跡

`part1.py` 透過機器人的 **編碼器 + IMU** 計算機器人軌跡：

```python
from part1 import compute_trajectory
x, y, theta = compute_trajectory(FR, FL, RR, RL, yaw_rate, encoder_stamps, imu_stamps)
```

---

### 3. 雷射掃描匹配 (ICP)

執行 `scan_matching.py` 來使用 **ICP** 配準雷射掃描點雲：

```python
from scan_matching import ICP
T_icp = ICP(source_points, target_points, init_odometry=np.eye(3))
```

---

### 4. 佔據柵格地圖 (Occupancy Grid Map)

執行 `Occupancy.py` 建立機器人的佔據柵格地圖：

```bash
python Occupancy.py
```

---

### 5. 姿態圖優化 (Pose Graph Optimization)

執行 `pose_graph_optimization.py` 進行 **GTSAM** 姿態圖優化：

```bash
python pose_graph_optimization.py
```

---

### 6. 生成紋理地圖

執行 `texture.py` 來結合 RGB-D 圖像生成彩色地圖：

```bash
python texture.py
```

---

## **結果展示**

- **機器人軌跡 (Odometry + ICP)**\


- **佔據柵格地圖**\


- **紋理地圖**\


---

## **聯絡資訊**

如有任何問題，請聯絡 chl306\@ucsd.edu 或提交 issue。

