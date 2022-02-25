# Set up

```commandline
cd data_processing
pip install -r requirements.txt
```
## Tạo database:

```commandline
python database_migration.py
```

## Sử dụng hàm:
```commandline
from data_processing.data_processing import tên_hàm
```
get_faces: Lấy accountTitle và face vector <br>
get_infor: Lấy dữ   từ api <br>
send_data: Gửi dữ liệu check in lên server
