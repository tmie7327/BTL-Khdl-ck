import pandas as pd
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu (đảm bảo file csv nằm cùng thư mục với file .py)
df = pd.read_csv('student-mat.csv', sep=';')

# 2. Xem 5 dòng đầu tiên
print("--- 5 dòng đầu của dữ liệu ---")
print(df.head())

# 3. Tính điểm trung bình G3 theo thời gian học tập (studytime)
# Đây là bài toán tính giá trị trung bình theo yêu cầu số 3 trong ảnh
avg_grades = df.groupby('studytime')['G3'].mean()
print("\n--- Điểm trung bình theo thời gian học ---")
print(avg_grades)

# 4. Vẽ biểu đồ đơn giản để đánh giá kết quả (Yêu cầu số 2)
avg_grades.plot(kind='bar', color='skyblue')
plt.title('Anh hung cua thoi gian hoc den diem cuoi ky')
plt.xlabel('Thoi gian hoc (1: thap - 4: cao)')
plt.ylabel('Diem trung binh G3')
plt.show()