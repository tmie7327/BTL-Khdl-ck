# Web định giá chứng khoán

Ứng dụng web tĩnh để nhập dữ liệu công ty và ước tính giá trị nội tại cổ phiếu.

## Cách dùng

1. Mở `stock-valuation/index.html` trong trình duyệt.
2. Nhập Giá hiện tại, EPS, tốc độ tăng trưởng, hệ số P/E mục tiêu, tỷ lệ chiết khấu và số cổ phiếu lưu hành.
3. Nhấn `Tính giá trị` để xem kết quả.

## Mô hình định giá

- Giá trị theo P/E = EPS × P/E mục tiêu
- Giá trị theo DCF = Dòng tiền chiết khấu đến giá trị nội tại
- Giá trị trung bình = trung bình của hai mô hình trên
