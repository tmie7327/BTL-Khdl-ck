import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Cấu hình trang ứng dụng
st.set_page_config(
    page_title="Công Cụ Định Giá Cổ Phiếu 3 Năm",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Ứng Dụng Định Giá Cổ Phiếu Dự Phóng 3 Năm")
st.markdown("---")

# Tạo 2 cột: Cột bên trái nhập liệu, cột bên phải hiển thị kết quả
col_input, col_result = st.columns([1, 2])

with col_input:
    st.header("⚙️ Giả Định Đầu Vào")
    
    st.subheader("Thông tin hiện tại (Năm 0)")
    ticker = st.text_input("Mã cổ phiếu", value="ABC")
    current_eps = st.number_input("EPS hiện tại (VND/cổ phiếu)", min_value=0, value=5000, step=500)
    current_fcfe = st.number_input("FCFE hiện tại trên mỗi CP (VND)", min_value=0, value=4500, step=500)
    shares_outstanding = st.number_input("Số lượng CP lưu hành (Triệu CP)", min_value=1.0, value=100.0, step=1.0)
    
    st.subheader("Dự báo & Kỳ vọng (3 năm tới)")
    growth_rate = st.slider("Tốc độ tăng trưởng (%)", min_value=-20.0, max_value=50.0, value=15.0, step=0.5) / 100
    r_discount = st.slider("Tỷ lệ chiết khấu / Chi phí vốn Ke (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.5) / 100
    target_pe = st.number_input("P/E mục tiêu ở năm thứ 3", min_value=1.0, value=12.0, step=0.5)

# Xử lý tính toán logic định giá
years = [1, 2, 3]

# 1. Dự phóng EPS và Giá trị cuối cùng (Terminal Value) dựa trên P/E
projected_eps = [current_eps * ((1 + growth_rate) ** t) for t in years]
terminal_price = projected_eps[-1] * target_pe # Giá cổ phiếu ước tính ở cuối năm 3

# 2. Dự phóng FCFE và chiết khấu dòng tiền về hiện tại
projected_fcfe = [current_fcfe * ((1 + growth_rate) ** t) for t in years]
pv_fcfe = [fcfe / ((1 + r_discount) ** t) for t, fcfe in zip(years, projected_fcfe)]

# Chiết khấu giá trị cuối cùng về năm 0
pv_terminal_price = terminal_price / ((1 + r_discount) ** 3)

# Tổng giá trị nội tại của cổ phiếu hôm nay
intrinsic_value = sum(pv_fcfe) + pv_terminal_price

# Tạo bảng dữ liệu dự phóng để hiển thị
df_forecast = pd.DataFrame({
    "Năm Dự Phóng": [f"Năm {t}" for t in years],
    "EPS Dự Phóng (VND)": [round(eps) for eps in projected_eps],
    "FCFE Dự Phóng (VND)": [round(fcf) for fcf in projected_fcfe],
    "Giá Trị Hiện Tại FCFE (VND)": [round(pv) for pv in pv_fcfe]
})

with col_result:
    st.header(f"📊 Kết Quả Định Giá Cổ Phiếu {ticker.upper()}")
    
    # Hiển thị thẻ kết quả lớn
    st.metric(
        label="GIÁ TRỊ NỘI TẠI ƯỚC TÍNH (HÔM NAY)",
        value=f"{round(intrinsic_value):,} VND",
        help="Được tính bằng tổng Dòng tiền FCFE chiết khấu trong 3 năm + Giá trị cổ phiếu cuối năm 3 chiết khấu về hiện tại."
    )
    
    # Bảng chi tiết dòng tiền
    st.subheader("📋 Bảng Dự Phóng Dòng Tiền & Thu Nhập")
    st.dataframe(df_forecast, use_container_width=True, hide_index=True)
    
    # Thông tin bổ sung về năm thứ 3
    st.info(f"💡 **Giá mục tiêu cuối năm 3:** {round(terminal_price):,} VND (Dựa trên EPS Năm 3 là {round(projected_eps[-1]):,} VND và P/E = {target_pe})")
    
    # Vẽ đồ thị trực quan cấu thành giá trị cổ phiếu
    st.subheader("📈 Cấu Phần Giá Trị Nội Tại")
    
    categories = ['PV của FCFE Năm 1', 'PV của FCFE Năm 2', 'PV của FCFE Năm 3', 'PV của Giá Mục Tiêu Năm 3']
    values = pv_fcfe + [pv_terminal_price]
    
    fig = go.Figure(data=[go.Bar(
        x=categories, 
        y=values,
        text=[f"{round(v):,} VND" for v in values],
        textposition='auto',
        marker_color=['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c']
    )])
    
    fig.update_layout(
        ylabel_title="Giá trị (VND)",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Thêm ghi chú lý thuyết ngắn gọn
    with st.expander("📚 Xem giải thích về phương pháp định giá"):
        st.markdown("""
        Mô hình này kết hợp giữa cấu phần **Dòng tiền ngắn hạn (3 năm)** và **Kỳ vọng thị trường tương lai**:
        - **FCFE (Free Cash Flow to Equity)**: Là dòng tiền thực tế còn lại có thể chia cho cổ đông sau khi đã chi trả chi phí hoạt động, thuế và các nghĩa vụ nợ.
        - **Chiết khấu (Discounting)**: Vì tiền có giá trị theo thời gian, dòng tiền nhận được trong tương lai phải được giảm đi theo tỷ lệ chiết khấu $R_e$ (mức sinh lời kỳ vọng của bạn).
        - **Giá trị cuối cùng (Terminal Value)**: Sử dụng chỉ số P/E để ước tính xem vào cuối năm thứ 3, thị trường sẽ sẵn sàng trả bao nhiêu cho mức EPS lúc đó của doanh nghiệp.
        """)