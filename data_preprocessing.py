import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def execute(file_path='creditcard.csv',test_size_param=0.3):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(file_path)

    # Hiển thị 5 dòng đầu tiên để kiểm tra dữ liệu
    print("Dữ liệu đầu vào:")
    print(df.head())

    # Kiểm tra giá trị thiếu
    print("\nKiểm tra giá trị thiếu:")
    print(df.isnull().sum())

    # Chuẩn hóa cột 'Amount' và 'Time'
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    df['Time'] = scaler.fit_transform(df[['Time']])

    # Tách dữ liệu thành X và y
    X = df.drop(['Class'], axis=1)  # Xóa cột 'Class' để lấy tất cả đặc trưng
    y = df['Class']  # Lấy cột 'Class' làm nhãn

    # Chia tách dữ liệu thành tập huấn luyện (train) và tập kiểm thử (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_param, random_state=42)

    # In ra kích thước của dữ liệu huấn luyện và kiểm thử
    print("\nKích thước của tập huấn luyện và kiểm thử:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test
