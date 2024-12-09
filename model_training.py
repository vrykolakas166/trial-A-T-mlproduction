import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import data_preprocessing

X_train, X_test, y_train, y_test = data_preprocessing.execute()

# Bắt đầu theo dõi với MLflow
with mlflow.start_run():
    # Khởi tạo mô hình
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42)
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    # Đánh giá mô hình
    report = classification_report(y_test, y_pred, output_dict=True)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Ghi lại các tham số mô hình và kết quả vào MLflow
    mlflow.log_param(model.get_params())
    
    mlflow.log_metric("accuracy", report['accuracy'])
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Lưu mô hình vào thư mục
    mlflow.sklearn.save_model(model, "fraud_detection_model_mlflow")

    print(report)