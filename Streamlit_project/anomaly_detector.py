from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


class AnomalyDetector:
    def __init__(
        self,
        default_percentile: float = 90, # Thay đổi percentile của threshold tại đây. Sau đó chạy lại file save_model để train. 
        iso_n_estimators: int = 100,
        iso_contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.default_percentile_ = default_percentile

        self.iso_forest = IsolationForest(
            n_estimators=iso_n_estimators,
            contamination=iso_contamination,
            random_state=random_state,
        )

        self.is_fitted_ = False
        self.feature_columns_ = None

        # threshold cuối cùng để classify anomaly
        self.threshold_ = None

        # business bounds
        self.min_price_ = None
        self.max_price_ = None

        # residual stats
        self.res_mean_ = None
        self.res_std_ = None
        self.z_min_ = None
        self.z_max_ = None

        # percentile stats của giá train
        self.p10_ = None
        self.p90_ = None
        self.d_min_ = None
        self.d_max_ = None

        # isolation forest score stats
        self.if_score_min_ = None
        self.if_score_max_ = None

    def _check_fitted(self):
        required_attrs = [
            "feature_columns_",
            "threshold_",
            "min_price_",
            "max_price_",
            "res_mean_",
            "res_std_",
            "z_min_",
            "z_max_",
            "p10_",
            "p90_",
            "d_min_",
            "d_max_",
            "if_score_min_",
            "if_score_max_",
        ]

        if not self.is_fitted_:
            raise ValueError("Detector chưa được fit. Hãy gọi fit() trước.")

        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise ValueError(
                f"Detector thiếu thuộc tính sau khi load pkl: {missing}. "
                "Hãy train lại và dump lại file pkl bằng class mới."
            )

    @staticmethod
    def _safe_normalize(value, vmin, vmax):
        value = float(value)
        vmin = float(vmin)
        vmax = float(vmax)

        if vmax == vmin:
            return 0.0

        result = (value - vmin) / (vmax - vmin)
        return float(np.clip(result, 0.0, 1.0))

    def fit(self, df_X, y_true, y_pred, min_price, max_price):
        """
        Fit detector trên tập train:
        - fit IsolationForest
        - học các thống kê cần cho predict_one
        - tính threshold anomaly từ distribution train
        """
        if not isinstance(df_X, pd.DataFrame):
            raise TypeError("df_X phải là pandas DataFrame.")

        X_num = df_X.select_dtypes(include=[np.number]).copy()
        if X_num.empty:
            raise ValueError("Không có cột numeric để fit IsolationForest.")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(X_num) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError("df_X, y_true, y_pred phải có cùng số dòng.")

        self.feature_columns_ = X_num.columns.tolist()
        self.min_price_ = float(min_price)
        self.max_price_ = float(max_price)

        # 1) Fit IF
        self.iso_forest.fit(X_num)

        # 2) Residual stats
        residuals = y_true - y_pred
        self.res_mean_ = float(np.mean(residuals))
        self.res_std_ = float(np.std(residuals))

        if self.res_std_ == 0:
            z_scores = np.zeros_like(residuals, dtype=float)
        else:
            z_scores = np.abs((residuals - self.res_mean_) / self.res_std_)

        self.z_min_ = float(np.min(z_scores))
        self.z_max_ = float(np.max(z_scores))

        # 3) Price percentile stats
        self.p10_ = float(np.percentile(y_true, 10))
        self.p90_ = float(np.percentile(y_true, 90))

        d = np.zeros_like(y_true, dtype=float)

        mask_low = y_true < self.p10_
        mask_high = y_true > self.p90_

        d[mask_low] = np.abs(y_true[mask_low] - self.p10_)
        d[mask_high] = np.abs(y_true[mask_high] - self.p90_)

        self.d_min_ = float(np.min(d))
        self.d_max_ = float(np.max(d))

        # 4) Isolation Forest score stats
        train_if_scores = self.iso_forest.decision_function(X_num)
        self.if_score_min_ = float(np.min(train_if_scores))
        self.if_score_max_ = float(np.max(train_if_scores))

        # 5) Tính score trên train để lấy threshold
        s_resid_train = np.array(
            [self._calc_resid_one(actual, pred) for actual, pred in zip(y_true, y_pred)],
            dtype=float,
        )

        s_minmax_train = np.array(
            [self._calc_minmax_one(price) for price in y_true],
            dtype=float,
        )

        s_percentile_train = np.array(
            [self._calc_percentile_one(price) for price in y_true],
            dtype=float,
        )

        s_ml_train = np.array(
            [self._calc_ml_one(X_num.iloc[[i]]) for i in range(len(X_num))],
            dtype=float,
        )

        total_score = (
            0.4 * s_resid_train
            + 0.2 * s_minmax_train
            + 0.2 * s_percentile_train
            + 0.2 * s_ml_train
        )

        self.threshold_ = float(np.percentile(total_score, self.default_percentile_))
        self.is_fitted_ = True
        return self

    def _calc_resid_one(self, actual_price, predicted_price):
        residual = float(actual_price) - float(predicted_price)

        if self.res_std_ == 0:
            return 0.0

        z = abs((residual - self.res_mean_) / self.res_std_)

        if z > 3.0:
            return 1.0

        return self._safe_normalize(z, self.z_min_, self.z_max_)

    def _calc_minmax_one(self, actual_price):
        price = float(actual_price)
        return 1.0 if (price < self.min_price_ or price > self.max_price_) else 0.0

    def _calc_percentile_one(self, actual_price):
        price = float(actual_price)

        if price < self.p10_:
            d = abs(price - self.p10_)
        elif price > self.p90_:
            d = abs(price - self.p90_)
        else:
            d = 0.0

        return self._safe_normalize(d, self.d_min_, self.d_max_)

    def _calc_ml_one(self, x_input):
        if not isinstance(x_input, pd.DataFrame):
            raise TypeError("x_input phải là pandas DataFrame.")

        X_num = x_input[self.feature_columns_].copy()
        score = float(self.iso_forest.decision_function(X_num)[0])

        raw = self.if_score_max_ - score
        denom_max = self.if_score_max_ - self.if_score_min_

        return self._safe_normalize(raw, 0.0, denom_max)

    def get_anomaly_reason(self, row):
        reasons = []

        if row["S_MinMax"] > 0:
            reasons.append("Giá vượt trần/sàn khu vực")

        if row["S_Resid"] >= 0.6:
            reasons.append("Lệch lớn so với định giá máy học")

        if row["S_Percentile"] >= 0.6:
            reasons.append("Mức giá quá hiếm (ngoài P10-P90)")

        if row["S_ML"] >= 0.6:
            reasons.append("Cấu trúc thông tin nhà bất thường")

        if not reasons:
            reasons.append("Bất thường tổng hợp đa chiều")

        return " + ".join(reasons)

    def predict_one(self, x_input, actual_price, predicted_price):
        """
        Dự đoán anomaly cho 1 căn nhà.
        x_input: dict hoặc DataFrame 1 dòng
        """
        self._check_fitted()

        if isinstance(x_input, dict):
            x_input = pd.DataFrame([x_input])
        elif not isinstance(x_input, pd.DataFrame):
            raise TypeError("x_input phải là dict hoặc pandas DataFrame.")

        s_resid = self._calc_resid_one(actual_price, predicted_price)
        s_minmax = self._calc_minmax_one(actual_price)
        s_percentile = self._calc_percentile_one(actual_price)
        s_ml = self._calc_ml_one(x_input)

        composite_score = (
            0.4 * s_resid
            + 0.2 * s_minmax
            + 0.2 * s_percentile
            + 0.2 * s_ml
        )

        result = {
            "Gia_rao_ban_VND": float(actual_price),
            "Gia_du_doan_VND": float(predicted_price),
            "Chenh_lech_VND": float(actual_price) - float(predicted_price),
            "S_Resid": round(float(s_resid), 4),
            "S_MinMax": round(float(s_minmax), 4),
            "S_Percentile": round(float(s_percentile), 4),
            "S_ML": round(float(s_ml), 4),
            "Composite_Score": round(float(composite_score), 4),
            "Anomaly_Threshold": round(float(self.threshold_), 4),
        }

        result["Is_Anomaly"] = result["Composite_Score"] >= self.threshold_
        result["Li_do_nghi_van"] = self.get_anomaly_reason(result)

        return result

    def predict_batch(self, df_X, y_true, y_pred):
        """
        Dự đoán anomaly cho nhiều dòng.
        """
        self._check_fitted()

        if not isinstance(df_X, pd.DataFrame):
            raise TypeError("df_X phải là pandas DataFrame.")

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if len(df_X) != len(y_true) or len(y_true) != len(y_pred):
            raise ValueError("df_X, y_true, y_pred phải có cùng số dòng.")

        rows = []
        for i in range(len(df_X)):
            rows.append(
                self.predict_one(
                    x_input=df_X.iloc[[i]],
                    actual_price=y_true[i],
                    predicted_price=y_pred[i],
                )
            )

        return pd.DataFrame(rows) 