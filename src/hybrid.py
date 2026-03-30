def hybrid_predict(xgb_model, iso_model, scaler, X):

    X_scaled = scaler.transform(X)

    xgb_pred = xgb_model.predict(X)
    iso_pred = iso_model.predict(X_scaled)

    iso_pred = [1 if p == -1 else 0 for p in iso_pred]

    final_pred = []

    for i in range(len(X)):
        if xgb_pred[i] == 1:
            final_pred.append(1)  # Known attack
        elif iso_pred[i] == 1:
            final_pred.append(1)  # Zero-day anomaly
        else:
            final_pred.append(0)  # Normal

    return final_pred