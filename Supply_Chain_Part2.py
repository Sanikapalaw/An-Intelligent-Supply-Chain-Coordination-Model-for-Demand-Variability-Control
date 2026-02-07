# CONTINUATION OF SUPPLY CHAIN COORDINATION SYSTEM NOTEBOOK
# PART 2: Advanced ML/DL Models and Analysis

# Add these cells to the existing notebook after Section 4

# ═══════════════════════════════════════════════════════════
# 5.2 CREATE SEQUENCES FOR LSTM
# ═══════════════════════════════════════════════════════════

def create_sequences(data, seq_length=7):
    """
    Create sequences for LSTM training
    
    Parameters:
    -----------
    data : array
        Time series data
    seq_length : int
        Length of input sequence
        
    Returns:
    --------
    X, y : arrays
        Input sequences and targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

if daily_demand is not None:
    print("Creating sequences for LSTM...\\n")
    
    # Use total_quantity as target
    data = daily_demand['total_quantity'].values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create sequences
    SEQ_LENGTH = 7  # Use past 7 days to predict next day
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    print(f"✅ Created sequences")
    print(f"   Input shape: {X.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Sequence length: {SEQ_LENGTH} days")
    
    # Train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\\n📊 Data Split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# ═══════════════════════════════════════════════════════════
# 5.3 BUILD LSTM MODEL
# ═══════════════════════════════════════════════════════════

if daily_demand is not None:
    print("\\nBuilding LSTM model...\\n")
    
    # Define LSTM architecture
    model_lstm = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu', return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Compile model
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Model summary
    print("="*60)
    print("🤖 LSTM MODEL ARCHITECTURE")
    print("="*60)
    model_lstm.summary()
    
    # Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    print("\\n🚀 Starting model training...")
    print("="*60)


# ═══════════════════════════════════════════════════════════
# 5.4 TRAIN LSTM MODEL
# ═══════════════════════════════════════════════════════════

if daily_demand is not None:
    # Train the model
    history = model_lstm.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print("\\n✅ Training complete!")
    
    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
    axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════
# 5.5 EVALUATE LSTM MODEL
# ═══════════════════════════════════════════════════════════

if daily_demand is not None:
    print("\\nEvaluating LSTM model...\\n")
    
    # Make predictions
    y_pred_train = model_lstm.predict(X_train)
    y_pred_test = model_lstm.predict(X_test)
    
    # Inverse transform predictions
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_train_inv = scaler.inverse_transform(y_pred_train)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
    train_mae = mean_absolute_error(y_train_inv, y_pred_train_inv)
    test_mae = mean_absolute_error(y_test_inv, y_pred_test_inv)
    train_r2 = r2_score(y_train_inv, y_pred_train_inv)
    test_r2 = r2_score(y_test_inv, y_pred_test_inv)
    
    # Calculate MAPE
    train_mape = np.mean(np.abs((y_train_inv - y_pred_train_inv) / (y_train_inv + 1e-6))) * 100
    test_mape = np.mean(np.abs((y_test_inv - y_pred_test_inv) / (y_test_inv + 1e-6))) * 100
    
    print("="*60)
    print("📊 LSTM MODEL PERFORMANCE")
    print("="*60)
    print("\\nTraining Set:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE: {train_mae:.2f}")
    print(f"  R²: {train_r2:.4f}")
    print(f"  MAPE: {train_mape:.2f}%")
    
    print("\\nTest Set:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE: {test_mae:.2f}")
    print(f"  R²: {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    
    # Accuracy percentage
    accuracy = 100 - test_mape
    print(f"\\n🎯 Forecast Accuracy: {accuracy:.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Training predictions
    axes[0].plot(y_train_inv, label='Actual', linewidth=2, alpha=0.7)
    axes[0].plot(y_pred_train_inv, label='Predicted', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Demand Quantity', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Training Set: Actual vs Predicted (R² = {train_r2:.4f})', 
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Test predictions
    axes[1].plot(y_test_inv, label='Actual', linewidth=2, alpha=0.7, color='green')
    axes[1].plot(y_pred_test_inv, label='Predicted', linewidth=2, alpha=0.7, color='red')
    axes[1].set_xlabel('Time', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Demand Quantity', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Test Set: Actual vs Predicted (R² = {test_r2:.4f}, MAPE = {test_mape:.2f}%)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    model_lstm.save('lstm_demand_forecasting_model.h5')
    joblib.dump(scaler, 'demand_scaler.pkl')
    print("\\n💾 Model saved as 'lstm_demand_forecasting_model.h5'")
    print("💾 Scaler saved as 'demand_scaler.pkl'")


# ═══════════════════════════════════════════════════════════
# SECTION 6: DELIVERY RISK PREDICTION (XGBOOST)
# ═══════════════════════════════════════════════════════════

print("\\n")
print("="*60)
print("SECTION 6: DELIVERY RISK PREDICTION (XGBOOST)")
print("="*60)


# ═══════════════════════════════════════════════════════════
# 6.1 FEATURE SELECTION FOR RISK PREDICTION
# ═══════════════════════════════════════════════════════════

print("\\nPreparing features for delivery risk prediction...\\n")

if 'late_delivery_risk' in df.columns:
    # Define feature columns
    feature_cols = []
    
    # Numerical features
    numerical_risk_features = [
        'days_for_shipment_scheduled',
        'order_item_quantity',
        'order_item_discount_rate',
        'sales',
        'benefit_per_order',
        'latitude',
        'longitude'
    ]
    feature_cols.extend([f for f in numerical_risk_features if f in df.columns])
    
    # Encoded categorical features
    encoded_features = [f for f in df.columns if f.endswith('_encoded')]
    feature_cols.extend(encoded_features)
    
    # Date features
    date_features = ['order_month', 'order_dayofweek', 'order_quarter', 'is_weekend']
    feature_cols.extend([f for f in date_features if f in df.columns])
    
    print(f"Selected {len(feature_cols)} features for risk prediction:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    
    # Create feature matrix
    X_risk = df[feature_cols].copy()
    y_risk = df['late_delivery_risk'].copy()
    
    # Handle missing values
    X_risk = X_risk.fillna(X_risk.median())
    
    print(f"\\n📊 Dataset shape: {X_risk.shape}")
    print(f"📊 Target distribution:")
    print(y_risk.value_counts())
    print(f"\\nClass balance:")
    print(f"  Low Risk (0): {(y_risk == 0).sum()} ({(y_risk == 0).sum()/len(y_risk)*100:.1f}%)")
    print(f"  High Risk (1): {(y_risk == 1).sum()} ({(y_risk == 1).sum()/len(y_risk)*100:.1f}%)")
else:
    print("⚠️  'late_delivery_risk' column not found")
    X_risk, y_risk = None, None


# ═══════════════════════════════════════════════════════════
# 6.2 TRAIN-TEST SPLIT & HANDLE CLASS IMBALANCE
# ═══════════════════════════════════════════════════════════

if X_risk is not None and y_risk is not None:
    print("\\nSplitting data and handling class imbalance...\\n")
    
    # Train-test split
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
        X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    print(f"Training set: {X_train_risk.shape[0]} samples")
    print(f"Test set: {X_test_risk.shape[0]} samples")
    
    # Check if imbalanced
    class_ratio = (y_train_risk == 1).sum() / (y_train_risk == 0).sum()
    
    if class_ratio < 0.5 or class_ratio > 2.0:
        print(f"\\n⚠️  Class imbalance detected (ratio: {class_ratio:.2f})")
        print("Applying SMOTE to balance classes...\\n")
        
        smote = SMOTE(random_state=42)
        X_train_risk_balanced, y_train_risk_balanced = smote.fit_resample(X_train_risk, y_train_risk)
        
        print(f"✅ After SMOTE:")
        print(f"   Training samples: {X_train_risk_balanced.shape[0]}")
        print(f"   Class 0: {(y_train_risk_balanced == 0).sum()}")
        print(f"   Class 1: {(y_train_risk_balanced == 1).sum()}")
    else:
        X_train_risk_balanced = X_train_risk
        y_train_risk_balanced = y_train_risk
        print("✅ Classes are reasonably balanced, no SMOTE needed")


# ═══════════════════════════════════════════════════════════
# 6.3 TRAIN XGBOOST MODEL
# ═══════════════════════════════════════════════════════════

if X_risk is not None:
    print("\\n🚀 Training XGBoost model...\\n")
    print("="*60)
    
    # Define XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    xgb_model.fit(
        X_train_risk_balanced, 
        y_train_risk_balanced,
        verbose=False
    )
    
    print("✅ XGBoost model trained successfully!")


# ═══════════════════════════════════════════════════════════
# 6.4 EVALUATE XGBOOST MODEL
# ═══════════════════════════════════════════════════════════

if X_risk is not None:
    print("\\n📊 Evaluating model performance...\\n")
    
    # Predictions
    y_pred_train = xgb_model.predict(X_train_risk)
    y_pred_test = xgb_model.predict(X_test_risk)
    y_pred_proba_test = xgb_model.predict_proba(X_test_risk)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train_risk, y_pred_train)
    test_acc = accuracy_score(y_test_risk, y_pred_test)
    precision = precision_score(y_test_risk, y_pred_test)
    recall = recall_score(y_test_risk, y_pred_test)
    f1 = f1_score(y_test_risk, y_pred_test)
    roc_auc = roc_auc_score(y_test_risk, y_pred_proba_test)
    
    print("="*60)
    print("🎯 XGBOOST MODEL PERFORMANCE")
    print("="*60)
    print(f"\\nTraining Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\\nClassification Metrics (Test Set):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    print("\\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_risk, y_pred_test, target_names=['Low Risk', 'High Risk']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_risk, y_pred_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test_risk, y_pred_proba_test)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════
# 6.5 FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════

if X_risk is not None:
    print("\\n📊 Analyzing feature importance...\\n")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualize top 15 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(top_features['feature'], top_features['importance'], color='steelblue', edgecolor='black')
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title('Top 15 Feature Importances for Delivery Risk Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model
    joblib.dump(xgb_model, 'xgboost_delivery_risk_model.pkl')
    joblib.dump(feature_cols, 'risk_prediction_features.pkl')
    print("\\n💾 XGBoost model saved as 'xgboost_delivery_risk_model.pkl'")
    print("💾 Feature list saved as 'risk_prediction_features.pkl'")


# ═══════════════════════════════════════════════════════════
# SECTION 7: GEOGRAPHIC ANALYSIS & CLUSTERING
# ═══════════════════════════════════════════════════════════

print("\\n")
print("="*60)
print("SECTION 7: GEOGRAPHIC ANALYSIS & CLUSTERING")
print("="*60)


# ═══════════════════════════════════════════════════════════
# 7.1 K-MEANS CLUSTERING BY GEOGRAPHIC LOCATION
# ═══════════════════════════════════════════════════════════

if 'latitude' in df.columns and 'longitude' in df.columns:
    print("\\nPerforming K-Means clustering on geographic locations...\\n")
    
    # Prepare geographic features
    geo_features = df[['latitude', 'longitude']].copy()
    geo_features = geo_features.dropna()
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(geo_features)
        inertias.append(kmeans_temp.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Inertia', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('clustering_elbow.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Choose optimal K (let's use 5 for this example)
    optimal_k = 5
    print(f"✅ Using K = {optimal_k} clusters\\n")
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(geo_features)
    
    # Add cluster labels back to dataframe
    df_geo = df.dropna(subset=['latitude', 'longitude']).copy()
    df_geo['cluster'] = cluster_labels
    
    # Analyze clusters
    print("="*60)
    print("📍 CLUSTER ANALYSIS")
    print("="*60)
    
    for cluster_id in range(optimal_k):
        cluster_data = df_geo[df_geo['cluster'] == cluster_id]
        print(f"\\nCluster {cluster_id}:")
        print(f"  Size: {len(cluster_data)} orders")
        print(f"  Center: ({kmeans.cluster_centers_[cluster_id][0]:.4f}, {kmeans.cluster_centers_[cluster_id][1]:.4f})")
        
        if 'late_delivery_risk' in cluster_data.columns:
            late_risk_pct = (cluster_data['late_delivery_risk'].sum() / len(cluster_data)) * 100
            print(f"  Late Delivery Risk: {late_risk_pct:.1f}%")
        
        if 'order_item_quantity' in cluster_data.columns:
            print(f"  Avg Order Quantity: {cluster_data['order_item_quantity'].mean():.2f}")
    
    # Visualize clusters
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(df_geo['longitude'], df_geo['latitude'], 
                         c=df_geo['cluster'], cmap='viridis', 
                         alpha=0.6, s=10, edgecolors='none')
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0],
               c='red', marker='X', s=300, edgecolors='black', linewidth=2,
               label='Cluster Centers')
    plt.xlabel('Longitude', fontsize=12, fontweight='bold')
    plt.ylabel('Latitude', fontsize=12, fontweight='bold')
    plt.title(f'Geographic Clustering (K={optimal_k})', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster ID')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('geographic_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save clustering results
    joblib.dump(kmeans, 'kmeans_geographic_model.pkl')
    df_geo.to_csv('clustered_supply_chain_data.csv', index=False)
    print("\\n💾 Clustering model saved as 'kmeans_geographic_model.pkl'")
    print("💾 Clustered data saved as 'clustered_supply_chain_data.csv'")
else:
    print("⚠️  Latitude/Longitude columns not found")


# ═══════════════════════════════════════════════════════════
# SECTION 8: REINFORCEMENT LEARNING (DQN)
# ═══════════════════════════════════════════════════════════

print("\\n")
print("="*60)
print("SECTION 8: REINFORCEMENT LEARNING (ORDER OPTIMIZATION)")
print("="*60)
print("\\nNote: This is a simplified DQN implementation for demonstration.")
print("For production use, consider more sophisticated implementations.\\n")

# Custom Supply Chain Environment
class SupplyChainEnv:
    """
    Simple supply chain environment for order quantity optimization
    Goal: Minimize (excess_inventory + stockout + bullwhip_effect)
    """
    def __init__(self, mean_demand=100, std_demand=20):
        self.mean_demand = mean_demand
        self.std_demand = std_demand
        self.inventory = 0
        self.max_steps = 100
        self.current_step = 0
        
    def reset(self):
        self.inventory = self.mean_demand
        self.current_step = 0
        return np.array([self.inventory, 0, 0])  # [inventory, prev_demand, prev_order]
    
    def step(self, order_quantity):
        # Simulate demand
        demand = max(0, np.random.normal(self.mean_demand, self.std_demand))
        
        # Update inventory
        self.inventory += order_quantity - demand
        
        # Calculate costs
        holding_cost = max(0, self.inventory) * 1.0  # Cost per unit held
        stockout_cost = max(0, -self.inventory) * 10.0  # Higher penalty for stockout
        variability_cost = abs(order_quantity - demand) * 0.5  # Bullwhip penalty
        
        # Total cost (negative reward)
        reward = -(holding_cost + stockout_cost + variability_cost)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        next_state = np.array([self.inventory, demand, order_quantity])
        
        return next_state, reward, done

# Simple DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train DQN Agent
print("🤖 Training DQN agent for order quantity optimization...\\n")

# Define action space (order quantities)
order_quantities = [50, 75, 100, 125, 150, 175, 200]
state_size = 3  # [inventory, prev_demand, prev_order]
action_size = len(order_quantities)

# Initialize environment and agent
env = SupplyChainEnv(mean_demand=100, std_demand=20)
agent = DQNAgent(state_size, action_size)

# Training
episodes = 100
batch_size = 32
scores = []

print("Training progress:")
for e in tqdm(range(episodes)):
    state = env.reset()
    total_reward = 0
    
    for time_step in range(env.max_steps):
        action = agent.act(state)
        order_qty = order_quantities[action]
        
        next_state, reward, done = env.step(order_qty)
        agent.remember(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    agent.replay(batch_size)
    scores.append(total_reward)
    
    if (e + 1) % 10 == 0:
        avg_score = np.mean(scores[-10:])
        print(f"Episode {e+1}/{episodes}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")

print("\\n✅ DQN training complete!")

# Plot training progress
plt.figure(figsize=(12, 6))
plt.plot(scores, alpha=0.6, label='Episode Score')
plt.plot(pd.Series(scores).rolling(10).mean(), linewidth=2, label='10-Episode MA')
plt.xlabel('Episode', fontsize=12, fontweight='bold')
plt.ylabel('Total Reward', fontsize=12, fontweight='bold')
plt.title('DQN Training Progress', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dqn_training.png', dpi=300, bbox_inches='tight')
plt.show()

# Save DQN model
agent.model.save('dqn_order_optimization_model.h5')
print("💾 DQN model saved as 'dqn_order_optimization_model.h5'")


# ═══════════════════════════════════════════════════════════
# SECTION 9: COORDINATION RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════

print("\\n")
print("="*60)
print("SECTION 9: SUPPLY CHAIN COORDINATION RECOMMENDATIONS")
print("="*60)

recommendations = []

# Recommendation 1: Information Sharing
if 'bullwhip_results' in locals() and amplification_ratios:
    avg_amplification = np.mean([r['ratio'] for r in amplification_ratios])
    if avg_amplification > 1.5:
        recommendations.append({
            'strategy': 'Implement Real-Time Information Sharing',
            'issue': f'High demand amplification ({avg_amplification:.2f}x)',
            'action': 'Deploy POS data sharing system with upstream partners',
            'expected_impact': f'Reduce amplification by 30-40% (to ~{avg_amplification * 0.65:.2f}x)',
            'difficulty': 'Medium',
            'priority': 'HIGH'
        })

# Recommendation 2: Shipping Mode Optimization
if 'late_delivery_risk' in df.columns and 'shipping_mode' in df.columns:
    risk_by_mode = df.groupby('shipping_mode')['late_delivery_risk'].mean()
    best_mode = risk_by_mode.idxmin()
    worst_mode = risk_by_mode.idxmax()
    
    recommendations.append({
        'strategy': f'Optimize Shipping Mode Selection',
        'issue': f'{worst_mode} has {risk_by_mode[worst_mode]*100:.1f}% late delivery rate',
        'action': f'Switch high-risk routes to {best_mode} ({risk_by_mode[best_mode]*100:.1f}% late rate)',
        'expected_impact': f'Reduce late deliveries by {(risk_by_mode[worst_mode] - risk_by_mode[best_mode])*100:.1f}%',
        'difficulty': 'Low',
        'priority': 'MEDIUM'
    })

# Recommendation 3: Order Batching
if 'order_item_quantity' in df.columns:
    qty_std = df['order_item_quantity'].std()
    qty_mean = df['order_item_quantity'].mean()
    cv = qty_std / qty_mean
    
    if cv > 1.0:
        recommendations.append({
            'strategy': 'Implement Smaller, Frequent Orders',
            'issue': f'High order variability (CV = {cv:.2f})',
            'action': 'Replace monthly orders with weekly orders, reduce batch sizes',
            'expected_impact': 'Reduce order variability by 40-50%',
            'difficulty': 'Medium',
            'priority': 'HIGH'
        })

# Recommendation 4: Discount Strategy
if 'order_item_discount_rate' in df.columns and 'order_item_quantity' in df.columns:
    discount_corr = df[['order_item_discount_rate', 'order_item_quantity']].corr().iloc[0, 1]
    
    if abs(discount_corr) > 0.3:
        recommendations.append({
            'strategy': 'Adopt Everyday Low Pricing (EDLP)',
            'issue': f'Discount-driven demand spikes (correlation: {discount_corr:.2f})',
            'action': 'Replace promotional pricing with stable, competitive prices',
            'expected_impact': 'Stabilize demand patterns, reduce forward buying by 25-35%',
            'difficulty': 'High',
            'priority': 'MEDIUM'
        })

# Recommendation 5: Lead Time Reduction
if 'days_for_shipping_real' in df.columns:
    avg_lead_time = df['days_for_shipping_real'].mean()
    if avg_lead_time > 3:
        recommendations.append({
            'strategy': 'Reduce Lead Times',
            'issue': f'Long average lead time ({avg_lead_time:.1f} days)',
            'action': 'Work with suppliers to reduce processing/shipping time',
            'expected_impact': f'Target: {avg_lead_time * 0.7:.1f} days (30% reduction)',
            'difficulty': 'High',
            'priority': 'MEDIUM'
        })

# Display recommendations
print("\\n🎯 TOP COORDINATION STRATEGIES:\\n")
print("="*60)

for i, rec in enumerate(recommendations, 1):
    print(f"\\n{i}. {rec['strategy']}")
    print(f"   Priority: {rec['priority']}")
    print(f"   Issue: {rec['issue']}")
    print(f"   Action: {rec['action']}")
    print(f"   Expected Impact: {rec['expected_impact']}")
    print(f"   Implementation Difficulty: {rec['difficulty']}")
    print(f"   {'-'*58}")

# Save recommendations
recommendations_df = pd.DataFrame(recommendations)
recommendations_df.to_csv('coordination_recommendations.csv', index=False)
print("\\n💾 Recommendations saved as 'coordination_recommendations.csv'")


# ═══════════════════════════════════════════════════════════
# SECTION 10: FINAL RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════

print("\\n")
print("="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print("\\n📊 MODEL PERFORMANCE SUMMARY:")
print("="*60)

if 'test_r2' in locals():
    print(f"\\n1. LSTM Demand Forecasting:")
    print(f"   - Test R²: {test_r2:.4f}")
    print(f"   - Test MAPE: {test_mape:.2f}%")
    print(f"   - Forecast Accuracy: {100-test_mape:.2f}%")

if 'test_acc' in locals():
    print(f"\\n2. XGBoost Delivery Risk Prediction:")
    print(f"   - Test Accuracy: {test_acc*100:.2f}%")
    print(f"   - Precision: {precision:.4f}")
    print(f"   - Recall: {recall:.4f}")
    print(f"   - ROC-AUC: {roc_auc:.4f}")

if 'total_amplification' in locals():
    print(f"\\n3. Bullwhip Effect Analysis:")
    print(f"   - Total Variance Amplification: {total_amplification:.2f}x")
    print(f"   - Supply Chain Tiers Analyzed: {len(existing_tiers)}")

if 'optimal_k' in locals():
    print(f"\\n4. Geographic Clustering:")
    print(f"   - Number of Clusters: {optimal_k}")
    print(f"   - Locations Analyzed: {len(geo_features)}")

print(f"\\n5. Reinforcement Learning:")
print(f"   - Training Episodes: {episodes}")
print(f"   - Final Avg Reward: {np.mean(scores[-10:]):.2f}")

print("\\n" + "="*60)
print("📁 SAVED FILES:")
print("="*60)
saved_files = [
    'processed_supply_chain_data.csv',
    'lstm_demand_forecasting_model.h5',
    'demand_scaler.pkl',
    'xgboost_delivery_risk_model.pkl',
    'risk_prediction_features.pkl',
    'kmeans_geographic_model.pkl',
    'dqn_order_optimization_model.h5',
    'coordination_recommendations.csv',
    'bullwhip_effect_analysis.png',
    'lstm_predictions.png',
    'xgboost_evaluation.png',
    'feature_importance.png',
    'geographic_clusters.png',
    'dqn_training.png'
]

for f in saved_files:
    print(f"  ✅ {f}")

print("\\n" + "="*60)
print("🎉 ANALYSIS COMPLETE!")
print("="*60)
print("\\nNext steps:")
print("  1. Review all visualizations and model performances")
print("  2. Examine coordination recommendations")
print("  3. Deploy models in Streamlit app for interactive predictions")
print("  4. Present findings to stakeholders/CEOs")
print("\\n" + "="*60)
