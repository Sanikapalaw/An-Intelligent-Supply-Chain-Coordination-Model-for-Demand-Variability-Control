# 📋 DATACO COLUMN NAME MAPPING GUIDE

## Purpose
The code assumes certain column names. If your DataCo dataset has different names, use this guide to update the code.

---

## 🔑 CRITICAL COLUMNS (Required)

### 1. Order Quantity
**Code expects:** `order_item_quantity`  
**Your column might be:** `Quantity`, `Order Quantity`, `Item Quantity`, etc.

**Where to update:**
- Section 2.2 (line with `'order_item_quantity'`)
- Section 4.1 (bullwhip calculation)
- Section 5.1 (LSTM preparation)

**How to fix:**
```python
# If your column is named 'Quantity' instead:
df = df.rename(columns={'Quantity': 'order_item_quantity'})
```

### 2. Late Delivery Risk
**Code expects:** `late_delivery_risk`  
**Your column might be:** `Late Delivery Risk`, `Delivery Risk`, etc.

**Where to update:**
- Section 6.1 (XGBoost feature selection)

**How to fix:**
```python
df = df.rename(columns={'Late Delivery Risk': 'late_delivery_risk'})
```

### 3. Geographic Coordinates
**Code expects:** `latitude`, `longitude`  
**Your column might be:** `Lat`, `Long`, `Latitude`, `Longitude`, etc.

**Where to update:**
- Section 7.1 (K-means clustering)

**How to fix:**
```python
df = df.rename(columns={
    'Latitude': 'latitude',
    'Longitude': 'longitude'
})
```

### 4. Order Date
**Code expects:** `order_date_dateorders` or contains `'order_date'` + `'dateorders'`  
**Your column might be:** `Order Date`, `order date (DateOrders)`, etc.

**Where to update:**
- Section 3.1 (date feature extraction)
- Section 5.1 (time series preparation)

**How to fix:**
```python
# Find your date column name first
date_cols = [col for col in df.columns if 'date' in col.lower()]
print(date_cols)

# Then rename it
df = df.rename(columns={'Order Date': 'order_date_dateorders'})
```

---

## 📊 IMPORTANT COLUMNS (Recommended)

### 5. Shipping Days
**Code expects:** 
- `days_for_shipping_real` (actual)
- `days_for_shipment_scheduled` (scheduled)

**Your columns might be:**
- `Days for shipping (real)`
- `Days for shipment (scheduled)`

**How to fix:**
```python
df = df.rename(columns={
    'Days for shipping (real)': 'days_for_shipping_real',
    'Days for shipment (scheduled)': 'days_for_shipment_scheduled'
})
```

### 6. Sales
**Code expects:** `sales`  
**Your column might be:** `Sales`, `Total Sales`, `Revenue`, etc.

### 7. Shipping Mode
**Code expects:** `shipping_mode`  
**Your column might be:** `Shipping Mode`, `Ship Mode`, etc.

### 8. Customer Segment
**Code expects:** `customer_segment`  
**Your column might be:** `Customer Segment`, `Segment`, etc.

### 9. Market
**Code expects:** `market`  
**Your column might be:** `Market`, `Region`, etc.

### 10. Order Region
**Code expects:** `order_region`  
**Your column might be:** `Order Region`, `Region`, etc.

---

## 🛠️ BULK COLUMN RENAMING

If you have many columns to rename, use this template at the start of Section 1.3:

```python
# ═══════════════════════════════════════════════════════════
# CUSTOM COLUMN RENAMING (Add this after loading data)
# ═══════════════════════════════════════════════════════════

column_mapping = {
    # Format: 'Your Column Name': 'expected_name_in_code'
    
    # Critical columns
    'Order Item Quantity': 'order_item_quantity',
    'Late_delivery_risk': 'late_delivery_risk',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'order date (DateOrders)': 'order_date_dateorders',
    
    # Shipping columns
    'Days for shipping (real)': 'days_for_shipping_real',
    'Days for shipment (scheduled)': 'days_for_shipment_scheduled',
    'Shipping Mode': 'shipping_mode',
    
    # Business columns
    'Sales': 'sales',
    'Customer Segment': 'customer_segment',
    'Market': 'market',
    'Order Region': 'order_region',
    'Category Name': 'category_name',
    'Department Name': 'department_name',
    'Order Status': 'order_status',
    'Delivery Status': 'delivery_status',
    'Order Item Discount Rate': 'order_item_discount_rate',
    'Benefit per order': 'benefit_per_order',
    'Order Id': 'order_id',
    'Customer Id': 'customer_id',
    
    # Add more as needed...
}

# Apply renaming
df = df.rename(columns=column_mapping)

print("✅ Columns renamed successfully!")
print(f"\\nNew column names preview:")
print(df.columns.tolist()[:20])  # Show first 20
```

---

## 🔍 HOW TO FIND YOUR COLUMN NAMES

Run this code IMMEDIATELY after loading your data:

```python
# Display all column names
print("="*60)
print("YOUR DATASET COLUMNS:")
print("="*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:3d}. {col}")

# Check for key columns
print("\\n" + "="*60)
print("CHECKING FOR CRITICAL COLUMNS:")
print("="*60)

critical_checks = {
    'quantity': ['quantity', 'qty', 'order quantity', 'item quantity'],
    'risk': ['risk', 'late', 'delivery risk'],
    'coordinates': ['lat', 'long', 'latitude', 'longitude'],
    'date': ['date', 'time', 'order date'],
    'shipping': ['shipping', 'shipment', 'days'],
}

for category, keywords in critical_checks.items():
    matches = [col for col in df.columns if any(kw.lower() in col.lower() for kw in keywords)]
    print(f"\\n{category.upper()} columns found:")
    if matches:
        for match in matches:
            print(f"  ✅ {match}")
    else:
        print(f"  ❌ None found - please search manually")
```

---

## ⚡ QUICK FIX TEMPLATE

If you get errors about missing columns, add this debug cell:

```python
# ═══════════════════════════════════════════════════════════
# COLUMN DEBUGGING
# ═══════════════════════════════════════════════════════════

required_columns = {
    'order_item_quantity': None,
    'late_delivery_risk': None,
    'latitude': None,
    'longitude': None,
    'order_date_dateorders': None,
}

print("Checking required columns...\\n")

for required_col in required_columns.keys():
    if required_col in df.columns:
        print(f"✅ {required_col} - FOUND")
    else:
        print(f"❌ {required_col} - MISSING")
        
        # Try to find similar column
        similar = [col for col in df.columns if required_col.replace('_', ' ') in col.lower()]
        if similar:
            print(f"   💡 Did you mean: {similar[0]}?")
            print(f"   Fix: df = df.rename(columns={{'{similar[0]}': '{required_col}'}})")

print("\\n" + "="*60)
```

---

## 📝 COLUMN NAME STANDARDIZATION SCRIPT

Run this to auto-standardize all column names:

```python
# Make all columns lowercase and replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

print("✅ All columns standardized!")
print("\\nSample columns:")
print(df.columns.tolist()[:10])
```

---

## 🎯 COMMON SCENARIOS

### Scenario 1: "My dataset has 'Qty' instead of 'order_item_quantity'"
```python
df = df.rename(columns={'qty': 'order_item_quantity'})
```

### Scenario 2: "I don't have late_delivery_risk column"
```python
# Create it based on shipping delay
if 'days_for_shipping_real' in df.columns and 'days_for_shipment_scheduled' in df.columns:
    df['late_delivery_risk'] = (df['days_for_shipping_real'] > df['days_for_shipment_scheduled']).astype(int)
else:
    print("⚠️  Cannot create late_delivery_risk - skipping XGBoost section")
```

### Scenario 3: "My coordinates are in different columns"
```python
# If you have separate City/Country but no lat/long
# You'll need to geocode (not covered in this notebook)
# For now, you can skip the clustering section
```

---

## 💡 PRO TIP

Add this at the very beginning of your notebook (after loading data):

```python
# Save original column names for reference
original_columns = df.columns.tolist()

# Your renaming here...
# ...

# Compare before/after
print("COLUMN NAME CHANGES:")
print("="*60)
for old, new in zip(original_columns, df.columns):
    if old != new:
        print(f"{old:40s} → {new}")
```

---

## 🚨 IF ALL ELSE FAILS

Contact me with:
1. Output of `df.columns.tolist()`
2. Output of `df.head()`
3. Which section is failing

I'll provide exact column mapping for your dataset!

---

**Remember:** The code is flexible! As long as you map the column names correctly, everything will work. 🎯
