import pandas as pd

DATA_DIR = "data/"

def load_raw_data():
    orders = pd.read_csv(f"{DATA_DIR}olist_orders_dataset.csv")
    order_items = pd.read_csv(f"{DATA_DIR}olist_order_items_dataset.csv")
    products = pd.read_csv(f"{DATA_DIR}olist_products_dataset.csv")
    translation = pd.read_csv(f"{DATA_DIR}product_category_name_translation.csv")
    payments = pd.read_csv(f"{DATA_DIR}olist_order_payments_dataset.csv")
    
    print(f"orders:      {orders.shape}")
    print(f"order_items: {order_items.shape}")
    print(f"products:    {products.shape}")
    print(f"payments:    {payments.shape}")
    
    return orders, order_items, products, translation, payments

def transform(orders, order_items, products, translation, payments):
    # Clean layer
    products_traducidos = products.merge(translation, on='product_category_name', how='left')
    
    payments_agg = (
        payments
        .groupby('order_id', as_index=False)['payment_value']
        .sum()
    )
    
    # Fact table
    fact = (
        order_items
        .merge(orders, on='order_id', how='inner')
        .merge(products_traducidos, on='product_id', how='left')
        .merge(payments_agg, on='order_id', how='left')
    )
    
    # Flags
    fact['order_delivered_customer_date'] = pd.to_datetime(fact['order_delivered_customer_date'])
    fact['order_estimated_delivery_date'] = pd.to_datetime(fact['order_estimated_delivery_date'])
    fact['is_delivered'] = fact['order_status'] == 'delivered'
    fact['is_canceled'] = fact['order_status'].isin(['canceled', 'unavailable'])
    fact['is_on_time'] = (
        fact['order_delivered_customer_date'] <= fact['order_estimated_delivery_date']
    )
    
    print(f"fact table: {fact.shape}")
    return fact

def calculate_kpis(fact):
    total_orders = fact['order_id'].nunique()
    delivered_orders = fact[fact['is_delivered']]['order_id'].nunique()
    canceled_orders = fact[fact['is_canceled']]['order_id'].nunique()
    on_time_orders = fact[fact['is_on_time'] == True]['order_id'].nunique()

    kpis = {
        'gmv':                round(fact['price'].sum(), 2),
        'revenue':            round(fact['payment_value'].sum(), 2),
        'total_orders':       total_orders,
        'total_freight':      round(fact['freight_value'].sum(), 2),
        'cancellation_rate':  round(canceled_orders / total_orders * 100, 2),
        'on_time_delivery':   round(on_time_orders / delivered_orders * 100, 2),
        'aov':                round(fact['payment_value'].sum() / total_orders, 2),
    }

    return kpis
def check_data_quality(fact):
    print("\n=== CALIDAD DE DATOS ===\n")

    # Nulos por columna (solo las que importan)
    columnas_clave = [
        'order_id', 'product_id', 'price', 'freight_value',
        'payment_value', 'order_status', 'order_purchase_timestamp',
        'order_delivered_customer_date', 'order_estimated_delivery_date',
        'product_category_name_english'
    ]
    
    nulos = fact[columnas_clave].isnull().sum()
    nulos_pct = (nulos / len(fact) * 100).round(2)
    
    print("Nulos por columna:")
    print(pd.DataFrame({'cantidad': nulos, 'porcentaje': nulos_pct}))
    
    # Outliers en price usando IQR
    # IQR = rango intercuartil: la distancia entre el 25% y el 75% de los datos
    # Todo lo que está muy por encima o por debajo de ese rango es outlier
    Q1 = fact['price'].quantile(0.25)
    Q3 = fact['price'].quantile(0.75)
    IQR = Q3 - Q1
    outliers_price = fact[
        (fact['price'] < Q1 - 1.5 * IQR) |
        (fact['price'] > Q3 + 1.5 * IQR)
    ]
    print(f"\nOutliers en price:     {len(outliers_price)} filas ({round(len(outliers_price)/len(fact)*100, 2)}%)")
    print(f"Price máximo:          {fact['price'].max()}")
    print(f"Price promedio:        {round(fact['price'].mean(), 2)}")

    # Duplicados
    duplicados = fact.duplicated(subset=['order_id', 'order_item_id']).sum()
    print(f"\nFilas duplicadas:      {duplicados}")

    # Órdenes con payment_value en cero
    sin_pago = fact[fact['payment_value'] == 0]['order_id'].nunique()
    print(f"Órdenes sin pago:      {sin_pago}")

import numpy as np

def calcular_estadisticas(fact):
    print("\n=== ESTADÍSTICAS DE PRECIO ===\n")
    
    precios = fact['price'].values  # convierte la columna a array de NumPy
    
    print(f"Promedio:   {round(np.mean(precios), 2)}")
    print(f"Mediana:    {round(np.median(precios), 2)}")
    print(f"Mínimo:     {round(np.min(precios), 2)}")
    print(f"Máximo:     {round(np.max(precios), 2)}")
    print(f"Desv. std:  {round(np.std(precios), 2)}")


if __name__ == "__main__":
    orders, order_items, products, translation, payments = load_raw_data()
    fact = transform(orders, order_items, products, translation, payments)
    kpis = calculate_kpis(fact)
    print(kpis)
    check_data_quality(fact)
    calcular_estadisticas(fact)
