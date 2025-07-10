import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

current_time = time.time()
timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
time_folder = time.strftime("%m-%Y", time.localtime(current_time))
analysis_folder = f'Reports/analysis/{time_folder}'
os.makedirs(analysis_folder, exist_ok=True)



# Load data
visits = pd.read_csv(f'Reports/monthly/{time_folder}.csv', parse_dates=['date'])
sales = pd.read_csv(f'Reports/sales/sales_data_{time_folder}.csv', parse_dates=['date'])

# Step 1: Hitung total visitor per rak per hari
daily_visits = visits.groupby(['date', 'rak'])['total'].sum().reset_index()
daily_visits.rename(columns={'rak': 'category', 'total': 'total_visitors'}, inplace=True)

# Step 2: Hitung total sales per kategori per hari
daily_sales = sales.groupby(['date', 'category'])['units_sold'].sum().reset_index()
daily_sales.rename(columns={'units_sold': 'total_sales'}, inplace=True)

# Step 3: Gabungkan berdasarkan tanggal dan kategori
merged = pd.merge(daily_visits, daily_sales, how='left', on=['date', 'category'])
merged['total_sales'] = merged['total_sales'].fillna(0)

# Step 4: Hitung rasio konversi sederhana
merged['conversion_rate'] = merged['total_sales'] / merged['total_visitors']
merged['conversion_rate'] = merged['conversion_rate'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Step 5: Label peluang & produk wajib
visitor_threshold = merged['total_visitors'].quantile(0.75)
sales_threshold = merged['total_sales'].quantile(0.25)

def label_opportunity(row):
    if row['total_visitors'] >= visitor_threshold and row['total_sales'] <= sales_threshold:
        return 'Peluang Terlewatkan'
    elif row['total_visitors'] <= visitor_threshold and row['total_sales'] >= sales_threshold:
        return 'Produk Wajib'
    else:
        return 'Normal'

merged['insight'] = merged.apply(label_opportunity, axis=1)

# Step 6: Simpan hasil analisis ke CSV
output_csv = 'analisis_penjualan_vs_kunjungan.csv'
merged.to_csv(output_csv, index=False)
print(f"✅ Analisis disimpan di: {output_csv}")

# Step 7: Visualisasi line chart per kategori
categories = merged['category'].unique()
num_categories = len(categories)

# Setup plot
sns.set(style='whitegrid')
fig, axs = plt.subplots(num_categories, 1, figsize=(10, 5 * num_categories), sharex=True)

if num_categories == 1:
    axs = [axs]  # agar iterable jika cuma 1 subplot

for i, category in enumerate(categories):
    data = merged[merged['category'] == category]
    axs[i].plot(data['date'], data['total_visitors'], label='Visitors', color='blue', marker='o')
    axs[i].plot(data['date'], data['total_sales'], label='Sales', color='green', marker='x')
    axs[i].set_title(f'Kategori: {category}', fontsize=14)
    axs[i].set_ylabel('Jumlah')
    axs[i].legend()
    axs[i].grid(True)

plt.xlabel('Tanggal')
plt.tight_layout()

# Simpan plot ke file
plot_file = f'{analysis_folder}/analisis_plot.png'
plt.savefig(plot_file)
# Step 8: Pie chart demografi usia per kategori rak
age_columns = ['kid', 'teen', 'adult', 'elder']
age_data = visits.groupby('rak')[age_columns].sum().reset_index()

# Setup plot
num_pies = len(age_data)
fig, axs = plt.subplots(1, num_pies, figsize=(6 * num_pies, 6))

if num_pies == 1:
    axs = [axs]  # agar iterable jika cuma 1 pie

for i, row in age_data.iterrows():
    labels = ['Anak', 'Remaja', 'Dewasa', 'Lansia']
    values = [row['kid'], row['teen'], row['adult'], row['elder']]
    axs[i].pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    axs[i].axis('equal')  # bentuk lingkaran
    total_pengunjung = visits[visits['rak'] == row['rak']]['total'].sum()
    axs[i].set_title(f'Demografi Pengunjung - Rak: {row["rak"]} (Total: {total_pengunjung})', fontsize=14)

# Simpan pie chart ke file
plt.tight_layout()
pie_chart_file = f'{analysis_folder}/pie_chart_demografi_per_rak.png'
plt.savefig(pie_chart_file)
print(f"🥧 Pie chart demografi disimpan di: {pie_chart_file}")
print(f"📊 Gambar visualisasi disimpan di: {plot_file}")
