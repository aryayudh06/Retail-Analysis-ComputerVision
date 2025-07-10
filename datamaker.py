import pandas as pd

# Baca file CSV
df = pd.read_csv("Reports/sales/sales.csv", parse_dates=['Order_Date'])

# Ganti nama kolom agar seragam
df.rename(columns={"Order_Date": "date", "Category": "category", "Price": "price"}, inplace=True)

# Filter hanya kategori yang diminta
target_categories = ["Men's Fashion", "Home & Kitchen", "Girls' Fashion"]
filtered = df[df['category'].isin(target_categories)]

# Hitung total penjualan per tanggal dan kategori
sales_summary = (
    filtered.groupby(['date', 'category'])['price']
    .sum()
    .reset_index()
    .pivot(index='date', columns='category', values='price')
    .fillna(0)
    .reset_index()
)

# Pastikan kolom muncul dalam urutan yang diminta
sales_summary = sales_summary[['date'] + target_categories]

# Simpan ke CSV
sales_summary.to_csv("penjualan_terpilih_per_tanggal.csv", index=False)

print("✅ CSV berhasil dibuat: penjualan_terpilih_per_tanggal.csv")
