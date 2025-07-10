import csv
import random
from datetime import datetime, timedelta

# Konfigurasi item dan kategorinya
items = [
    ("Samsung Smart TV", "electronics"),
    ("Philips Rice Cooker", "electronics"),
    ("Fresh Apples", "fruits"),
    ("Aqua 600ml", "drinks"),
    ("Indomie Goreng", "instant_food"),
    ("Dove Shampoo", "personal_care"),
    ("Sari Roti", "bakery"),
    ("Ultra Milk", "dairy")
]

# Tentukan bulan dan tahun
month = 7
year = 2025
output_file = f"Reports/sales/{month:02d}-{year}.csv"

# Buat rentang tanggal 1 s/d akhir bulan
start_date = datetime(year, month, 1)
end_date = (start_date.replace(month=month + 1) if month < 12 else datetime(year + 1, 1, 1)) - timedelta(days=1)

# Simpan ke CSV
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["date", "item_name", "category", "units_sold"])
    
    current_date = start_date
    while current_date <= end_date:
        for item_name, category in items:
            units_sold = random.randint(0, 50)  # bisa ubah untuk variasi
            writer.writerow([current_date.strftime('%Y-%m-%d'), item_name, category, units_sold])
        current_date += timedelta(days=1)

print(f"✅ Dummy sales data saved as {output_file}")
