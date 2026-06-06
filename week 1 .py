import csv
from openpyxl import Workbook

INPUT_CSV = 'sales_journal.csv'
OUTPUT_XLSX = 'sales_report.xlsx'
VIP_THRESHOLD = 20_000_000


def main():
    rows = []
    with open(INPUT_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            r['UnitPrice'] = int(r['UnitPrice'])
            r['Quantity'] = int(r['Quantity'])
            r['Total'] = r['UnitPrice'] * r['Quantity']
            rows.append(r)

    total_revenue = sum(r['Total'] for r in rows)
    vip_rows = [r for r in rows if r['Total'] >= VIP_THRESHOLD]

    print(f"Total revenue: {total_revenue:,} VND")
    if vip_rows:
        print(f"\nVIP orders (Total >= {VIP_THRESHOLD:,} VND):")
        for r in vip_rows:
            print(f"{r['No']}. {r['Product']} - {r['Total']:,} VND")
    else:
        print("\nNo VIP orders found.")

    wb = Workbook()
    ws_all = wb.active
    ws_all.title = 'All'
    ws_all.append(['No', 'Product', 'UnitPrice', 'Quantity', 'Total'])
    for r in rows:
        ws_all.append([r['No'], r['Product'], r['UnitPrice'], r['Quantity'], r['Total']])

    ws_vip = wb.create_sheet('VIP')
    ws_vip.append(['No', 'Product', 'UnitPrice', 'Quantity', 'Total'])
    for r in vip_rows:
        ws_vip.append([r['No'], r['Product'], r['UnitPrice'], r['Quantity'], r['Total']])

    wb.save(OUTPUT_XLSX)
    print(f"\nReport written to {OUTPUT_XLSX}")


if __name__ == '__main__':
    main()
