def giai_thua(n):
    if n == 0: return 1
    return n * giai_thua(n-1)

print(f"Giai thừa của 5 là: {giai_thua(5)}")