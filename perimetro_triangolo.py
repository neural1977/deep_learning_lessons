def perimetro_triangolo(a, b, c):
    return a + b + c


if __name__ == "__main__":
    a = float(input("Lato a: "))
    b = float(input("Lato b: "))
    c = float(input("Lato c: "))
    print(f"Perimetro: {perimetro_triangolo(a, b, c)}")
