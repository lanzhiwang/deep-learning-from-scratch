# coding: utf-8
from layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
print(apple_price)  # 200
price = mul_tax_layer.forward(apple_price, tax)
print(price)  # 220.00000000000003

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
print(dapple_price, dtax)  # 1.1 200
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num)  # 2.2 110.00000000000001

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)
